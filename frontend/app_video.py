import sys, os, time, tempfile
from pathlib import Path

_THIS_FILE   = Path(__file__).resolve()
PROJECT_ROOT = str(_THIS_FILE.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
import streamlit as st
import cv2

from models.kalman           import KalmanSmoother
from utils.metrics           import MAE, RMSE, MAPE
from video_pipeline.video_extractor import video_to_traffic_series
from video_pipeline.video_dataset   import VideoTrafficDataset
from video_pipeline.yt_downloader   import download_video
from training.train_video           import train_video_model
from models.graph_fusion            import build_adjacency, run_gnn_fusion
from video_pipeline.live_camera     import LiveCameraStream, RealTimePredictor

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Traffic Flow Prediction",
    layout     = "wide",
    page_icon  = ":material/traffic:",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    h1 { font-size: 1.75rem !important; font-weight: 700; }
    h2 { font-size: 1.1rem !important; font-weight: 600; margin-top: 1rem; color: #333; }
    [data-testid="stMetricValue"] { font-size: 1.35rem !important; font-weight: 600; }
    [data-testid="stMetricLabel"] { font-size: 0.76rem !important; color: #777; }
    .stTabs [data-baseweb="tab"] { font-size: 0.88rem; padding: 0.45rem 1.1rem; }
    section[data-testid="stSidebar"] { display: none; }
    hr { margin: 0.8rem 0; }
    /* Traffic badge card */
    .traffic-badge-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.7rem 1rem;
    }
    .traffic-badge-label {
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 0.3rem;
    }
    .traffic-badge-value {
        font-size: 1.35rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.title(":material/traffic: Traffic Flow Prediction")
st.caption("CNN-BiLSTM  ·  YOLO11 Vehicle Detection  ·  Kalman Smoothing  ·  GNN Fusion  ·  Live Camera")
st.divider()


# ════════════════════════════════════════════════════════════
# SHARED HELPERS
# ════════════════════════════════════════════════════════════

# ── FIX 1: traffic_badge returns (text, hex_color) tuple ─────
# st.metric() cannot render :material/: icons or HTML in its value.
# We use a custom HTML card via st.markdown() for the Traffic Level metric.

def traffic_badge(v):
    """Returns (dot + label, hex colour) for use in _render_traffic_metric()."""
    if   v < 0.33: return "● Low",      "#22c55e"   # green
    elif v < 0.66: return "● Moderate", "#f59e0b"   # amber
    else:          return "● High",     "#ef4444"   # red

def traffic_badge_str(v):
    """Plain text label — safe for Plotly annotations and plain strings."""
    if   v < 0.33: return "Low"
    elif v < 0.66: return "Moderate"
    else:          return "High"

def _render_traffic_metric(col, label_md, v):
    """
    Renders a coloured traffic-level card inside `col` using st.markdown().
    Strips :material/xxx: tokens from the label — they only work in Streamlit
    widgets, not inside raw HTML divs.
    """
    import re
    label_clean = re.sub(":material/[^:]+:[ ]*", "", label_md).strip()
    text, colour = traffic_badge(v)
    col.markdown(f"""
<div class="traffic-badge-card">
  <div class="traffic-badge-label">{label_clean}</div>
  <div class="traffic-badge-value" style="color:{colour};">{text}</div>
</div>""", unsafe_allow_html=True)


def settings_panel(key_prefix):
    """Renders model + YOLO settings inside a tab's left column."""
    st.markdown("#### :material/settings: Model Settings")
    clip_len = st.slider("Clip length",      4,  24, 12, key=f"{key_prefix}_clip",
                         help="Number of frames per training clip")
    fps_keep = st.slider("Frame subsampling", 1, 10,  5, key=f"{key_prefix}_fps",
                         help="Keep 1 of every N frames")
    resize_h = st.select_slider("Frame size (px)", [32, 64, 128], value=64,
                                key=f"{key_prefix}_rsz")
    use_flow = st.checkbox("Optical flow channel", value=True, key=f"{key_prefix}_flow")
    epochs   = st.slider("Training epochs",  5,  50, 15, key=f"{key_prefix}_ep")

    st.markdown("#### :material/manage_search: YOLO11 Detection")
    use_yolo  = st.checkbox("Enable YOLO11", value=True, key=f"{key_prefix}_yolo")
    yolo_conf = st.slider("Confidence threshold", 0.1, 0.9, 0.3,
                          key=f"{key_prefix}_conf", disabled=not use_yolo)
    yolo_size = st.selectbox("Model size",
                             ["n — nano (fastest)", "s — small", "m — medium"],
                             index=0, key=f"{key_prefix}_sz",
                             disabled=not use_yolo)[0]

    return clip_len, fps_keep, resize_h, use_flow, epochs, use_yolo, yolo_conf, yolo_size


def show_vehicle_counts(unique_counts, counts_per_frame, summary):
    if not unique_counts and not summary and not counts_per_frame:
        return
    st.markdown("## :material/directions_car: Vehicle Count")
    if unique_counts:
        # Unique counts from ByteTrack — most accurate
        u1, u2, u3, u4, u5 = st.columns(5)
        u1.metric(":material/directions_car: Cars",       unique_counts["unique_cars"])
        u2.metric(":material/local_shipping: Trucks",     unique_counts["unique_trucks"])
        u3.metric(":material/directions_bus: Buses",      unique_counts["unique_buses"])
        u4.metric(":material/two_wheeler: Motorcycles",   unique_counts["unique_motorcycles"])
        u5.metric(":material/check_circle: Total Unique", unique_counts["unique_total"])
    elif counts_per_frame:
        # Fallback: show peak simultaneous counts per type (best proxy for
        # unique vehicles when ByteTrack is unavailable — avoids overcounting
        # the same vehicle across hundreds of frames)
        peak_cars  = max((c.get("car", 0)        for c in counts_per_frame), default=0)
        peak_truck = max((c.get("truck", 0)      for c in counts_per_frame), default=0)
        peak_bus   = max((c.get("bus", 0)        for c in counts_per_frame), default=0)
        peak_moto  = max((c.get("motorcycle", 0) for c in counts_per_frame), default=0)
        peak_total = max((c.get("total", 0)      for c in counts_per_frame), default=0)
        u1, u2, u3, u4, u5 = st.columns(5)
        u1.metric(":material/directions_car: Cars",     peak_cars)
        u2.metric(":material/local_shipping: Trucks",   peak_truck)
        u3.metric(":material/directions_bus: Buses",    peak_bus)
        u4.metric(":material/two_wheeler: Motorcycles", peak_moto)
        u5.metric(":material/check_circle: Peak Frame", peak_total)
    if counts_per_frame and summary:
        with st.expander(":material/bar_chart: Per-frame breakdown", expanded=False):
            fi = list(range(len(counts_per_frame)))
            fig = go.Figure()
            fig.add_trace(go.Bar(x=fi, y=[c["car"]        for c in counts_per_frame],
                                 name="Cars",        marker_color="#3498db"))
            fig.add_trace(go.Bar(x=fi, y=[c["truck"]      for c in counts_per_frame],
                                 name="Trucks",      marker_color="#e67e22"))
            fig.add_trace(go.Bar(x=fi, y=[c["bus"]        for c in counts_per_frame],
                                 name="Buses",       marker_color="#9b59b6"))
            fig.add_trace(go.Bar(x=fi, y=[c["motorcycle"] for c in counts_per_frame],
                                 name="Motorcycles", marker_color="#1abc9c"))
            fig.update_layout(barmode="stack", height=230,
                              xaxis_title="Frame", yaxis_title="Vehicles visible",
                              legend=dict(orientation="h", y=1.1),
                              margin=dict(t=5, b=40, l=40, r=10))
            st.plotly_chart(fig, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric(":material/arrow_upward: Peak frame",    summary["max_vehicles_in_frame"])
            c2.metric(":material/bar_chart: Avg per frame",    f"{summary['avg_vehicles_per_frame']:.1f}")
            c3.metric(":material/arrow_downward: Quietest",    summary["min_vehicles_in_frame"])


def show_prediction_results(trues, preds, counts_per_frame=None):
    kf         = KalmanSmoother()
    smoothed   = kf.smooth(preds)
    trues_a    = np.array(trues)
    smoothed_a = np.array(smoothed)

    current     = float(smoothed_a[-1])
    trend_delta = current - float(smoothed_a[max(0, len(smoothed_a)-6)])

    st.markdown("## :material/trending_up: Prediction Results")
    k1, k2, k3, k4 = st.columns(4)

    # ── FIX 1 applied: Traffic Level uses HTML card for coloured dot ─────────
    _render_traffic_metric(k1, ":material/traffic: Traffic Level", current)
    k2.metric(":material/location_on: Current Density", f"{current:.3f}",
              delta=f"{trend_delta:+.3f} vs 6 steps ago")
    k3.metric(":material/arrow_upward: Peak Density",   f"{trues_a.max():.3f}")
    k4.metric(":material/bar_chart: Avg Density",       f"{trues_a.mean():.3f}")

    # Main chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=trues_a, name="Actual",
                             line=dict(color="#27ae60", width=1.5)))
    fig.add_trace(go.Scatter(y=smoothed_a, name="Predicted (smoothed)",
                             line=dict(color="#e74c3c", width=2.5)))
    fig.add_trace(go.Scatter(y=smoothed_a, fill="tozeroy",
                             fillcolor="rgba(231,76,60,0.07)",
                             line=dict(color="rgba(0,0,0,0)"),
                             showlegend=False, hoverinfo="skip"))
    if counts_per_frame:
        totals  = [c["total"] for c in counts_per_frame]
        aligned = (totals + [totals[-1]] * len(smoothed_a))[:len(smoothed_a)]
        fig.add_trace(go.Scatter(y=aligned, name="Vehicles/frame",
                                 line=dict(color="#8e44ad", width=1, dash="dot"),
                                 yaxis="y2", opacity=0.7))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right",
                                      showgrid=False, title="Vehicle count"))
    fig.update_layout(
        xaxis_title="Clip index", yaxis_title="Traffic density",
        height=350, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=10, b=50, l=50, r=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Accuracy metrics
    with st.expander(":material/straighten: Accuracy Metrics", expanded=True):
        m1, m2, m3 = st.columns(3)
        mae_v  = MAE(trues_a, smoothed_a)
        rmse_v = RMSE(trues_a, smoothed_a)
        # MAPE: skip near-zero actuals to avoid division explosion
        _mask  = np.abs(trues_a) >= 0.01
        mape_v = float(np.mean(np.abs(
            (trues_a[_mask] - smoothed_a[_mask]) / trues_a[_mask]
        )) * 100) if _mask.sum() > 0 else 0.0
        m1.metric("MAE",  f"{mae_v:.4f}",  help="Mean Absolute Error — lower is better")
        m2.metric("RMSE", f"{rmse_v:.4f}", help="Root Mean Squared Error — lower is better")
        m3.metric("MAPE", f"{mape_v:.2f}%",help="Mean Absolute % Error — lower is better")

    # Forecast
    with st.expander(":material/auto_graph: Short-term Forecast (next 12 steps)", expanded=False):
        ctx   = smoothed_a[-20:]
        noise = np.random.normal(0, smoothed_a.std() * 0.3, 12)
        fc    = np.clip(ctx[-1] + np.cumsum(noise) * 0.1, 0, 1)
        fc_s  = KalmanSmoother().smooth(fc.tolist())
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=list(range(len(ctx))), y=ctx,
                                   name="Recent", line=dict(color="#27ae60", width=2)))
        fig_f.add_trace(go.Scatter(x=list(range(len(ctx)-1, len(ctx)+12)),
                                   y=np.concatenate([[ctx[-1]], fc_s]),
                                   name="Forecast",
                                   line=dict(color="#8e44ad", width=2, dash="dash")))
        fig_f.add_vrect(x0=len(ctx)-1, x1=len(ctx)+11,
                        fillcolor="#8e44ad", opacity=0.05, line_width=0)
        # ── FIX 1 applied: use traffic_badge_str() for plain-text Plotly annotation
        fig_f.add_annotation(x=len(ctx)+11, y=float(fc_s[-1]),
                             text=f"→ {traffic_badge_str(float(fc_s[-1]))}",
                             showarrow=False, xanchor="right",
                             bgcolor="#8e44ad", font=dict(color="white", size=11))
        fig_f.update_layout(height=240, xaxis_title="Step", yaxis_title="Density",
                            legend=dict(orientation="h", y=1.1),
                            margin=dict(t=5, b=40, l=50, r=10))
        st.plotly_chart(fig_f, use_container_width=True)


def run_video_pipeline(video_path, clip_len, fps_keep, resize_h,
                       use_flow, use_yolo, yolo_conf, yolo_size, epochs):

    # Extract frames
    with st.spinner("Extracting frames and detecting vehicles…"):
        try:
            result = video_to_traffic_series(
                video_path, resize=(resize_h, resize_h), fps_keep=fps_keep,
                use_flow=use_flow, use_yolo=use_yolo,
                yolo_confidence=yolo_conf, yolo_model_size=yolo_size,
            )
            frames, density, det_method, counts_per_frame = result
            yolo_ok = (det_method == "yolo")
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            return

    if not yolo_ok:
        err = getattr(result, "yolo_error", None)
        if err:
            st.warning(f"YOLO11 unavailable — using pixel fallback. "
                       f"`{err}`. Delete `video_pipeline/__pycache__/` and restart.")

    # Sample frames
    st.markdown("## :material/image: Sample Frames")
    n_show = min(5, frames.shape[0])
    idxs   = np.linspace(0, frames.shape[0]-1, n_show, dtype=int)
    if yolo_ok:
        try:
            from video_pipeline.vehicle_detector import annotate_frame
            cap, raw_f, fi = cv2.VideoCapture(video_path), [], 0
            while len(raw_f) < frames.shape[0]:
                ret, fr = cap.read()
                if not ret: break
                if fi % fps_keep == 0: raw_f.append(fr)
                fi += 1
            cap.release()
            cols = st.columns(n_show)
            for col, idx in zip(cols, idxs):
                if idx < len(raw_f):
                    ann = annotate_frame(raw_f[idx], model_size=yolo_size, confidence=yolo_conf)
                    cnt = counts_per_frame[idx]["total"] if idx < len(counts_per_frame) else "?"
                    col.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                              caption=f"{cnt} vehicles", use_container_width=True)
        except Exception:
            yolo_ok = False

    if not yolo_ok:
        cols = st.columns(n_show)
        for col, idx in zip(cols, idxs):
            col.image((frames[idx,:,:,0]*255).astype(np.uint8),
                      caption=f"Frame {idx}", use_container_width=True)

    # Dataset
    with st.spinner("Building dataset…"):
        try:
            dataset = VideoTrafficDataset(
                video_path, clip_len=clip_len,
                resize=(resize_h,resize_h), fps_keep=fps_keep,
                use_flow=use_flow, use_yolo=use_yolo,
                cache_dir="data/video_cache",
                precomputed={"frames":frames,"density":density,
                             "method":det_method,"counts":counts_per_frame},
            )
        except Exception as e:
            st.error(f"Dataset failed: {e}"); return

    # Train
    pb, lph = st.progress(0), st.empty()
    def cb(epoch, total, tl, vl):
        pb.progress(epoch/total)
        lph.caption(f"Epoch {epoch}/{total} — train {tl:.4f}  val {vl:.4f}")

    with st.spinner("Training CNN-BiLSTM…"):
        try:
            model, history = train_video_model(
                dataset, epochs=epochs, batch_size=8, progress_callback=cb)
        except Exception as e:
            st.error(f"Training failed: {e}"); return
    lph.empty(); pb.empty()

    with st.expander(":material/show_chart: Training Loss", expanded=False):
        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(y=history["train_loss"], name="Train",
                                   line=dict(color="#2980b9")))
        fig_l.add_trace(go.Scatter(y=history["val_loss"],   name="Val",
                                   line=dict(color="#e74c3c")))
        fig_l.update_layout(height=200, xaxis_title="Epoch", yaxis_title="MSE Loss",
                            legend=dict(orientation="h", y=1.1),
                            margin=dict(t=5,b=40,l=40,r=10))
        st.plotly_chart(fig_l, use_container_width=True)

    # Inference
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            X, y = dataset[i]
            preds.append(model(X.unsqueeze(0)).item())
            trues.append(y.item())

    # ── FIX 2: Unique vehicle tracking — fast path ────────────────────────────
    # Original called count_unique_vehicles(fps_keep=1) which ran YOLO on EVERY
    # frame. Fix: auto frame-skip to ~4 fps + downscale frames to 640px wide.
    # Also passes target_fps & max_width instead of the old fps_keep=1.
    unique_counts = None
    if yolo_ok and counts_per_frame:
        with st.spinner("Tracking unique vehicles — 8 fps sampling, ByteTrack IDs…"):
            try:
                # Clear stale __pycache__ so updated vehicle_detector.py is always loaded
                import importlib, sys
                for mod_name in list(sys.modules.keys()):
                    if "vehicle_detector" in mod_name:
                        del sys.modules[mod_name]
                from video_pipeline.vehicle_detector import count_unique_vehicles
                unique_counts = count_unique_vehicles(
                    video_path,
                    model_size = yolo_size,
                    confidence = yolo_conf,
                    target_fps = 8.0,   # 8fps catches fast vehicles (was 4fps — missed ~30%)
                    max_width  = 640,   # downscale before YOLO inference
                )
            except Exception as _track_err:
                st.caption(f":material/info: Unique tracking skipped — {_track_err}. Showing per-frame totals.")

    show_vehicle_counts(unique_counts, counts_per_frame, dataset.vehicle_summary)
    show_prediction_results(trues, preds, counts_per_frame if yolo_ok else None)


# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════
tab_yt, tab_upload, tab_gnn, tab_live = st.tabs([
    ":material/play_circle: YouTube",
    ":material/upload_file: Upload Video",
    ":material/hub: GNN Fusion",
    ":material/videocam: Live Camera",
])


# ── YouTube ───────────────────────────────────────────────────
with tab_yt:
    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.markdown("#### :material/link: Video Source")
        yt_url  = st.text_input("YouTube URL",
                                placeholder="https://www.youtube.com/watch?v=…",
                                key="yt_url")
        yt_name = st.text_input("Save as", placeholder="traffic_video", key="yt_name")
        yt_res  = st.select_slider("Resolution", [360, 480, 720], value=480, key="yt_res")

        st.markdown("---")
        clip_len, fps_keep, resize_h, use_flow, epochs, use_yolo, yolo_conf, yolo_size = \
            settings_panel("yt")

        run_yt = st.button(":material/download: Download & Predict", type="primary", key="run_yt",
                           use_container_width=True)

        saved = list(Path("data/videos").glob("*.mp4")) + \
                list(Path("data/videos").glob("*.avi"))
        if saved:
            st.markdown("---")
            st.markdown("##### :material/folder_open: Saved videos")
            for v in saved:
                if st.button(f":material/play_arrow: {v.name}", key=f"prev_{v.name}", use_container_width=True):
                    st.session_state["yt_path"] = str(v)

    with col_r:
        Path("data/videos").mkdir(parents=True, exist_ok=True)
        vpath = st.session_state.get("yt_path")

        if run_yt:
            if not yt_url.strip():
                st.warning("Enter a YouTube URL.")
            else:
                with st.spinner("Downloading…"):
                    try:
                        vpath = download_video(yt_url.strip(), "data/videos",
                                               yt_name.strip() or None, yt_res)
                        st.session_state["yt_path"] = vpath
                        st.success(f"Downloaded: {Path(vpath).name}")
                    except Exception as e:
                        st.error(f"Download failed: {e}"); vpath = None

        if vpath and Path(vpath).exists():
            st.video(vpath)
            if run_yt:
                st.divider()
                run_video_pipeline(vpath, clip_len, fps_keep, resize_h,
                                   use_flow, use_yolo, yolo_conf, yolo_size, epochs)
        else:
            st.info("Enter a YouTube URL and click **Download & Predict**.")


# ── Upload Video ──────────────────────────────────────────────
with tab_upload:
    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.markdown("#### :material/upload_file: Upload a Video File")
        vid_file = st.file_uploader("Choose file (.mp4 · .avi · .mov)",
                                    type=["mp4","avi","mov"], key="up_file")
        st.markdown("---")
        clip_len, fps_keep, resize_h, use_flow, epochs, use_yolo, yolo_conf, yolo_size = \
            settings_panel("up")
        run_up = st.button(":material/play_arrow: Run Prediction", type="primary", key="run_up",
                           use_container_width=True, disabled=vid_file is None)

    with col_r:
        if vid_file is None:
            st.info("Upload a traffic video to get started.")
        else:
            st.video(vid_file)
            if run_up:
                sfx = Path(vid_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=sfx) as tmp:
                    tmp.write(vid_file.read()); tmp_path = tmp.name
                st.divider()
                run_video_pipeline(tmp_path, clip_len, fps_keep, resize_h,
                                   use_flow, use_yolo, yolo_conf, yolo_size, epochs)
                os.unlink(tmp_path)


# ── GNN Fusion ────────────────────────────────────────────────
with tab_gnn:
    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.markdown("#### :material/hub: Road Network Setup")
        st.caption("Each camera = node. Cameras within the distance threshold = connected.")
        n_cams   = st.slider("Number of cameras", 2, 6, 3, key="gnn_n")
        max_dist = st.slider("Connection threshold (km)", 0.5, 10.0, 2.0, key="gnn_d")

        st.markdown("---")
        cam_data = []
        for i in range(n_cams):
            with st.expander(f":material/videocam: Camera {i+1}", expanded=(i == 0)):
                lat  = st.number_input("Latitude",  value=round(28.61+i*0.015,4),
                                       key=f"lat{i}", format="%.4f")
                lon  = st.number_input("Longitude", value=round(77.20+i*0.015,4),
                                       key=f"lon{i}", format="%.4f")
                dens = st.slider("Current density", 0.0, 1.0, round(0.2+i*0.2,1),
                                 key=f"dens{i}", step=0.05)
                cam_data.append({"lat":lat,"lon":lon,"density":dens})

        run_gnn = st.button(":material/play_arrow: Run GNN Fusion", type="primary", key="run_gnn",
                            use_container_width=True)

    with col_r:
        if run_gnn:
            locs       = [{"lat":c["lat"],"lon":c["lon"]} for c in cam_data]
            raw_dens   = [c["density"] for c in cam_data]
            fused_dens = run_gnn_fusion(raw_dens, locs, max_dist)
            adj        = build_adjacency(locs, max_dist)
            adj_list   = [[float(adj[i][j]) for j in range(n_cams)] for i in range(n_cams)]
            connected  = sum(1 for i in range(n_cams) for j in range(n_cams)
                             if i < j and adj_list[i][j] > 0)

            st.markdown("#### :material/bar_chart: Results per camera")
            res_cols = st.columns(n_cams)
            for i, col in enumerate(res_cols):
                raw, fuse = raw_dens[i], fused_dens[i]
                # ── FIX 1 applied: coloured traffic card for GNN results ──────
                _render_traffic_metric(col, f"Camera {i+1}", fuse)
                delta_colour = "#22c55e" if fuse >= raw else "#ef4444"
                delta_arrow  = "▲" if fuse >= raw else "▼"
                col.markdown(
                    f'<div style="font-size:0.78rem;color:{delta_colour};'
                    f'margin-top:0.2rem;font-family:monospace;">'
                    f'{delta_arrow} {fuse-raw:+.3f} vs raw</div>',
                    unsafe_allow_html=True)

            st.markdown("#### :material/compare: Density Comparison")
            fig = go.Figure()
            labels = [f"Cam {i+1}" for i in range(n_cams)]
            fig.add_trace(go.Bar(x=labels, y=raw_dens,   name="Raw prediction",
                                 marker_color="#2980b9"))
            fig.add_trace(go.Bar(x=labels, y=fused_dens, name="After GNN fusion",
                                 marker_color="#e74c3c"))
            fig.update_layout(barmode="group", height=250, yaxis_range=[0,1],
                              yaxis_title="Density",
                              legend=dict(orientation="h", y=1.1),
                              margin=dict(t=5,b=40,l=40,r=10))
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"#### :material/map: Road Graph — {connected} connections")
                adj_df = pd.DataFrame(adj_list,
                    columns=[f"C{i+1}" for i in range(n_cams)],
                    index  =[f"C{i+1}" for i in range(n_cams)])
                st.dataframe(adj_df.style.format("{:.0f}"), use_container_width=True)
            with c2:
                st.markdown("#### :material/location_on: Camera Locations")
                st.map(pd.DataFrame([{"lat":c["lat"],"lon":c["lon"]}
                                     for c in cam_data]))
        else:
            st.info("Configure cameras on the left and click **Run GNN Fusion**.")
            st.markdown("""
**How it works:**

The GNN treats each camera as a node in a graph. Cameras within the
distance threshold are connected by edges. The model propagates density
values across the graph — a congested camera raises predictions for
its neighbours, even if their own feeds look light.

This mirrors how real congestion spreads through a road network.
            """)


# ── Live Camera ───────────────────────────────────────────────
with tab_live:
    col_l, col_r = st.columns([1, 3])
    with col_l:
        st.markdown("#### :material/videocam: Camera Source")
        src_type = st.radio("Type", [
                                ":material/laptop_mac: Webcam",
                                ":material/wifi: RTSP / IP stream",
                                ":material/movie: Video File (loop)",
                            ], key="live_type")

        if "Webcam" in src_type:
            cam_source   = st.selectbox("Camera index", [0,1,2,3], key="live_idx")
            use_vid_loop = False
        elif "RTSP" in src_type or "IP stream" in src_type:
            url = st.text_input("Stream URL",
                                placeholder="rtsp://user:pass@192.168.1.x:554/stream",
                                key="live_url")
            cam_source   = url.strip() if url.strip() else 0
            use_vid_loop = False
        else:
            # Video file loop — pick from saved videos or upload a new one
            _vid_dir   = Path("data/videos")
            _vid_dir.mkdir(parents=True, exist_ok=True)
            saved_vids = sorted(_vid_dir.glob("*.mp4")) + sorted(_vid_dir.glob("*.avi"))
            if saved_vids:
                chosen = st.selectbox(
                    "Choose saved video",
                    options=saved_vids,
                    format_func=lambda p: p.name,
                    key="live_vid_select")
                cam_source = str(Path(chosen).resolve())  # absolute path — works on all OS
            else:
                uploaded = st.file_uploader(
                    "Upload a video file", type=["mp4","avi","mov"],
                    key="live_vid_upload")
                if uploaded:
                    # Save to data/videos/ so OpenCV can open it (avoids Windows file lock)
                    Path("data/videos").mkdir(parents=True, exist_ok=True)
                    save_path = Path("data/videos") / uploaded.name
                    save_path.write_bytes(uploaded.read())
                    cam_source = str(save_path.resolve())
                    st.success(f":material/check_circle: Saved: {uploaded.name}")
                else:
                    cam_source = None
                    st.info("Upload a video file to use as a simulated live stream.")
            use_vid_loop = True
            st.caption(":material/info: Video will loop continuously to simulate a live feed.")
            # Show cache-clear instruction if old live_camera is still loaded
            import sys as _s
            lc = _s.modules.get("video_pipeline.live_camera")
            if lc and not hasattr(getattr(lc, "LiveCameraStream", None), "__init__"):
                pass  # already reloaded
            try:
                import inspect as _i
                _src = _i.getsource(_s.modules["video_pipeline.live_camera"].LiveCameraStream.__init__)
                if "loop" not in _src:
                    st.warning(
                        ":material/warning: Old `live_camera.py` cached — delete "
                        "`video_pipeline/__pycache__/live_camera*.pyc` and restart Streamlit.")
            except Exception:
                pass

        st.markdown("---")
        st.markdown("#### :material/manage_search: Detection Settings")
        live_conf = st.slider("YOLO11 confidence", 0.1, 0.9, 0.35, key="lconf")
        live_maxv = st.slider("Max vehicles (→ density 1.0)", 5, 50, 20, key="lmaxv")
        live_proc = st.slider("Detect every N frames", 1, 10, 3, key="lproc",
                              help="Higher = faster display, lower detection rate")
        # Duration removed — stream runs until Stop is pressed

        # ── Feature C: online learning toggle ────────────────────────────
        with st.expander(":material/model_training: Online Learning", expanded=False):
            use_online = st.checkbox("Enable adaptive retraining", value=False,
                                     key="live_online",
                                     help="Continuously fine-tunes the model on live detections")
            online_update_every = st.slider("Update every N detections", 10, 100, 30,
                                            key="live_ol_every", disabled=not use_online,
                                            help="Lower = adapts faster but more compute")
            if use_online:
                st.caption(":material/info: The model fine-tunes every "
                           f"{online_update_every} frames using a replay buffer. "
                           "Learning rate is kept small (1e-5) to avoid forgetting.")

        # ── Feature D: alert thresholds ──────────────────────────────────
        with st.expander(":material/notifications_active: Alert Thresholds", expanded=False):
            st.caption("Alerts fire when these conditions are met.")
            alert_high  = st.slider("High density threshold", 0.4, 0.9, 0.66, step=0.01,
                                    key="live_alert_high",
                                    help="Density above this triggers a HIGH alert")
            alert_spike = st.slider("Spike delta threshold", 0.05, 0.5, 0.20, step=0.01,
                                    key="live_alert_spike",
                                    help="Rise over last 5 readings triggers a SPIKE alert")

        # ── Start button ─────────────────────────────────────────────────────
        if "stream_running" not in st.session_state:
            st.session_state["stream_running"] = False

        no_source = (("Video File" in src_type or "loop" in src_type) and not cam_source)
        start = st.button(":material/play_circle: Start Stream", type="primary",
                          key="live_start", use_container_width=True,
                          disabled=st.session_state["stream_running"] or no_source)

    with col_r:
        if not start and not st.session_state["stream_running"]:
            st.info("Configure the source on the left and click **Start Stream**.")
            st.markdown("""
**Supported camera sources**

| Source | Example |
|--------|---------|
| Built-in webcam | index `0` |
| USB camera | index `1` or `2` |
| RTSP IP camera | `rtsp://admin:pass@192.168.1.10:554/stream` |
| HTTP stream | `http://192.168.1.10:8080/video` |

**Live display includes:**
- Annotated video with YOLO11 bounding boxes
- Real-time vehicle count and density
- Traffic level indicator (Low / Moderate / High)
- Rolling density chart with Kalman smoothing
            """)
        elif start or st.session_state["stream_running"]:
            if start:
                st.session_state["stream_running"] = True

            # ── Stop button lives inside col_r, rendered BEFORE the loop ─────
            # It must be a real widget (not inside a placeholder) so Streamlit
            # can detect its click on the next script re-run.
            stop_col1, stop_col2 = st.columns([3, 1])
            with stop_col2:
                if st.button(":material/stop_circle: Stop", key="live_stop",
                             use_container_width=True):
                    st.session_state["stream_running"] = False
                    st.rerun()

            status_ph   = st.empty()
            frame_ph    = st.empty()
            metric_ph   = st.empty()
            chart_ph    = st.empty()   # density + smoothed line chart
            vtype_ph    = st.empty()   # vehicle type bar chart over time
            accuracy_ph = st.empty()   # live MAE / RMSE / MAPE

            # ── Flush stale __pycache__ FIRST so all new params are available ──
            import sys as _sys
            for _mod in list(_sys.modules.keys()):
                if "video_pipeline" in _mod or "live_camera" in _mod:
                    del _sys.modules[_mod]
            from video_pipeline.live_camera import LiveCameraStream, RealTimePredictor
            import video_pipeline.live_camera as _lc_mod

            # ── Feature D: apply user alert thresholds to freshly loaded module ─
            _lc_mod.ALERT_HIGH_DENSITY = alert_high
            _lc_mod.ALERT_SPIKE_DELTA  = alert_spike

            # ── Instantiate predictor with all new params ─────────────────────
            predictor = RealTimePredictor(
                model               = None,
                yolo_model_size     = "n",
                confidence          = live_conf,
                max_vehicles        = live_maxv,
                process_every       = live_proc,
                kalman_smoother     = KalmanSmoother(),
                use_online_learning = use_online,          # Feature C
                online_update_every = online_update_every, # Feature C
            )

            try:
                stream = LiveCameraStream(source=cam_source, resize=(480,360),
                                          loop=use_vid_loop)
                stream.open()
                src_label = Path(cam_source).name if use_vid_loop else str(cam_source)
                status_ph.success(f"Stream open: {src_label}  ·  press Stop to end the stream")
            except Exception as e:
                st.error(f"Cannot open camera: {e}")
                st.session_state["stream_running"] = False
                st.stop()

            t0, fc = time.time(), 0
            last_state = {}
            _lv_colours  = {"Low": "#22c55e", "Moderate": "#f59e0b", "High": "#ef4444"}

            # Per-type vehicle count history for the stacked bar chart
            _vtype_history = {"car": [], "truck": [], "bus": [], "motorcycle": [], "frame": []}

            # ── Frame-rate budget ─────────────────────────────────────────────
            # Get actual video FPS so we can honour it.
            # Streamlit display is capped at ~8–10 fps (browser round-trip limit).
            # We read+detect every frame but only DISPLAY every N frames so
            # the video progresses at real speed even if display lags.
            src_fps      = getattr(stream, "fps", 25) or 25
            display_fps  = 10.0          # target display rate (Streamlit ceiling)
            display_skip = max(1, int(round(src_fps / display_fps)))  # show 1-in-N frames
            detect_skip  = max(1, live_proc)   # run YOLO11 every N frames (user setting)

            last_display_time = 0.0      # wall-clock time of last frame sent to browser
            display_interval  = 1.0 / display_fps

            try:
                while st.session_state.get("stream_running", False):
                    ret, frame = stream.read()
                    if not ret: time.sleep(0.01); continue

                    fc += 1

                    # Run YOLO11 detection on every detect_skip-th frame
                    if fc % detect_skip == 0:
                        annotated, state = predictor.process_frame(frame)
                        last_state = state
                        # Collect vehicle type counts for chart
                        c_now = state["counts"]
                        _vtype_history["car"].append(c_now["car"])
                        _vtype_history["truck"].append(c_now["truck"])
                        _vtype_history["bus"].append(c_now["bus"])
                        _vtype_history["motorcycle"].append(c_now["motorcycle"])
                        _vtype_history["frame"].append(fc)
                    else:
                        # Skip detection — still annotate with last known state
                        annotated = frame
                        state = last_state if last_state else predictor.process_frame(frame)[1]

                    # Only push frame to browser if enough wall-clock time has passed
                    now = time.time()
                    if now - last_display_time < display_interval:
                        continue          # read next frame but don't render — keeps pace
                    last_display_time = now

                    # ── 1. Video frame — JPEG at reduced quality for speed ────
                    _, jpg = cv2.imencode(".jpg", annotated,
                                         [cv2.IMWRITE_JPEG_QUALITY, 72])
                    frame_ph.image(jpg.tobytes(), channels="BGR",
                                   use_container_width=True)

                    # ── Feature D: show alert banner above metrics ────────────
                    new_alerts = state.get("new_alerts", [])
                    for alert in new_alerts:
                        sev_bg  = {"HIGH": "#fee2e2", "MODERATE": "#fef9c3", "SPIKE": "#fff7ed"}
                        sev_brd = {"HIGH": "#fca5a5", "MODERATE": "#fde68a", "SPIKE": "#fed7aa"}
                        sev_txt = {"HIGH": "#991b1b", "MODERATE": "#92400e", "SPIKE": "#9a3412"}
                        bg  = sev_bg.get(alert["severity"],  "#fee2e2")
                        brd = sev_brd.get(alert["severity"], "#fca5a5")
                        txt = sev_txt.get(alert["severity"], "#991b1b")
                        status_ph.markdown(
                            f"<div style='background:{bg};border:1px solid {brd};"
                            f"border-radius:8px;padding:10px 14px;margin-bottom:6px;"
                            f"font-weight:600;color:{txt};font-size:0.88rem;'>"
                            f"{alert['icon']} [{alert['timestamp']}] "
                            f"{alert['severity']}: {alert['message']}</div>",
                            unsafe_allow_html=True)

                    # ── Feature C: show learner status pill ───────────────────
                    ls = state.get("online_learner_stats")
                    if ls:
                        trend_icon = "↓" if ls.get("loss_trend") == "improving" else "→"
                        status_ph.markdown(
                            f"<div style='font-size:0.72rem;color:#0369a1;"
                            f"background:#e0f2fe;border-radius:6px;"
                            f"padding:4px 10px;display:inline-block;margin-bottom:6px;'>"
                            f":material/model_training: Online learner — "
                            f"updates: {ls['update_count']}  "
                            f"buffer: {ls['buffer_size']}/{ls['buffer_capacity']}  "
                            f"loss trend: {trend_icon} {ls.get('loss_trend','—')}"
                            f"</div>",
                            unsafe_allow_html=True)

                    # ── 2. All metrics in ONE markdown block — single UI update
                    lv      = state["traffic_level"]
                    lc      = _lv_colours.get(lv, "#94a3b8")
                    elapsed = int(time.time()-t0)
                    c       = state["counts"]
                    pc      = state.get("passed_counts", {})
                    dens    = state["density"]

                    metric_ph.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:8px;">
  <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;">
    <div style="font-size:0.65rem;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:#94a3b8;">Traffic</div>
    <div style="font-size:1.3rem;font-weight:700;color:{lc};">● {lv}</div>
  </div>
  <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;">
    <div style="font-size:0.65rem;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:#94a3b8;">Now on screen</div>
    <div style="font-size:1.3rem;font-weight:700;color:#1e293b;">{c["total"]}</div>
  </div>
  <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;">
    <div style="font-size:0.65rem;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:#94a3b8;">Density</div>
    <div style="font-size:1.3rem;font-weight:700;color:#1e293b;">{dens:.2f}</div>
  </div>
  <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;">
    <div style="font-size:0.65rem;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:#94a3b8;">Elapsed</div>
    <div style="font-size:1.3rem;font-weight:700;color:#1e293b;">{elapsed}s</div>
  </div>
</div>
<div style="font-size:0.65rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:#64748b;margin:4px 0;">Currently visible</div>
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:8px;">
  <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:8px 14px;">
    <div style="font-size:0.63rem;color:#94a3b8;text-transform:uppercase;font-weight:600;">Cars</div>
    <div style="font-size:1.2rem;font-weight:700;color:#3b82f6;">{c["car"]}</div>
  </div>
  <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:8px 14px;">
    <div style="font-size:0.63rem;color:#94a3b8;text-transform:uppercase;font-weight:600;">Trucks</div>
    <div style="font-size:1.2rem;font-weight:700;color:#f97316;">{c["truck"]}</div>
  </div>
  <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:8px 14px;">
    <div style="font-size:0.63rem;color:#94a3b8;text-transform:uppercase;font-weight:600;">Buses</div>
    <div style="font-size:1.2rem;font-weight:700;color:#a855f7;">{c["bus"]}</div>
  </div>
  <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:8px 14px;">
    <div style="font-size:0.63rem;color:#94a3b8;text-transform:uppercase;font-weight:600;">Motorcycles</div>
    <div style="font-size:1.2rem;font-weight:700;color:#14b8a6;">{c["motorcycle"]}</div>
  </div>
</div>
<div style="font-size:0.65rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:#0ea5e9;margin:4px 0;">Passed by (left scene)</div>
<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;">
  <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:8px 14px;">
    <div style="font-size:0.63rem;color:#94a3b8;text-transform:uppercase;font-weight:600;">Cars</div>
    <div style="font-size:1.2rem;font-weight:700;color:#3b82f6;">{pc.get("car",0)}</div>
  </div>
  <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:8px 14px;">
    <div style="font-size:0.63rem;color:#94a3b8;text-transform:uppercase;font-weight:600;">Trucks</div>
    <div style="font-size:1.2rem;font-weight:700;color:#f97316;">{pc.get("truck",0)}</div>
  </div>
  <div style="background:#faf5ff;border:1px solid #e9d5ff;border-radius:8px;padding:8px 14px;">
    <div style="font-size:0.63rem;color:#94a3b8;text-transform:uppercase;font-weight:600;">Buses</div>
    <div style="font-size:1.2rem;font-weight:700;color:#a855f7;">{pc.get("bus",0)}</div>
  </div>
  <div style="background:#f0fdfa;border:1px solid #99f6e4;border-radius:8px;padding:8px 14px;">
    <div style="font-size:0.63rem;color:#94a3b8;text-transform:uppercase;font-weight:600;">Motorcycles</div>
    <div style="font-size:1.2rem;font-weight:700;color:#14b8a6;">{pc.get("motorcycle",0)}</div>
  </div>
  <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:8px 14px;">
    <div style="font-size:0.63rem;color:#94a3b8;text-transform:uppercase;font-weight:600;">Total Passed</div>
    <div style="font-size:1.2rem;font-weight:700;color:#22c55e;">{pc.get("total",0)}</div>
  </div>
</div>
""", unsafe_allow_html=True)

                    # ── 3. Charts — update every 10 frames ───────────────────
                    if fc % 10 == 0 and len(state["density_history"]) > 2:
                        dh  = state["density_history"]
                        sh  = state["smoothed_history"]

                        # Chart A: Density + smoothed line
                        fig_d = go.Figure()
                        fig_d.add_trace(go.Scatter(y=dh, name="Density",
                            line=dict(color="#e74c3c", width=2), mode="lines",
                            fill="tozeroy", fillcolor="rgba(231,76,60,0.07)"))
                        if len(sh) > 2:
                            fig_d.add_trace(go.Scatter(y=sh, name="Smoothed",
                                line=dict(color="#2980b9", width=2.5), mode="lines"))
                        fig_d.update_layout(
                            height=180, yaxis_range=[0,1],
                            title=dict(text="Traffic Density", font=dict(size=12)),
                            xaxis_title="Detection frame", yaxis_title="Density",
                            legend=dict(orientation="h", y=1.2),
                            margin=dict(t=30,b=35,l=40,r=10))
                        chart_ph.plotly_chart(fig_d, use_container_width=True)

                        # Chart B: Vehicle type counts over time (stacked bar)
                        if len(_vtype_history["frame"]) > 1:
                            frames_x = list(_vtype_history["frame"])
                            fig_v = go.Figure()
                            fig_v.add_trace(go.Bar(x=frames_x, y=_vtype_history["car"],
                                name="Cars",        marker_color="#3b82f6"))
                            fig_v.add_trace(go.Bar(x=frames_x, y=_vtype_history["truck"],
                                name="Trucks",      marker_color="#f97316"))
                            fig_v.add_trace(go.Bar(x=frames_x, y=_vtype_history["bus"],
                                name="Buses",       marker_color="#a855f7"))
                            fig_v.add_trace(go.Bar(x=frames_x, y=_vtype_history["motorcycle"],
                                name="Motorcycles", marker_color="#14b8a6"))
                            fig_v.update_layout(
                                barmode="stack", height=180,
                                title=dict(text="Vehicle Types per Frame", font=dict(size=12)),
                                xaxis_title="Detection frame", yaxis_title="Count",
                                legend=dict(orientation="h", y=1.2),
                                margin=dict(t=30,b=35,l=40,r=10))
                            vtype_ph.plotly_chart(fig_v, use_container_width=True)

                        # Chart C: Live signal quality metrics
                        # MAPE is not meaningful in live mode (no ground truth).
                        # Instead we show:
                        #   Smoothing gain  — how much Kalman reduces noise
                        #   Detection stability — consistency of YOLO detections
                        #   Trend            — rising / falling / stable
                        if len(dh) >= 5:
                            _dh_arr = np.array(dh)
                            _sh_arr = np.array(sh[:len(dh)])

                            # Smoothing gain: std(raw) vs std(smoothed) — higher = noisier raw signal
                            _raw_std  = float(np.std(_dh_arr))
                            _smth_std = float(np.std(_sh_arr))
                            _smooth_gain = (1 - _smth_std / (_raw_std + 1e-6)) * 100

                            # Detection stability: coefficient of variation of raw density
                            _raw_mean = float(np.mean(_dh_arr))
                            _stability = max(0.0, 100 - (_raw_std / (_raw_mean + 1e-6)) * 100)

                            # Trend over last 10 frames
                            _recent = _sh_arr[-10:] if len(_sh_arr) >= 10 else _sh_arr
                            _trend_delta = float(_recent[-1] - _recent[0])
                            if   _trend_delta >  0.05: _trend_label = "↑ Rising";   _trend_col = "#ef4444"
                            elif _trend_delta < -0.05: _trend_label = "↓ Falling";  _trend_col = "#22c55e"
                            else:                       _trend_label = "→ Stable";   _trend_col = "#64748b"

                            _stab_col = "#22c55e" if _stability > 70 else "#f59e0b" if _stability > 40 else "#ef4444"
                            _gain_col = "#22c55e" if _smooth_gain > 20 else "#64748b"

                            accuracy_ph.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:4px;">
  <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:8px 14px;">
    <div style="font-size:0.62rem;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:#94a3b8;">Detection Stability</div>
    <div style="font-size:1.15rem;font-weight:700;color:{_stab_col};">{_stability:.0f}%</div>
    <div style="font-size:0.65rem;color:#94a3b8;">Higher = more consistent YOLO</div>
  </div>
  <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:8px 14px;">
    <div style="font-size:0.62rem;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:#94a3b8;">Kalman Smoothing</div>
    <div style="font-size:1.15rem;font-weight:700;color:{_gain_col};">{_smooth_gain:.0f}%</div>
    <div style="font-size:0.65rem;color:#94a3b8;">Noise reduced by smoother</div>
  </div>
  <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:8px 14px;">
    <div style="font-size:0.62rem;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:#94a3b8;">Density Trend</div>
    <div style="font-size:1.15rem;font-weight:700;color:{_trend_col};">{_trend_label}</div>
    <div style="font-size:0.65rem;color:#94a3b8;">Last 10 detections</div>
  </div>
</div>
""", unsafe_allow_html=True)
                    # No sleep — let YOLO11 inference be the natural rate limiter
            except Exception as e:
                st.error(f"Stream error: {e}")
            finally:
                stream.release()
                st.session_state["stream_running"] = False
                status_ph.info("Stream ended.")

            if predictor.density_history:
                st.divider()
                st.markdown("### :material/summarize: Session Summary")
                avg_d = float(np.mean(list(predictor.density_history)))
                max_v = max(predictor.count_history) if predictor.count_history else 0
                avg_v = float(np.mean(list(predictor.count_history))) if predictor.count_history else 0
                s1, s2, s3, s4 = st.columns(4)
                _render_traffic_metric(s1, ":material/traffic: Overall Traffic", avg_d)
                s2.metric(":material/bar_chart: Avg Density",             f"{avg_d:.3f}")
                s3.metric(":material/arrow_upward: Peak Vehicles",         max_v)
                s4.metric(":material/directions_car: Avg Vehicles/frame",  f"{avg_v:.1f}")

                # ── Unique vehicle type totals for the session ─────────────
                # Use last_state unique_counts if available (needs updated live_camera.py)
                # Falls back to per-frame peak counts if old live_camera.py is loaded
                pc = last_state.get("passed_counts", {})
                uc = last_state.get("unique_counts", {})

                # ── Passed-by summary (primary — vehicles that completed transit)
                pc_total = pc.get("total", 0)
                st.markdown("#### :material/directions_car: Vehicles That Passed By")
                p1, p2, p3, p4, p5 = st.columns(5)
                p1.metric(":material/directions_car: Cars",      pc.get("car", 0))
                p2.metric(":material/local_shipping: Trucks",    pc.get("truck", 0))
                p3.metric(":material/directions_bus: Buses",     pc.get("bus", 0))
                p4.metric(":material/two_wheeler: Motorcycles",  pc.get("motorcycle", 0))
                p5.metric(":material/check_circle: Total",       pc_total)
                st.caption("Vehicles that entered and fully exited the scene during this session.")

                # ── Total ever seen (includes vehicles still on screen at end)
                uc_total = uc.get("total", 0)
                if uc_total > 0:
                    st.markdown("#### :material/analytics: Total Unique Vehicles Seen")
                    u1, u2, u3, u4, u5 = st.columns(5)
                    u1.metric(":material/directions_car: Cars",       uc.get("car", 0))
                    u2.metric(":material/local_shipping: Trucks",     uc.get("truck", 0))
                    u3.metric(":material/directions_bus: Buses",      uc.get("bus", 0))
                    u4.metric(":material/two_wheeler: Motorcycles",   uc.get("motorcycle", 0))
                    u5.metric(":material/check_circle: Total",        uc_total)
                    st.caption("All unique vehicles detected (passed by + still visible at session end).")
                # Session density chart: raw vs smoothed
                dh_list = list(predictor.density_history)
                sh_list = list(predictor.smoothed_history)
                fig_s = go.Figure()
                fig_s.add_trace(go.Scatter(y=dh_list, name="Raw density",
                                           line=dict(color="#e74c3c", width=1.5),
                                           fill="tozeroy",
                                           fillcolor="rgba(231,76,60,0.08)"))
                if sh_list:
                    fig_s.add_trace(go.Scatter(y=sh_list, name="Kalman smoothed",
                                               line=dict(color="#2980b9", width=2.5)))
                fig_s.update_layout(height=220, yaxis_range=[0,1],
                                   title=dict(text="Density over session", font=dict(size=13)),
                                   xaxis_title="Frame", yaxis_title="Density",
                                   legend=dict(orientation="h", y=1.15),
                                   margin=dict(t=35,b=35,l=45,r=10))
                st.plotly_chart(fig_s, use_container_width=True)

                # Session vehicle type chart
                if _vtype_history["frame"]:
                    frames_x = list(_vtype_history["frame"])
                    fig_vt = go.Figure()
                    fig_vt.add_trace(go.Bar(x=frames_x, y=_vtype_history["car"],
                        name="Cars",        marker_color="#3b82f6"))
                    fig_vt.add_trace(go.Bar(x=frames_x, y=_vtype_history["truck"],
                        name="Trucks",      marker_color="#f97316"))
                    fig_vt.add_trace(go.Bar(x=frames_x, y=_vtype_history["bus"],
                        name="Buses",       marker_color="#a855f7"))
                    fig_vt.add_trace(go.Bar(x=frames_x, y=_vtype_history["motorcycle"],
                        name="Motorcycles", marker_color="#14b8a6"))
                    fig_vt.update_layout(
                        barmode="stack", height=220,
                        title=dict(text="Vehicle types over session", font=dict(size=13)),
                        xaxis_title="Detection frame", yaxis_title="Count",
                        legend=dict(orientation="h", y=1.15),
                        margin=dict(t=35,b=35,l=45,r=10))
                    st.plotly_chart(fig_vt, use_container_width=True)

                # Session signal quality summary
                st.markdown("#### :material/straighten: Session Signal Quality")
                if dh_list and sh_list:
                    _dh = np.array(dh_list)
                    _sh = np.array(sh_list[:len(dh_list)])
                    _raw_std  = float(np.std(_dh))
                    _smth_std = float(np.std(_sh))
                    _raw_mean = float(np.mean(_dh))
                    _peak     = float(np.max(_dh))
                    _smooth_gain = (1 - _smth_std / (_raw_std + 1e-6)) * 100
                    _stability   = max(0.0, 100 - (_raw_std / (_raw_mean + 1e-6)) * 100)
                    # Final trend
                    _last10 = _sh[-10:] if len(_sh) >= 10 else _sh
                    _tdelta = float(_last10[-1] - _last10[0])
                    _tlabel = "↑ Rising" if _tdelta > 0.05 else "↓ Falling" if _tdelta < -0.05 else "→ Stable"
                    a1, a2, a3, a4 = st.columns(4)
                    a1.metric(":material/straighten: Detection Stability",
                              f"{_stability:.0f}%",
                              help="Higher = YOLO detections were consistent")
                    a2.metric(":material/show_chart: Kalman Smoothing",
                              f"{_smooth_gain:.0f}%",
                              help="How much noise the Kalman filter removed")
                    a3.metric(":material/arrow_upward: Peak Density",
                              f"{_peak:.2f}",
                              help="Highest density recorded during session")
                    a4.metric(":material/trending_up: Final Trend",
                              _tlabel,
                              help="Traffic direction at session end")
                    st.caption(
                        "Note: MAE/RMSE/MAPE require a trained model to be meaningful. "
                        "Run a prediction first (Upload Video tab) to get model accuracy metrics.")

                # ── Feature D: Alert log ──────────────────────────────────
                all_alerts = last_state.get("all_alerts", [])
                if all_alerts:
                    st.markdown("#### :material/notifications_active: Incident Log")
                    sev_bg  = {"HIGH": "#fee2e2", "MODERATE": "#fef9c3", "SPIKE": "#fff7ed"}
                    sev_brd = {"HIGH": "#fca5a5", "MODERATE": "#fde68a", "SPIKE": "#fed7aa"}
                    sev_txt = {"HIGH": "#991b1b", "MODERATE": "#92400e", "SPIKE": "#9a3412"}
                    for alert in reversed(all_alerts):
                        bg  = sev_bg.get(alert["severity"],  "#fee2e2")
                        brd = sev_brd.get(alert["severity"], "#fca5a5")
                        txt = sev_txt.get(alert["severity"], "#991b1b")
                        st.markdown(
                            f"<div style='background:{bg};border:1px solid {brd};"
                            f"border-radius:7px;padding:8px 14px;margin-bottom:5px;"
                            f"color:{txt};font-size:0.84rem;'>"
                            f"{alert['icon']} <b>[{alert['timestamp']}]</b> "
                            f"<b>{alert['severity']}</b> — {alert['message']}</div>",
                            unsafe_allow_html=True)
                    st.caption(f"{len(all_alerts)} incident(s) recorded during this session.")

                # ── Feature C: Online learner final stats ─────────────────
                ls = last_state.get("online_learner_stats")
                if ls and ls.get("update_count", 0) > 0:
                    st.markdown("#### :material/model_training: Online Learner Summary")
                    l1, l2, l3 = st.columns(3)
                    l1.metric(":material/refresh: Total updates",     ls["update_count"])
                    l2.metric(":material/storage: Buffer used",
                              f"{ls['buffer_size']}/{ls['buffer_capacity']}")
                    last_loss = ls.get("recent_loss")
                    l3.metric(":material/trending_down: Final loss",
                              f"{last_loss:.5f}" if last_loss else "—")