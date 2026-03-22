"""
main.py
-------
CLI entry point for the Traffic Flow Prediction System.

Usage:
    python main.py --video data/videos/traffic.mp4
    python main.py --video data/videos/traffic.mp4 --no-yolo
    python main.py --help

For the interactive UI run:
    streamlit run frontend/app_video.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from video_pipeline.video_extractor import video_to_traffic_series
from video_pipeline.video_dataset   import VideoTrafficDataset
from training.train_video           import train_video_model
from training.evaluate              import evaluate_model
from models.kalman                  import KalmanSmoother
from utils.metrics                  import MAE, RMSE, MAPE


def main():
    parser = argparse.ArgumentParser(description="Traffic Flow Prediction — CLI")
    parser.add_argument("--video",      type=str,   required=True)
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--clip-len",   type=int,   default=12)
    parser.add_argument("--fps-keep",   type=int,   default=5)
    parser.add_argument("--resize",     type=int,   default=64)
    parser.add_argument("--no-yolo",    action="store_true")
    parser.add_argument("--no-flow",    action="store_true")
    parser.add_argument("--confidence", type=float, default=0.3)
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Traffic Flow Prediction System")
    print(f"{'='*55}")
    print(f"  Video  : {args.video}")
    print(f"  Epochs : {args.epochs}")
    print(f"  YOLO   : {'off' if args.no_yolo else 'on'}")
    print(f"{'='*55}\n")

    # 1. Extract
    print("[1/4] Extracting frames...")
    frames, density, method, counts = video_to_traffic_series(
        args.video,
        resize          = (args.resize, args.resize),
        fps_keep        = args.fps_keep,
        use_flow        = not args.no_flow,
        use_yolo        = not args.no_yolo,
        yolo_confidence = args.confidence,
    )
    print(f"      Frames: {frames.shape[0]}  method: {method}")
    if counts:
        totals = [c["total"] for c in counts]
        print(f"      Vehicles: avg {sum(totals)/len(totals):.1f}/frame  peak {max(totals)}")

    # 2. Dataset
    print("\n[2/4] Building dataset...")
    dataset = VideoTrafficDataset(
        args.video,
        clip_len    = args.clip_len,
        resize      = (args.resize, args.resize),
        fps_keep    = args.fps_keep,
        use_flow    = not args.no_flow,
        use_yolo    = not args.no_yolo,
        precomputed = {"frames": frames, "density": density,
                       "method": method, "counts": counts},
    )
    print(f"      Clips: {len(dataset)}  shape: {dataset.input_shape}")

    # 3. Train
    print(f"\n[3/4] Training ({args.epochs} epochs)...")

    def progress(epoch, total, tl, vl):
        if epoch % max(1, total // 5) == 0 or epoch == total:
            print(f"      Epoch {epoch:3d}/{total} | train {tl:.5f} | val {vl:.5f}")

    model, history = train_video_model(
        dataset,
        epochs            = args.epochs,
        batch_size        = 8,
        progress_callback = progress,
    )

    # 4. Evaluate
    print("\n[4/4] Evaluating...")
    metrics = evaluate_model(model, dataset)

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            X, y = dataset[i]
            preds.append(model(X.unsqueeze(0)).item())
            trues.append(y.item())

    smooth  = KalmanSmoother().smooth(preds)
    current = float(smooth[-1])

    def level(v):
        return "LOW" if v < 0.33 else "MODERATE" if v < 0.66 else "HIGH"

    print(f"\n{'='*55}")
    print(f"  RESULTS")
    print(f"{'='*55}")
    print(f"  Traffic level  : {level(current)}")
    print(f"  Current density: {current:.4f}")
    print(f"  Peak density   : {max(trues):.4f}")
    print(f"  Avg density    : {sum(trues)/len(trues):.4f}")
    print(f"  MAE            : {metrics['MAE']:.4f}")
    print(f"  RMSE           : {metrics['RMSE']:.4f}")
    print(f"  MAPE           : {metrics['MAPE']:.2f}%")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()