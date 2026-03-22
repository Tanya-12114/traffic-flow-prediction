"""
yt_downloader.py
----------------
Downloads traffic videos from YouTube using yt-dlp.
- No ffmpeg required  (downloads single pre-merged format)
- No JS runtime required  (uses android player API)
"""

import os
import subprocess
import sys
from pathlib import Path


def install_ytdlp():
    try:
        import yt_dlp
    except ImportError:
        print("Installing yt-dlp...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])


def download_video(
    url: str,
    output_dir: str = "data/videos",
    filename: str = None,
    max_height: int = 480,
) -> str:
    """
    Download a YouTube video — no ffmpeg, no JS runtime needed.

    Uses a single pre-merged mp4 format (video+audio already combined).
    Falls back through progressively simpler formats until one works.

    Returns path to the downloaded file.
    """
    install_ytdlp()
    import yt_dlp

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if filename:
        out_template = str(Path(output_dir) / f"{filename}.%(ext)s")
    else:
        out_template = str(Path(output_dir) / "%(title).60s.%(ext)s")

    # Format string: only pre-merged single-file formats, NO merging needed
    # Tries best mp4 up to max_height, then any mp4, then any video
    fmt = (
        f"best[height<={max_height}][ext=mp4]"
        f"/best[ext=mp4]"
        f"/best[height<={max_height}]"
        f"/best"
    )

    ydl_opts = {
        "format"              : fmt,
        "outtmpl"             : out_template,
        "noplaylist"          : True,
        "quiet"               : False,
        "no_warnings"         : False,
        # Use android client — avoids JS runtime requirement
        "extractor_args"      : {"youtube": {"player_client": ["android"]}},
        # Never attempt merging
        "merge_output_format" : None,
        # Abort if merge would be needed (we want single-file only)
        "postprocessors"      : [],
    }

    downloaded_path = None

    def progress_hook(d):
        nonlocal downloaded_path
        if d["status"] == "finished":
            downloaded_path = d["filename"]

    ydl_opts["progress_hooks"] = [progress_hook]

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if downloaded_path and Path(downloaded_path).exists():
        print(f"\n✅ Downloaded: {downloaded_path}")
        return str(downloaded_path)

    # Fallback: find newest file in output dir
    files = sorted(Path(output_dir).glob("*.*"), key=lambda f: f.stat().st_mtime)
    if files:
        print(f"\n✅ Downloaded: {files[-1]}")
        return str(files[-1])

    raise RuntimeError("Download appeared to succeed but file not found.")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else input("YouTube URL: ").strip()
    path = download_video(url)
    print(f"Saved to: {path}")