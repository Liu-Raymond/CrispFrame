"""ffmpeg/ffprobe 定位与媒体探测。"""
import glob
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .layout import ROOT

_WINGET_GLOB = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg*",
)

_EXE_CACHE: dict[str, str] = {}


def _find_exe(name: str) -> str:
    p = _EXE_CACHE.get(name)
    if p:
        return p
    # 随包分发的 ffmpeg 优先：包根 ffmpeg/、包根、runtime 同级目录
    py_dir = Path(sys.executable).resolve().parent
    for base in (ROOT / "ffmpeg", ROOT, py_dir, py_dir / "ffmpeg"):
        cand = base / (name + ".exe")
        if cand.exists():
            p = str(cand)
            break
    if not p:
        p = shutil.which(name)
    if not p:
        hits = glob.glob(os.path.join(_WINGET_GLOB, "**", "bin", name + ".exe"),
                         recursive=True)
        if hits:
            p = hits[0]
    if not p:
        raise FileNotFoundError(
            f"{name} not found; install with: winget install --id Gyan.FFmpeg -e")
    _EXE_CACHE[name] = p
    return p


def ffmpeg() -> str:
    return _find_exe("ffmpeg")


def ffprobe() -> str:
    return _find_exe("ffprobe")


# GUI（pythonw 无控制台）spawn ffmpeg/ffprobe 时抑制黑框闪烁；
# POSIX 上为空 dict（CREATE_NO_WINDOW 是 Windows 专属参数）
SUBPROC_FLAGS = ({"creationflags": subprocess.CREATE_NO_WINDOW}
                 if sys.platform == "win32" else {})


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frames: int  # 0 = 未知（按时长估算）
    duration: float

    @property
    def fps_str(self) -> str:
        # NVENC 需要确定的帧率；用常见精确值避免浮点噪声
        return f"{self.fps:.6f}"


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    out = subprocess.run(
        [ffprobe(), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames",
         "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True, **SUBPROC_FLAGS,
    ).stdout
    data = json.loads(out)
    st = data["streams"][0]
    num, _, den = (st.get("avg_frame_rate") or st.get("r_frame_rate") or "25/1").partition("/")
    fps = float(num) / float(den or 1)
    duration = float(data.get("format", {}).get("duration") or 0)
    frames = int(st.get("nb_frames") or 0)
    if not frames and duration:
        frames = int(duration * fps)
    return VideoInfo(path, int(st["width"]), int(st["height"]), fps, frames, duration)
