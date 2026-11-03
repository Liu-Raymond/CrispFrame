"""ffmpeg/ffprobe 定位与媒体探测。"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
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
    if not path.exists():
        raise ValueError(f"文件不存在或已被移动/删除：{path}")
    # 瞬时失败重试：来源程序可能正占用文件（如 Wallpaper Engine 播放壁纸时
    # 的间歇性句柄），ffprobe 此时 stdout 输出 {}——用户实测"重试就成功"，
    # 对这类竞态自动重试而不是把误导向"无视频轨"的错误抛给用户
    last: Exception | None = None
    for _ in range(3):
        try:
            out = subprocess.run(
                [ffprobe(), "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames",
                 "-show_entries", "format=duration",
                 "-of", "json", str(path)],
                capture_output=True, text=True, check=True, **SUBPROC_FLAGS,
            ).stdout
            data = json.loads(out)
            streams = data.get("streams") or []
            if not streams:
                raise KeyError("streams")     # 统一走下面的重试/报错路径
            st = streams[0]
            num, _, den = (st.get("avg_frame_rate") or st.get("r_frame_rate") or "25/1").partition("/")
            fps = float(num) / float(den or 1)
            duration = float(data.get("format", {}).get("duration") or 0)
            frames = int(st.get("nb_frames") or 0)
            if not frames and duration:
                frames = int(duration * fps)
            return VideoInfo(path, int(st["width"]), int(st["height"]),
                             fps, frames, duration)
        except (subprocess.CalledProcessError, json.JSONDecodeError,
                KeyError, IndexError) as e:
            last = e
            time.sleep(0.4)
    if isinstance(last, KeyError):        # streams 键缺失：三次重试后仍读不到流
        raise ValueError(
            f"暂时无法读取视频流（已重试）：文件可能正被其他程序占用"
            f"（如 Wallpaper Engine 正在播放），稍后重试即可；"
            f"若持续出现，则文件可能没有视频轨或已损坏（{path.name}）")
    raise ValueError(
        f"无法读取视频流：文件可能没有视频轨（纯音频/数据流）、已损坏，"
        f"或不是本工具支持的封装（{path.name}）") from last
