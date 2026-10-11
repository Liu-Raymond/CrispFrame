"""任务卡片组件：UniFab 式的视频任务卡片（缩略图/名称/IN-OUT/状态/操作）。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from crispframe.ffutils import VideoInfo, probe


def _thumb(src: Path, info: VideoInfo) -> QPixmap:
    """抽首帧作缩略图（失败返回空 pixmap）。"""
    import subprocess

    from crispframe.ffutils import SUBPROC_FLAGS, ffmpeg
    tmp = Path(src).with_suffix(".cfthumb.jpg")
    try:
        subprocess.run(
            [ffmpeg(), "-nostdin", "-y", "-ss", "1", "-i", str(src),
             "-frames:v", "1", "-q:v", "4", str(tmp)],
            capture_output=True, timeout=20, **SUBPROC_FLAGS)
        pm = QPixmap(str(tmp))
        tmp.unlink(missing_ok=True)
        return pm
    except Exception:  # noqa: BLE001
        return QPixmap()


class TaskCard(QFrame):
    """一个待处理视频任务的卡片。"""

    previewRequested = Signal(object)   # self
    removeRequested = Signal(object)

    MODELS = {
        "nomosuni": "NomosUni 2x（真人）",
        "anime": "AnimeJaNai 2x（动漫）",
        "nomos8k": "Nomos8k 4x（真人）",
        "ultrasharp": "UltraSharp 4x",
        "clearreality": "ClearReality 4x",
        "x4plus": "RealESRGAN 4x",
    }

    def __init__(self, src: Path, target_wh: tuple[int, int], parent=None):
        super().__init__(parent)
        self.src = src
        self.info: VideoInfo | None = None
        self.model = "nomosuni"
        self.status = "就绪"
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "TaskCard{background:#232428;border:1px solid #2b2d31;border-radius:8px;}")
        self._build(target_wh)
        self._probe(src, target_wh)

    def _build(self, target_wh):
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 10, 8)
        self.thumb = QLabel()
        self.thumb.setFixedSize(128, 72)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setStyleSheet("background:#111;border-radius:4px;")
        row.addWidget(self.thumb)

        col = QVBoxLayout()
        self.name_lab = QLabel(self.src.name)
        self.name_lab.setStyleSheet("font-weight:bold;")
        col.addWidget(self.name_lab)
        self.io_lab = QLabel("读取中…")
        self.io_lab.setStyleSheet("color:#9aa0a8;font-size:12px;")
        col.addWidget(self.io_lab)
        self.model_lab = QLabel()
        self.model_lab.setStyleSheet("color:#7fb0ff;font-size:12px;")
        col.addWidget(self.model_lab)
        row.addLayout(col, 1)

        ops = QVBoxLayout()
        self.preview_btn = QPushButton("AI 预览")
        self.preview_btn.clicked.connect(lambda: self.previewRequested.emit(self))
        self.remove_btn = QPushButton("移除")
        self.remove_btn.clicked.connect(lambda: self.removeRequested.emit(self))
        ops.addWidget(self.preview_btn)
        ops.addWidget(self.remove_btn)
        row.addLayout(ops)

        self.status_lab = QLabel(self.status)
        self.status_lab.setStyleSheet("color:#5fd08a;")
        row.addWidget(self.status_lab)

    def _probe(self, src, target_wh):
        try:
            self.info = probe(src)
            pm = _thumb(src, self.info)
            if not pm.isNull():
                self.thumb.setPixmap(pm.scaled(
                    128, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.refresh_output(target_wh)
        except Exception as e:  # noqa: BLE001
            self.io_lab.setText(f"无法读取: {e}")

    def refresh_output(self, target_wh: tuple[int, int]):
        if not self.info:
            return
        i = self.info
        in_txt = (f"INPUT  {i.width}x{i.height} · {i.fps:.0f}fps · "
                  f"{src_size_mb(self.src):.1f} MB")
        out_h = target_wh[1]
        out_txt = f"OUTPUT → {target_wh[0]}x{out_h}"
        self.io_lab.setText(f"{in_txt}    {out_txt}")

    def set_model(self, name: str):
        self.model = name
        self.model_lab.setText(f"AI 模型: {self.MODELS.get(name, name)}")

    def set_status(self, text: str, color: str = "#5fd08a"):
        self.status_lab.setText(text)
        self.status_lab.setStyleSheet(f"color:{color};")


def src_size_mb(p: Path) -> float:
    try:
        return p.stat().st_size / 1e6
    except OSError:
        return 0.0
