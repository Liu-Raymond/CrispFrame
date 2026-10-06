"""AI 预览对话框：时间轴选段 + 指定模型渲染 + 放大镜细节查看 + 多配置对比。

- 时间轴：QSlider 拖动 = ffmpeg 快速 seek 抽帧显示
- 渲染：对选中帧跑当前模型管线（真实输出，所见即所得）
- 放大：QGraphicsView 滚轮缩放 + 拖动平移
- 对比：4 个配置槽位，渲染结果可存入槽位，点击槽位切换显示，支持并排
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QGraphicsView, QHBoxLayout, QLabel, QPushButton,
    QSlider, QVBoxLayout, QWidget, QGridLayout, QProgressBar,
)

from crispframe import registry
from crispframe.ffutils import VideoInfo, ffmpeg
from crispframe.gpu_post import GpuPost
from crispframe.sr import SuperRes


def frame_to_pixmap(bgr: np.ndarray) -> QPixmap:
    bgr = np.ascontiguousarray(bgr)
    h, w = bgr.shape[:2]
    img = QImage(bgr.data, w, h, w * 3, QImage.Format_BGR888)
    return QPixmap.fromImage(img)


class SeekThumb(QThread):
    """后台 ffmpeg seek 抽帧。"""
    got = Signal(object)  # np.ndarray BGR

    def __init__(self, src: Path, t: float):
        super().__init__()
        self.src, self.t = src, t

    def run(self):
        tmp = Path(tempfile.gettempdir()) / f"cf_prev_{self.t:.2f}.jpg"
        try:
            subprocess.run(
                [ffmpeg(), "-nostdin", "-y", "-ss", f"{self.t:.3f}", "-i", str(self.src),
                 "-frames:v", "1", "-q:v", "3", str(tmp)],
                capture_output=True, timeout=25)
            img = cv2.imread(str(tmp))
            tmp.unlink(missing_ok=True)
            if img is not None:
                self.got.emit(img)
        except Exception:  # noqa: BLE001
            pass


class RenderFrame(QThread):
    """后台渲染当前帧：模型超分 + GPU 后处理（真实管线）。"""
    got = Signal(object, str)  # np.ndarray, config_name

    def __init__(self, frame_bgr: np.ndarray, model: str, sharpen: float,
                 target_wh: tuple[int, int], config_name: str):
        super().__init__()
        self.frame, self.model = frame_bgr, model
        self.sharpen, self.target_wh, self.config_name = sharpen, target_wh, config_name

    def run(self):
        try:
            spec = registry.resolve(self.model)
            sr = SuperRes(spec, use_trt=True)
            outs = sr([self.frame])
            post = GpuPost(self.target_wh[0], self.target_wh[1], sharpen=self.sharpen)
            outs = post(outs)
            self.got.emit(outs[0], self.config_name)
        except Exception as e:  # noqa: BLE001
            self.got.emit(self.frame, f"{self.config_name}（失败:{e}）")


class ZoomView(QGraphicsView):
    """滚轮缩放 + 拖拽平移的图像查看器。"""

    def __init__(self):
        super().__init__()
        self.setRenderHint(self.renderHints().__class__.Antialiasing, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._zoom = 1.0

    def set_image(self, pixmap: QPixmap):
        self._zoom = 1.0
        self.resetTransform()
        self.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        from PySide6.QtWidgets import QGraphicsScene
        sc = QGraphicsScene(self)
        sc.addPixmap(pixmap)
        self.setScene(sc)
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, ev):
        factor = 1.25 if ev.angleDelta().y() > 0 else 0.8
        self._zoom *= factor
        if self._zoom < 1.0:
            self._zoom = 1.0
            self.resetTransform()
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
            return
        self.scale(factor, factor)


class PreviewDialog(QDialog):
    def __init__(self, src: Path, parent=None):
        super().__init__(parent)
        self.src = src
        from crispframe.ffutils import probe
        self.info: VideoInfo = probe(src)
        self.setWindowTitle(f"AI 预览 — {src.name}")
        self.resize(1280, 760)
        self._render_thread = None
        self._thumb_thread = None
        self.slots: dict[str, QPixmap] = {}

        root = QVBoxLayout(self)
        self.view = ZoomView()
        root.addWidget(self.view, 1)

        # 控制条
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("模型"))
        self.model = QComboBox()
        for n in registry.available():
            spec = registry.BUILTIN[n]
            self.model.addItem(f"{n}（{spec.scale}x）", n)
        ctl.addWidget(self.model)
        ctl.addWidget(QLabel("锐化"))
        self.sharpen = QComboBox()
        self.sharpen.addItems(["0", "0.3", "0.6", "0.9", "1.2"])
        self.sharpen.setCurrentIndex(2)
        ctl.addWidget(self.sharpen)
        ctl.addWidget(QLabel("对比槽位"))
        self.slot = QComboBox()
        self.slot.addItems(["A", "B", "C", "D"])
        ctl.addWidget(self.slot)
        self.render_btn = QPushButton("渲染此帧")
        self.render_btn.clicked.connect(self._render_current)
        ctl.addWidget(self.render_btn)
        ctl.addStretch()
        root.addLayout(ctl)

        # 槽位缩略图条
        slots_row = QHBoxLayout()
        self.slot_labels: dict[str, QLabel] = {}
        for s in ["A", "B", "C", "D"]:
            lab = QLabel(f"[{s}] 空")
            lab.setFixedSize(220, 124)
            lab.setAlignment(Qt.AlignCenter)
            lab.setStyleSheet("background:#111;border:1px solid #2b2d31;border-radius:4px;color:#666;")
            lab.mousePressEvent = (lambda ev, s=s: self._show_slot(s))
            self.slot_labels[s] = lab
            slots_row.addWidget(lab)
        slots_row.addStretch()
        root.addLayout(slots_row)

        # 时间轴
        tl = QHBoxLayout()
        self.time_lab = QLabel("00:00.0")
        tl.addWidget(self.time_lab)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, max(0, int(self.info.duration * 10)))
        self.slider.setPageStep(5)
        self.slider.valueChanged.connect(self._on_seek)
        tl.addWidget(self.slider, 1)
        root.addLayout(tl)

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        root.addWidget(self.bar)

        self._show_original_at(0)

    # —— 抽帧 ——
    def _on_seek(self, v):
        t = v / 10
        self.time_lab.setText(f"{t//60:02.0f}:{t%60:04.1f}")
        self._show_original_at(t)

    def _show_original_at(self, t: float):
        if self._thumb_thread and self._thumb_thread.isRunning():
            return
        self._pending_t = t
        self._thumb_thread = SeekThumb(self.src, t)
        self._thumb_thread.got.connect(self._set_original)
        self._thumb_thread.start()

    def _set_original(self, img: np.ndarray):
        self.view.set_image(frame_to_pixmap(img))

    def _render_current(self):
        if self._render_thread and self._render_thread.isRunning():
            return
        # 取当前时间轴帧
        img = self._grab_current_frame()
        if img is None:
            return
        name = f"{self.model.currentText()}+s{self.sharpen.currentText()}"
        self.render_btn.setEnabled(False)
        self.bar.setVisible(True)
        self.bar.setRange(0, 0)
        self._render_thread = RenderFrame(
            img, self.model.currentData(), float(self.sharpen.currentText()),
            (3840, 2160), name)
        self._render_thread.got.connect(self._on_rendered)
        self._render_thread.start()

    def _grab_current_frame(self) -> np.ndarray | None:
        t = self.slider.value() / 10
        tmp = Path(tempfile.gettempdir()) / "cf_prev_render.jpg"
        subprocess.run(
            [ffmpeg(), "-nostdin", "-y", "-ss", f"{t:.3f}", "-i", str(self.src),
             "-frames:v", "1", "-q:v", "2", str(tmp)],
            capture_output=True, timeout=25)
        img = cv2.imread(str(tmp))
        tmp.unlink(missing_ok=True)
        return img

    def _on_rendered(self, img: np.ndarray, name: str):
        self.render_btn.setEnabled(True)
        self.bar.setVisible(False)
        pm = frame_to_pixmap(img)
        self.view.set_image(pm)
        slot = self.slot.currentText()
        self.slots[slot] = pm
        lab = self.slot_labels[slot]
        lab.setPixmap(pm.scaled(220, 124, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lab.setText("")
        self.slot_labels[slot].setToolTip(name)

    def _show_slot(self, s: str):
        if s in self.slots:
            self.view.set_image(self.slots[s])
