"""模块级预览对话框：当前帧只进目标模块（不经链上其他环节），用于横向选模型/调参。

- 时间轴：QSlider 拖动 = ffmpeg 快速 seek 抽帧显示
- 渲染：按模块注册表真实构建该模块的 stage（所见即所得）
- 插帧模块：取 t 与下一帧做真实 midframe
- 放大：QGraphicsView 滚轮缩放 + 拖动平移
- 对比：4 个配置槽位（A/B/C/D），槽位记录当时的参数快照
"""
from __future__ import annotations

import copy
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QGraphicsScene, QGraphicsView, QGridLayout,
    QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget,
)

from crispframe import modules
from crispframe.ffutils import VideoInfo, ffmpeg, probe
from crispframe.modules import BuildContext, ModuleInstance


def frame_to_pixmap(bgr: np.ndarray) -> QPixmap:
    bgr = np.ascontiguousarray(bgr)
    h, w = bgr.shape[:2]
    img = QImage(bgr.data, w, h, w * 3, QImage.Format_BGR888)
    return QPixmap.fromImage(img)


def grab_frame(src: Path, t: float, quality: int = 2) -> np.ndarray | None:
    tmp = Path(tempfile.gettempdir()) / f"cf_prev_{t:.3f}.jpg"
    try:
        subprocess.run(
            [ffmpeg(), "-nostdin", "-y", "-ss", f"{t:.3f}", "-i", str(src),
             "-frames:v", "1", "-q:v", str(quality), str(tmp)],
            capture_output=True, timeout=25)
        img = cv2.imread(str(tmp))
        return img
    finally:
        tmp.unlink(missing_ok=True)


class SeekThumb(QThread):
    """后台 ffmpeg seek 抽帧。"""
    got = Signal(object)  # np.ndarray BGR

    def __init__(self, src: Path, t: float):
        super().__init__()
        self.src, self.t = src, t

    def run(self):
        img = grab_frame(self.src, self.t, quality=3)
        if img is not None:
            self.got.emit(img)


class RenderModule(QThread):
    """后台渲染：当前帧只过目标模块。插帧模块取相邻两帧做真实 midframe。"""
    got = Signal(object, str)  # np.ndarray | None, message

    def __init__(self, src: Path, t: float, inst: ModuleInstance, label: str):
        super().__init__()
        self.src, self.t, self.inst, self.label = src, t, inst, label

    def run(self):
        try:
            info: VideoInfo = probe(self.src)
            ctx = BuildContext(width=info.width, height=info.height, fps=info.fps)
            built = self.inst.spec.build(self.inst.params, ctx)
            if built.interpolator is not None:
                f1 = grab_frame(self.src, self.t)
                f2 = grab_frame(self.src, self.t + 1.0 / max(info.fps, 1e-6))
                if f1 is None or f2 is None:
                    raise RuntimeError("抽帧失败")
                out = built.interpolator.midframe(f1, f2)
            else:
                f1 = grab_frame(self.src, self.t)
                if f1 is None:
                    raise RuntimeError("抽帧失败")
                seq = [f1]
                for stage in built.stages:
                    seq = stage(seq)
                out = seq[0]
            self.got.emit(out, self.label)
        except Exception as e:  # noqa: BLE001
            self.got.emit(None, f"{self.label}（失败: {e}）")


class ZoomView(QGraphicsView):
    """滚轮缩放 + 拖拽平移的图像查看器。"""

    def __init__(self):
        super().__init__()
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._zoom = 1.0

    def set_image(self, pixmap: QPixmap):
        self._zoom = 1.0
        self.resetTransform()
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


class ParamPanel(QWidget):
    """按 ParamSpec schema 自动生成的参数编辑区。changed 在任何参数变动时发射。"""
    changed = Signal()

    def __init__(self, inst: ModuleInstance, parent=None):
        super().__init__(parent)
        self.inst = inst
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        self._editors: dict[str, QWidget] = {}
        for r, p in enumerate(inst.spec.params):
            grid.addWidget(QLabel(p.label), r, 0)
            editor = self._make_editor(p)
            grid.addWidget(editor, r, 1)
            if p.tip:
                tip = QLabel(p.tip)
                tip.setStyleSheet("color:#8a8f98;font-size:11px;")
                tip.setWordWrap(True)
                grid.addWidget(tip, r, 2)
        grid.setColumnStretch(1, 1)

    def _make_editor(self, p) -> QWidget:
        if p.kind == "choice":
            box = QComboBox()
            for disp, val in p.choices:
                box.addItem(disp, val)
            cur = next((i for i, (_, v) in enumerate(p.choices) if v == self.inst.params[p.key]), 0)
            box.setCurrentIndex(cur)
            box.currentIndexChanged.connect(lambda _: self._set(p, box.currentData()))
            return box
        if p.kind == "bool":
            chk = QCheckBox()
            chk.setChecked(bool(self.inst.params[p.key]))
            chk.toggled.connect(lambda v: self._set(p, v))
            return chk
        # float/int：滑杆 + 数值
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Horizontal)
        steps = int(round((p.max - p.min) / p.step))
        slider.setRange(0, steps)
        val = float(self.inst.params[p.key])
        slider.setValue(int(round((val - p.min) / p.step)))
        lab = QLabel(self._fmt(p, val))
        slider.valueChanged.connect(
            lambda v: (lab.setText(self._fmt(p, p.min + v * p.step)),
                       self._set(p, p.min + v * p.step)))
        h.addWidget(slider, 1)
        h.addWidget(lab)
        return row

    @staticmethod
    def _fmt(p, v: float) -> str:
        return f"{v:g}"

    def _set(self, p, value):
        self.inst.params[p.key] = p.coerce(value)
        self.changed.emit()


class PreviewDialog(QDialog):
    """模块预览：参数面板 + 时间轴 + 放大镜 + 4 槽位对比。"""

    SLOT_KEYS = ["A", "B", "C", "D"]

    def __init__(self, src: Path, inst: ModuleInstance, parent=None):
        super().__init__(parent)
        self.src = src
        self.inst = inst                      # 工作副本（面板直接改它）
        self.info: VideoInfo = probe(src)
        self.setWindowTitle(f"模块预览 — {inst.spec.name} · {src.name}")
        self.resize(1280, 760)
        self._render_thread: RenderModule | None = None
        self._thumb_thread: SeekThumb | None = None
        self.slots: dict[str, tuple[QPixmap, str]] = {}   # key -> (pixmap, summary)

        root = QVBoxLayout(self)
        self.view = ZoomView()
        root.addWidget(self.view, 1)

        # 参数面板 + 渲染按钮
        ctl = QHBoxLayout()
        self.panel = ParamPanel(self.inst)
        panel_holder = QWidget()
        panel_holder.setLayout(QHBoxLayout())
        panel_holder.layout().setContentsMargins(0, 0, 0, 0)
        panel_holder.layout().addWidget(self.panel)
        panel_holder.layout().addStretch()
        ctl.addWidget(panel_holder, 1)
        self.slot = QComboBox()
        self.slot.addItems(self.SLOT_KEYS)
        ctl.addWidget(QLabel("存入槽位"))
        ctl.addWidget(self.slot)
        self.render_btn = QPushButton("渲染此帧")
        self.render_btn.clicked.connect(self._render_current)
        ctl.addWidget(self.render_btn)
        root.addLayout(ctl)

        # 槽位缩略图条
        slots_row = QHBoxLayout()
        self.slot_labels: dict[str, QLabel] = {}
        for s in self.SLOT_KEYS:
            lab = QLabel(f"[{s}] 空")
            lab.setFixedSize(220, 124)
            lab.setAlignment(Qt.AlignCenter)
            lab.setStyleSheet(
                "background:#111;border:1px solid #2b2d31;border-radius:4px;color:#666;")
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

        self._show_original_at(0)

    # —— 时间轴 ——
    def _on_seek(self, v):
        t = v / 10
        self.time_lab.setText(f"{t // 60:02.0f}:{t % 60:04.1f}")
        self._show_original_at(t)

    def _show_original_at(self, t: float):
        if self._thumb_thread and self._thumb_thread.isRunning():
            return
        self._thumb_thread = SeekThumb(self.src, t)
        self._thumb_thread.got.connect(self._set_original)
        self._thumb_thread.start()

    def _set_original(self, img: np.ndarray):
        self.view.set_image(frame_to_pixmap(img))

    def _current_t(self) -> float:
        return self.slider.value() / 10

    # —— 渲染 ——
    def _render_current(self):
        if self._render_thread and self._render_thread.isRunning():
            return
        self.render_btn.setEnabled(False)
        self.render_btn.setText("渲染中…")
        # 参数快照：渲染用副本，槽位记录的就是这份配置
        inst = ModuleInstance(self.inst.key, copy.deepcopy(self.inst.params), True)
        self._render_thread = RenderModule(
            self.src, self._current_t(), inst, inst.summary())
        self._render_thread.got.connect(self._on_rendered)
        self._render_thread.start()

    def _on_rendered(self, img: np.ndarray | None, label: str):
        self.render_btn.setEnabled(True)
        self.render_btn.setText("渲染此帧")
        if img is None:
            self.view.set_image(frame_to_pixmap(grab_frame(self.src, self._current_t())
                                                or np.zeros((360, 640, 3), np.uint8)))
            self.setWindowTitle(f"模块预览 — {label}")
            return
        pm = frame_to_pixmap(img)
        self.view.set_image(pm)
        slot = self.slot.currentText()
        self.slots[slot] = (pm, label)
        lab = self.slot_labels[slot]
        lab.setPixmap(pm.scaled(220, 124, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lab.setText("")
        lab.setToolTip(label)

    def _show_slot(self, s: str):
        if s in self.slots:
            self.view.set_image(self.slots[s][0])
