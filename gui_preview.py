"""模块级预览对话框：当前帧/选段只进目标模块（不经链上其他环节），横向选模型/调参。

- 时间轴：QSlider 拖动 = ffmpeg 快速 seek 抽帧；片段渲染以当前位置为起点
- 片段渲染：真实管线跑一段（默认 10s，5-30s 可选），草稿编码（x264 veryfast），
  完成后内嵌播放器自动循环播放，可存入 A/B/C/D 槽位对比
- 帧渲染：单帧静态对比（快速看效果）
- 插帧模块：帧渲染取相邻帧真实 midframe；片段渲染走完整插帧管线
- 放大：Ctrl+滚轮 / 右上角按钮缩放 + 拖动平移（仅静态帧）
"""
from __future__ import annotations

import copy
import itertools
import os
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QGraphicsScene, QGraphicsView, QGridLayout,
    QHBoxLayout, QLabel, QProgressBar, QPushButton, QSlider, QStackedWidget,
    QToolButton, QVBoxLayout, QWidget,
)

from crispframe.ffutils import VideoInfo, SUBPROC_FLAGS, ffmpeg, probe
from crispframe.modules import BuildContext, ModuleInstance
from crispframe.pipeline import EncodeConfig, run as pipeline_run


def frame_to_pixmap(bgr: np.ndarray) -> QPixmap:
    bgr = np.ascontiguousarray(bgr)
    h, w = bgr.shape[:2]
    img = QImage(bgr.data, w, h, w * 3, QImage.Format_BGR888)
    return QPixmap.fromImage(img)


_tmp_seq = itertools.count()


def grab_frame(src: Path, t: float, quality: int = 2,
               max_width: int = 0) -> np.ndarray | None:
    """抽一帧。max_width>0 时降采样到该宽度以内（只缩不放大）——
    时间轴浏览用，4K 源全分辨率抽帧 ~300ms/帧，降采样后快数倍；
    喂模型/像素级对比的场景不要传（保持全分辨率）。
    临时文件名带 pid+调用序号：同秒并发抽帧（浏览/渲染/原片槽重抽多线程）
    各写各的，不互踩。"""
    tmp = (Path(tempfile.gettempdir()) /
           f"cf_prev_{t:.3f}_{max_width}_{os.getpid()}_{next(_tmp_seq)}.jpg")
    try:
        cmd = [ffmpeg(), "-nostdin", "-y", "-ss", f"{t:.3f}", "-i", str(src),
               "-frames:v", "1"]
        if max_width:
            cmd += ["-vf", f"scale={max_width}:{max_width}:force_original_aspect_ratio=decrease"]
        cmd += ["-q:v", str(quality), str(tmp)]
        subprocess.run(cmd, capture_output=True, timeout=25, **SUBPROC_FLAGS)
        img = cv2.imread(str(tmp))
        return img
    finally:
        tmp.unlink(missing_ok=True)


class SeekThumb(QThread):
    """后台 ffmpeg seek 抽帧（时间轴浏览用，降采样提速）。"""
    got = Signal(object)  # np.ndarray BGR

    def __init__(self, src: Path, t: float):
        super().__init__()
        self.src, self.t = src, t

    def run(self):
        img = grab_frame(self.src, self.t, quality=3, max_width=1920)
        if img is not None:
            self.got.emit(img)


def _gpu_release(built) -> None:
    """预览用完即卸：模型权重下卡、TRT 引擎析构、显存池归还 OS。

    背景：torch 缓存池从不主动归还（模型/激活释放后留在池内），换模型预览时
    nvidia-smi 显存单调增长；TRT runner 无主动销毁代码。预览是短任务，
    用完即卸的代价（下次预览重新加载权重）远小于显存累积。清理失败不影响结果。"""
    try:
        import gc

        import torch
        if built is not None:
            for stage in built.stages:
                # StrengthBlend 等包装层的真实推理对象在 .inner 里
                for obj in (stage, getattr(stage, "inner", None)):
                    if obj is None:
                        continue
                    if getattr(obj, "runner", None) is not None:
                        obj.runner = None     # TRT engine/context/stream 随析构归还
                    model = getattr(obj, "model", None)
                    nn = getattr(model, "model", None)   # spandrel descriptor → nn.Module
                    if nn is not None:
                        nn.cpu()              # 权重下卡
            if built.interpolator is not None:
                built.interpolator = None
            try:
                # RIFE 的全局光流网格缓存：每 (device, shape) 一张 CUDA 张量
                # （720p 约 7.5MB，4K UHD 约 67MB）常驻进程；clear 后 warp()
                # 按 miss 重建，并发 Worker 也只是重建一次的开销
                from crispframe._rife import _rife_warplayer
                _rife_warplayer.backwarp_tenGrid.clear()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()          # 缓存池归还 OS（nvidia-smi 立即回落）
    except Exception:  # noqa: BLE001  清理是尽力而为，绝不掩盖渲染结果
        pass


class RenderModule(QThread):
    """后台渲染单帧：当前帧只过目标模块。插帧模块取相邻两帧做真实 midframe。"""
    got = Signal(object, str)  # np.ndarray | None, message

    def __init__(self, src: Path, t: float, inst: ModuleInstance, label: str,
                 info: VideoInfo | None = None):
        super().__init__()
        self.src, self.t, self.inst, self.label = src, t, inst, label
        self.info = info

    def run(self):
        built = None
        try:
            info = self.info or probe(self.src)
            ctx = BuildContext(width=info.width, height=info.height, fps=info.fps)
            if self.inst.spec.key == "smooth":
                # 平滑是时域效果：单帧看不到混合；且其 build 会急切加载 RIFE
                # 权重上 GPU，为一条提示不值得——build 之前就早退
                self.got.emit(None, f"{self.label}（时域效果，请用『渲染片段』预览）")
                return
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
        finally:
            _gpu_release(built)   # 用完即卸：换模型预览不累积显存


class RenderSegment(QThread):
    """后台渲染选段：真实管线 + 草稿编码（x264 veryfast，兼容性最好）。
    完成后 got 发射临时 mp4 路径（None = 失败/零帧）。"""
    progress = Signal(int, int)
    got = Signal(object, str)

    def __init__(self, src: Path, start: float, duration: float,
                 inst: ModuleInstance, label: str, info: VideoInfo | None = None):
        super().__init__()
        # 注意不能叫 self.start——会覆盖 QThread.start() 方法
        self.src, self.t0, self.dur = src, start, duration
        self.inst, self.label = inst, label
        self.info = info
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        out = Path(tempfile.gettempdir()) / f"cf_seg_{int(time.time() * 1000)}.mp4"
        built = None
        try:
            info = self.info or probe(self.src)
            ctx = BuildContext(width=info.width, height=info.height, fps=info.fps)
            built = self.inst.spec.build(self.inst.params, ctx)
            encode = EncodeConfig(
                args=["-c:v", "libx264", "-preset", "veryfast",
                      "-crf", "18", "-pix_fmt", "yuv420p"],
                # 平滑等时域模块的混合/回采滤镜进预览，片段才反映真实效果
                vf=(f"{built.vf_pre},scale=out_color_matrix=bt709:out_range=tv"
                    if built.vf_pre else None),
                audio_args=["-an"],           # 预览段不带音轨：兼容性 + 省时间
            )
            stats = pipeline_run(
                info, out, stages=built.stages, interpolator=built.interpolator,
                encode=encode, batch_size=1, start=self.t0,
                max_seconds=self.dur,
                progress_callback=lambda d, t: self.progress.emit(d, t),
                cancel_check=lambda: self._cancel)
            if stats["frames_out"] > 0:
                self.got.emit(out, self.label)
            else:                       # 零帧产物一并清掉
                out.unlink(missing_ok=True)
                self.got.emit(None, self.label)
        except Exception as e:  # noqa: BLE001
            out.unlink(missing_ok=True)
            self.got.emit(None, f"{self.label}（失败: {e}）")
        finally:
            _gpu_release(built)   # 用完即卸：换模型预览不累积显存


class ZoomView(QGraphicsView):
    """Ctrl+滚轮缩放（光标锚点）+ 拖拽平移 + 双击切换 1:1 像素的图像查看器。

    超分差异必须 1:1 像素对齐才可见（fit 视图会把 4K 帧缩到 1/3，
    源帧与输出帧显示像素完全一样）。左上角常驻倍率标签（100% = 1:1），
    右上角常驻缩放按钮组（−/＋/1:1/适应）——缩放能力有可见入口，不靠盲猜手势。"""

    def __init__(self):
        super().__init__()
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._zoom = 1.0          # 相对 fit 的倍率（1.0 = fit 适配窗口）
        self._mode = "fit"        # fit = 适配窗口（resize 跟随）；zoom = 用户显式缩放（保持）
        self._pm_size: tuple[int, int] | None = None
        self._ratio_lab = QLabel(self)
        self._ratio_lab.setStyleSheet(
            "background:rgba(20,20,24,185);color:#ddd;padding:2px 8px;"
            "border-radius:4px;font-size:12px;")
        self._ratio_lab.hide()
        # 右上角悬浮缩放控制条
        bar = QWidget(self)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(3, 2, 3, 2)
        lay.setSpacing(2)
        for text, tip, fn in (
                ("−", "缩小（Ctrl+滚轮）", lambda: self._zoom_by(0.8)),
                ("＋", "放大（Ctrl+滚轮）", lambda: self._zoom_by(1.25)),
                ("1:1", "实际像素（双击画面切换）", self._toggle_1to1),
                ("适应", "适应窗口", self._fit_window)):
            b = QToolButton(bar)
            b.setText(text)
            b.setToolTip(tip)
            b.setAutoRaise(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(fn)
            lay.addWidget(b)
        bar.setStyleSheet("background:rgba(20,20,24,185);border-radius:4px;")
        self._ctl = bar
        self._place_ctl()

    def _place_ctl(self):
        self._ctl.adjustSize()
        self._ctl.move(max(10, self.width() - self._ctl.width() - 10), 10)
        self._ctl.raise_()

    def set_image(self, pixmap: QPixmap, keep_zoom: bool = False):
        if not keep_zoom:
            self._zoom = 1.0
            self._mode = "fit"
        else:
            # 从当前 transform 反推倍率：双击 1:1 时 _zoom 含滚动条出现前的
            # fit_scale 推算误差，直接用实际 m11 重derive 保证精确保持
            self._zoom = (self.transform().m11()
                          / max(self._fit_scale(), 1e-6))
            self._mode = "zoom" if self._zoom > 1.0 else "fit"
        sc = QGraphicsScene(self)
        sc.addPixmap(pixmap)
        self.setScene(sc)
        self._pm_size = pixmap.width(), pixmap.height()
        self._apply()
        self._update_ratio()

    def _fit_scale(self) -> float:
        """fit 状态下 1 图像像素对应的 view 逻辑像素数。"""
        if not self._pm_size:
            return 1.0
        vr = self.viewport().rect()
        pw, ph = self._pm_size
        return min(vr.width() / pw, vr.height() / ph) if pw and ph else 1.0

    def _apply(self):
        """按 _mode 应用视图：fit 适配窗口；zoom 按绝对倍率（fit 缩放 × _zoom）。"""
        if self._mode == "fit" or not self._pm_size:
            self.resetTransform()
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
            self._update_ratio()
            return
        center = self.mapToScene(self.viewport().rect().center())
        self.resetTransform()
        self.scale(self._fit_scale() * self._zoom,
                   self._fit_scale() * self._zoom)
        self.centerOn(center)
        self._update_ratio()

    def _update_ratio(self):
        if not self._pm_size:
            self._ratio_lab.hide()
            return
        # 1:1 = m11(1/dpr)：换算成相对物理像素的百分比，高分屏缩放下也显示正确
        pct = self.transform().m11() * max(self.devicePixelRatioF(), 1e-6) * 100
        self._ratio_lab.setText(f"{pct:.0f}%")
        self._ratio_lab.adjustSize()
        self._ratio_lab.move(10, 10)
        self._ratio_lab.raise_()
        self._ratio_lab.show()

    def _zoom_by(self, factor: float, anchor=None):
        """以 anchor（viewport 坐标，None=中心）为锚点缩放；下限 fit。"""
        if not self._pm_size:
            return
        anchor = anchor or self.viewport().rect().center()
        sp = self.mapToScene(anchor)
        self._zoom = max(1.0, self._zoom * factor)
        self._mode = "fit" if self._zoom <= 1.0 else "zoom"
        self._apply()
        if self._mode == "zoom":
            # 把锚点下的场景点放回锚点处（光标缩放）
            delta = self.mapFromScene(sp) - anchor
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() + delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() + delta.y())

    def _toggle_1to1(self):
        """fit/zoom ↔ 1:1（1 图像像素 = 1 屏幕物理像素；绝对 transform，
        不经 fit 推算避免舍入）。zoom 模式下 resize/滚动条变化不再重算，
        保证 1:1 稳定成立。双击画面与「1:1」按钮共用。"""
        dpr = max(self.devicePixelRatioF(), 1e-6)
        if abs(self.transform().m11() - 1 / dpr) < 0.02:
            self._fit_window()
            return
        self._mode = "zoom"
        center = self.mapToScene(self.viewport().rect().center())
        self._zoom = (1 / dpr) / max(self._fit_scale(), 1e-6)
        self.resetTransform()
        self.scale(1 / dpr, 1 / dpr)
        self.centerOn(center)
        self._update_ratio()

    def _fit_window(self):
        self._zoom = 1.0
        self._mode = "fit"
        self._apply()

    def wheelEvent(self, ev):
        if ev.modifiers() & Qt.ControlModifier:
            self._zoom_by(1.25 if ev.angleDelta().y() > 0 else 0.8,
                          ev.position().toPoint())
        else:
            super().wheelEvent(ev)   # 非 Ctrl：放大态滚滚动条平移，fit 态无操作

    def mouseDoubleClickEvent(self, ev):
        self._toggle_1to1()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._pm_size and self._mode == "fit":
            self.resetTransform()
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._update_ratio()
        self._place_ctl()


class ParamPanel(QWidget):
    """按 ParamSpec schema 自动生成的参数编辑区。changed 在任何参数变动时发射。"""
    changed = Signal()

    def __init__(self, inst: ModuleInstance, parent=None):
        super().__init__(parent)
        self.inst = inst
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
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
            cur = next((i for i, (_, v) in enumerate(p.choices)
                        if v == self.inst.params[p.key]), 0)
            box.setCurrentIndex(cur)
            box.currentIndexChanged.connect(
                lambda _: self._set(p, box.currentData()))
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
        lab = QLabel(f"{val:g}")
        slider.valueChanged.connect(
            lambda v: (lab.setText(f"{p.min + v * p.step:g}"),
                       self._set(p, p.min + v * p.step)))
        h.addWidget(slider, 1)
        h.addWidget(lab)
        return row

    def _set(self, p, value):
        self.inst.params[p.key] = p.coerce(value)
        self.changed.emit()


class PreviewDialog(QDialog):
    """模块预览：选段/单帧渲染 + 内嵌播放 + 参数面板 + 4 槽位对比。"""

    ORIG_KEY = "orig"                # 原片槽：固定第一位，不占字母
    RENDER_SLOTS = ["A", "B", "C"]   # 渲染位，顺延编号
    SEG_DURATIONS = [("5 秒", 5.0), ("10 秒", 10.0), ("15 秒", 15.0),
                     ("20 秒", 20.0), ("30 秒", 30.0)]

    def __init__(self, src: Path, inst: ModuleInstance, parent=None,
                 info: VideoInfo | None = None):
        super().__init__(parent)
        self.src = src
        self.inst = inst                      # 工作副本（面板直接改它）
        self.info = info or probe(src)        # 复用任务已探测的 info，省一次 ffprobe
        self.setWindowTitle(f"模块预览 — {inst.spec.name} · {src.name}")
        self.resize(1280, 760)
        self._render_thread: QThread | None = None
        self._thumb_thread: SeekThumb | None = None
        self._pending_t: float | None = None
        self._seg_slot_key = "A"
        self.slots: dict[str, dict] = {}      # key -> {kind: img|seg, payload, label}

        root = QVBoxLayout(self)

        # 查看区：静态帧（缩放） / 片段播放器 二选一显示
        self.view = ZoomView()
        self.video_widget = QVideoWidget()
        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_out)
        self.player.setVideoOutput(self.video_widget)
        self.player.mediaStatusChanged.connect(self._on_media_status)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.view)
        self.stack.addWidget(self.video_widget)
        root.addWidget(self.stack, 1)

        # 参数面板 + 渲染控制
        ctl = QHBoxLayout()
        if self.inst.spec.params:
            self.panel: QWidget = ParamPanel(self.inst)
        else:  # 无参模块（去压缩/降噪）：放模块说明，避免左侧空白像坏了
            self.panel = QLabel(self.inst.spec.description)
            self.panel.setStyleSheet("color:#8a8f98;")
            self.panel.setWordWrap(True)
        panel_holder = QWidget()
        panel_holder.setLayout(QHBoxLayout())
        panel_holder.layout().setContentsMargins(0, 0, 0, 0)
        panel_holder.layout().addWidget(self.panel)
        panel_holder.layout().addStretch()
        ctl.addWidget(panel_holder, 1)

        ctl.addWidget(QLabel("片段"))
        self.seg_dur = QComboBox()
        for disp, val in self.SEG_DURATIONS:
            self.seg_dur.addItem(disp, val)
        self.seg_dur.setCurrentIndex(1)
        ctl.addWidget(self.seg_dur)
        self.seg_btn = QPushButton("渲染片段")
        self.seg_btn.setToolTip("从下方时间轴当前位置起，渲染一段真实效果并循环播放")
        self.seg_btn.clicked.connect(self._render_segment)
        ctl.addWidget(self.seg_btn)

        self.slot = QComboBox()
        self.slot.addItems(self.RENDER_SLOTS)     # 渲染结果存 B/C/D；A 固定为原片
        ctl.addWidget(QLabel("存入槽位"))
        ctl.addWidget(self.slot)
        self.render_btn = QPushButton("渲染此帧")
        self.render_btn.setToolTip("只渲染当前这一帧（静态对比，最快）")
        self.render_btn.clicked.connect(self._render_current)
        ctl.addWidget(self.render_btn)
        self.compare_btn = QPushButton("🔍 对比")
        self.compare_btn.setToolTip("两路画面全屏对比：擦除叠加 / 左右并排 / 细节缩放")
        self.compare_btn.setEnabled(False)        # 至少有一个渲染结果后可用
        self.compare_btn.clicked.connect(self._open_compare)
        ctl.addWidget(self.compare_btn)
        root.addLayout(ctl)

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        root.addWidget(self.bar)

        # 槽位缩略图条（原片 + A/B/C 渲染位）
        slots_row = QHBoxLayout()
        slots_tip = QLabel("对比槽位：换着参数各渲染一段/一帧，点缩略图切换对比")
        slots_tip.setStyleSheet("color:#8a8f98;font-size:11px;")
        slots_row.addWidget(slots_tip)
        slots_row.addSpacing(12)
        self.slot_labels: dict[str, QLabel] = {}
        for s in [self.ORIG_KEY] + self.RENDER_SLOTS:
            init = "[原片]" if s == self.ORIG_KEY else f"[{s}] 空"
            lab = QLabel(init)
            lab.setFixedSize(220, 124)
            lab.setAlignment(Qt.AlignCenter)
            lab.setStyleSheet(
                "background:#111;border:1px solid #2b2d31;border-radius:4px;color:#666;")
            lab.mousePressEvent = (lambda ev, s=s: self._show_slot(s))
            self.slot_labels[s] = lab
            slots_row.addWidget(lab)
        slots_row.addStretch()
        root.addLayout(slots_row)

        # 时间轴（选择源片位置）
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
        self.stack.setCurrentWidget(self.view)   # 拖时间轴 = 回到源片帧浏览
        self.player.stop()
        self._show_original_at(t)

    def _show_original_at(self, t: float):
        # 拖动节流：抽帧进行中只记录最新位置，线程结束后补一次（最新优先）
        self._pending_t = t
        if self._thumb_thread and self._thumb_thread.isRunning():
            return
        self._drain_pending()

    def _drain_pending(self):
        t = self._pending_t
        self._pending_t = None
        if t is None:
            return
        self._thumb_thread = SeekThumb(self.src, t)
        self._thumb_thread.got.connect(self._set_original)
        self._thumb_thread.finished.connect(self._drain_pending)
        self._thumb_thread.start()

    def _set_original(self, img: np.ndarray):
        self.view.set_image(frame_to_pixmap(img))
        # 原片槽固定显示缩略图（对比器的"原片"位）
        pm = frame_to_pixmap(img)
        self.slots.setdefault(self.ORIG_KEY, {"kind": "img", "payload": pm,
                                              "label": "原片", "t0": 0.0, "dur": 0.0})
        self.slots[self.ORIG_KEY]["payload"] = pm
        self.slots[self.ORIG_KEY]["t0"] = self._current_t()
        lab = self.slot_labels[self.ORIG_KEY]
        lab.setPixmap(pm.scaled(220, 124, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lab.setText("")
        lab.setToolTip("原片（未增强）")

    def _current_t(self) -> float:
        return self.slider.value() / 10

    # —— 播放器 ——
    def _on_media_status(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.player.setPosition(0)
            self.player.play()

    def _play_segment(self, path: Path):
        self.stack.setCurrentWidget(self.video_widget)
        if self.player.source() != QUrl.fromLocalFile(str(path)):
            self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.player.play()

    # —— 渲染：单帧 ——
    def _render_current(self):
        if self._busy():
            return
        self.render_btn.setEnabled(False)
        self.render_btn.setText("渲染中…")
        inst = ModuleInstance(self.inst.key, copy.deepcopy(self.inst.params), True)
        inst.src_h = self.inst.src_h
        self._render_thread = RenderModule(
            self.src, self._current_t(), inst, inst.summary(), info=self.info)
        self._render_thread.got.connect(self._on_rendered)
        self._render_thread.start()

    def _on_rendered(self, img: np.ndarray | None, label: str):
        self.render_btn.setEnabled(True)
        self.render_btn.setText("渲染此帧")
        slot = self.slot.currentText()
        if img is None:
            self.setWindowTitle(f"模块预览 — {label}")
            return
        pm = frame_to_pixmap(img)
        self.stack.setCurrentWidget(self.view)
        self.view.set_image(pm)
        self._store_slot(slot, "img", pm, label, t0=self._current_t())
        # 放大差异要 1:1 像素才可见（fit 视图会被缩小抹掉）——渲染完成即引导
        self.setWindowTitle(f"模块预览 — {label} · Ctrl+滚轮 缩放，"
                            f"双击画面 1:1 看细节，或点 🔍 对比 与原片擦除对比")

    # —— 渲染：片段 ——
    def _render_segment(self):
        if self._busy():
            return
        t0 = self._current_t()
        dur = float(self.seg_dur.currentData())
        if t0 >= self.info.duration - 0.2:
            self.setWindowTitle("模块预览 — 当前位置已接近片尾，往回拖一点再渲染")
            return
        self._seg_t0, self._seg_dur = t0, dur
        inst = ModuleInstance(self.inst.key, copy.deepcopy(self.inst.params), True)
        inst.src_h = self.inst.src_h
        self._seg_slot_key = self.slot.currentText()
        self.seg_btn.setEnabled(False)
        self.render_btn.setEnabled(False)
        self.seg_btn.setText("渲染中…")
        self.bar.setVisible(True)
        self.bar.setRange(0, 0)
        self._render_thread = RenderSegment(
            self.src, t0, dur, inst, inst.summary(), info=self.info)
        self._render_thread.progress.connect(self._on_seg_progress)
        self._render_thread.got.connect(self._on_seg_done)
        self._render_thread.start()

    def _on_seg_progress(self, done, total):
        if total:
            self.bar.setRange(0, total)
            self.bar.setValue(done)

    def _on_seg_done(self, path: Path | None, label: str):
        self.seg_btn.setEnabled(True)
        self.render_btn.setEnabled(True)
        self.seg_btn.setText("渲染片段")
        self.bar.setVisible(False)
        if path is None:
            self.setWindowTitle(f"模块预览 — 片段渲染失败/取消（{label}）")
            return
        thumb = grab_frame(path, 0.0, quality=3, max_width=640)
        pm = frame_to_pixmap(thumb) if thumb is not None else QPixmap()
        self._store_slot(self._seg_slot_key, "seg", path, label,
                         t0=self._seg_t0, dur=self._seg_dur, thumb=pm)
        self._play_segment(path)

    def _store_slot(self, key: str, kind: str, payload, label: str,
                    t0: float = 0.0, dur: float = 0.0,
                    thumb: QPixmap | None = None):
        self.slots[key] = {"kind": kind, "payload": payload, "label": label,
                           "t0": t0, "dur": dur}
        lab = self.slot_labels[key]
        if thumb is None and kind == "img":
            thumb = payload
        if thumb is not None and not thumb.isNull():
            lab.setPixmap(thumb.scaled(220, 124, Qt.KeepAspectRatio,
                                       Qt.SmoothTransformation))
            lab.setText("")
        else:
            lab.setText(f"[{key}] ▶ {label[:14]}")
        tag = "▶" if kind == "seg" else "🖼"
        lab.setToolTip(f"[{key}] {tag} {label}")
        if key in self.RENDER_SLOTS:
            self.last_render_key = key          # 对比器默认选它
        self.compare_btn.setEnabled(           # A=原片之外至少有一个渲染结果
            any(k in self.slots for k in self.RENDER_SLOTS))

    def _show_slot(self, s: str):
        if s == self.ORIG_KEY:   # 原片位：全分辨率重抽（1:1 对比需要，与输出同倍率查看）
            self.player.stop()
            self.stack.setCurrentWidget(self.view)
            t = self._current_t()
            img = grab_frame(self.src, t)
            if img is not None:
                self.view.set_image(frame_to_pixmap(img), keep_zoom=True)
            else:
                self._show_original_at(t)   # 抽帧失败回退时间轴浏览图
            return
        entry = self.slots.get(s)
        if not entry:
            return
        if entry["kind"] == "seg":
            self._play_segment(entry["payload"])
        else:
            self.player.stop()
            self.stack.setCurrentWidget(self.view)
            self.view.set_image(entry["payload"], keep_zoom=True)   # 同倍率切换对比

    def _open_compare(self):
        if not any(k in self.slots for k in self.RENDER_SLOTS):
            self.setWindowTitle("模块预览 — 先渲染一个结果才能对比")
            return
        from compare_view import CompareDialog
        dlg = CompareDialog(self)
        dlg.exec()

    def _busy(self) -> bool:
        return bool(self._render_thread and self._render_thread.isRunning())

    def done(self, r):
        # 关闭时停掉渲染线程（取消片段管线）与播放
        if isinstance(self._render_thread, RenderSegment):
            self._render_thread.cancel()
        if self._render_thread and self._render_thread.isRunning():
            self._render_thread.wait(5000)
        self._pending_t = None
        if self._thumb_thread and self._thumb_thread.isRunning():
            self._thumb_thread.wait(3000)
        self.player.stop()
        self._cleanup_temp_segments()
        super().done(r)

    def _cleanup_temp_segments(self):
        """删除预览产生的临时片段（只动系统临时目录里 cf_seg_ 前缀的自己的文件）。"""
        tmp = Path(tempfile.gettempdir())
        for entry in self.slots.values():
            if entry.get("kind") != "seg":
                continue
            try:
                p = Path(entry["payload"])
                if p.is_relative_to(tmp) and p.name.startswith("cf_seg_"):
                    p.unlink(missing_ok=True)
            except (OSError, ValueError, TypeError):
                pass
