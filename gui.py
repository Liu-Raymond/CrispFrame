"""CrispFrame GUI v2 —— Topaz/UniFab 式任务流。

首页（拖入视频）→ 任务页（任务头 + 可拖拽模块链 + 输出设置抽屉）→ 批量处理。
模块链与编码设置的全部业务逻辑在 crispframe.modules / crispframe.codec，
本文件只做 UI 组装与线程调度。
"""
from __future__ import annotations

import copy
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMenu, QProgressBar, QPushButton, QSlider,
    QSpinBox, QSplitter, QStackedWidget, QTabWidget, QTextEdit, QToolButton,
    QVBoxLayout, QWidget,
)

from crispframe import codec, modules, profiles
from crispframe.codec import CodecSettings
from crispframe.ffutils import VideoInfo, probe
from crispframe.modules import BuildContext, ModuleInstance
from crispframe.pipeline import EncodeConfig, run

APP_TITLE = "CrispFrame — 让每一帧都锐利"

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".ts", ".m2ts", ".flv"}

DARK_QSS = """
QWidget { background:#1e1f22; color:#d7d8db; font-size:13px; }
QListWidget, QTextEdit { background:#17181a; border:1px solid #2b2d31; border-radius:6px; }
QPushButton { background:#2f3136; border:1px solid #3f4147; border-radius:6px;
              padding:6px 12px; }
QPushButton:hover { background:#3a3c42; }
QPushButton:disabled { color:#6a6c72; }
QPushButton#startBtn { background:#3b82f6; border:none; font-weight:bold; padding:10px; }
QPushButton#startBtn:hover { background:#2f6fe0; }
QPushButton#startBtn:disabled { background:#2a3a55; color:#7d8ba1; }
QPushButton#stopBtn  { background:#8b3a3a; border:none; padding:10px; }
QPushButton#dropZone { background:#232428; border:2px dashed #3f4147; border-radius:10px; }
QPushButton#dropZone:hover { border-color:#3b82f6; background:#26282d; }
QFrame#taskHeader { background:#232428; border:1px solid #2b2d31; border-radius:8px; }
QFrame#moduleRow  { background:#232428; border:1px solid #2b2d31; border-radius:8px; }
QFrame#hintBar    { background:#4a3a15; border:1px solid #8a6d1a; border-radius:6px; }
QGroupBox { border:1px solid #2b2d31; border-radius:8px; margin-top:12px;
            padding-top:8px; font-weight:bold; }
QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 4px; color:#9aa0a8; }
QProgressBar { background:#17181a; border:1px solid #2b2d31; border-radius:6px;
               text-align:center; height:20px; }
QProgressBar::chunk { background:#3b82f6; border-radius:5px; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background:#17181a; border:1px solid #2b2d31;
                                      border-radius:4px; padding:3px 6px; }
QSlider::groove:horizontal { height:4px; background:#2b2d31; border-radius:2px; }
QSlider::handle:horizontal { width:14px; margin:-6px 0; background:#3b82f6; border-radius:7px; }
QTabWidget::pane { border:1px solid #2b2d31; border-radius:6px; }
QTabBar::tab { background:#232428; padding:6px 16px; border-top-left-radius:6px;
               border-top-right-radius:6px; }
QTabBar::tab:selected { background:#2f3136; color:#fff; }
"""


# ---------------------------------------------------------------- 任务模型

@dataclass
class Task:
    src: Path
    info: VideoInfo
    modules: list[ModuleInstance] = field(default_factory=list)
    codec_settings: CodecSettings = field(default_factory=CodecSettings)
    out_dir: Path | None = None
    status: str = "就绪"

    def __post_init__(self):
        if not self.modules:
            content = "anime" if _looks_anime(self.info) else "real"
            self.modules = modules.default_chain(content)
        if self.out_dir is None:
            self.out_dir = self.src.parent

    def enabled_modules(self) -> list[ModuleInstance]:
        return [m for m in self.modules if m.enabled]

    def out_path(self) -> Path:
        tags = "+".join(m.key for m in self.enabled_modules()) or "pass"
        return self.out_dir / f"{self.src.stem}.cf.{tags}{self.codec_settings.ext()}"

    def out_size(self) -> tuple[int, int]:
        """输出分辨率：按启用的超分模块推算（无则源分辨率/自定义）。"""
        cs = self.codec_settings
        if cs.res_mode == "custom":
            w, _, h = (cs.res_custom or "").lower().partition("x")
            if w.isdigit() and h.isdigit():
                return int(w), int(h)
        for m in self.enabled_modules():
            if m.key == "sr":
                target = str(m.params["target_res"] or "")
                if target:
                    w, _, h = target.partition("x")
                    return int(w), int(h)
                spec = modules.registry.resolve(m.params["model"])
                return self.info.width * spec.scale, self.info.height * spec.scale
        return self.info.width, self.info.height

    def out_fps_value(self) -> float:
        has_interp = any(m.key == "interp" for m in self.enabled_modules())
        return codec.out_fps(self.codec_settings, self.info.fps, has_interp)


def _looks_anime(info: VideoInfo) -> bool:
    """极简内容启发：仅影响默认模型选择，用户可在模块链改。"""
    return False  # 真人占 6 成且默认 nomosuni；动漫用户手动切换


# ---------------------------------------------------------------- Worker

class Worker(QThread):
    taskProgress = Signal(str, int, int)   # 任务名, done, total
    log = Signal(str)
    taskDone = Signal(object, bool, str)   # Task, ok, message
    allDone = Signal(bool, str)

    def __init__(self, tasks: list[Task]):
        super().__init__()
        self.tasks = tasks
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        ok_all = True
        for task in self.tasks:
            if self._cancel:
                break
            try:
                self._run_one(task)
            except Exception as e:  # noqa: BLE001
                self.log.emit(f"[错误] {task.src.name}: {e}")
                self.taskDone.emit(task, False, str(e))
                ok_all = False
        self.allDone.emit(ok_all and not self._cancel,
                          "已取消" if self._cancel else ("全部完成" if ok_all else "有任务失败"))

    def _run_one(self, task: Task):
        self.log.emit(f"▶ {task.src.name}")
        ctx = BuildContext(width=task.info.width, height=task.info.height,
                           fps=task.info.fps)
        stages, interp = modules.build_chain(task.modules, ctx)
        cs = task.codec_settings
        encode = EncodeConfig(**codec.build_encode_config_kwargs(
            cs, task.info.fps, interp is not None))
        out = task.out_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        run(task.info, out, stages=stages, interpolator=interp, encode=encode,
            batch_size=1, max_seconds=None,
            progress_callback=lambda d, t: self.taskProgress.emit(task.src.name, d, t),
            cancel_check=lambda: self._cancel)
        dt = time.perf_counter() - t0
        size_mb = out.stat().st_size / 1e6
        chain = "+".join(m.key for m in task.enabled_modules()) or "pass"
        self.log.emit(f"✓ {out.name}  [{chain}] {dt:.0f}s / {size_mb:.1f}MB")
        self.taskDone.emit(task, True, out.name)


# ---------------------------------------------------------------- 模块链 UI

class ModuleParamDialog(QDialog):
    """模块参数编辑：按 ParamSpec 自动生成，确定后写回实例。"""

    def __init__(self, inst: ModuleInstance, parent=None):
        super().__init__(parent)
        self.inst = inst   # 调用方传入的是工作副本；accept 后由调用方写回
        self.setWindowTitle(f"{inst.spec.name} — 参数")
        self.setMinimumWidth(480)
        v = QVBoxLayout(self)
        desc = QLabel(inst.spec.description)
        desc.setStyleSheet("color:#9aa0a8;")
        desc.setWordWrap(True)
        v.addWidget(desc)
        if inst.spec.params:
            from gui_preview import ParamPanel
            v.addWidget(ParamPanel(inst))
        else:
            v.addWidget(QLabel("此模块没有可调参数。"))
        row = QHBoxLayout()
        row.addStretch()
        ok = QPushButton("确定")
        ok.setObjectName("startBtn")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        row.addWidget(ok)
        row.addWidget(cancel)
        v.addLayout(row)


class ModuleRow(QFrame):
    previewRequested = Signal(object)   # ModuleInstance
    configApplied = Signal(object)      # ModuleInstance（参数写回后）
    removed = Signal(object)
    toggled = Signal(object, bool)

    def __init__(self, inst: ModuleInstance, has_src: bool, parent=None):
        super().__init__(parent)
        self.inst = inst
        self.setObjectName("moduleRow")
        h = QHBoxLayout(self)
        h.setContentsMargins(10, 6, 10, 6)
        h.setSpacing(8)
        handle = QLabel("☰")
        handle.setStyleSheet("color:#6a6c72;font-size:15px;")
        handle.setToolTip("拖动排序")
        h.addWidget(handle)
        col = QVBoxLayout()
        name = QLabel(inst.spec.name)
        name.setStyleSheet("font-weight:bold;")
        col.addWidget(name)
        self.summary_lab = QLabel(inst.summary())
        self.summary_lab.setStyleSheet("color:#9aa0a8;font-size:12px;")
        col.addWidget(self.summary_lab)
        h.addLayout(col, 1)
        if inst.spec.params:
            cfg = QPushButton("⚙ 参数")
            cfg.setToolTip("选模型 / 调参数")
            cfg.clicked.connect(self._open_config)
            h.addWidget(cfg)
        if has_src:
            pv = QPushButton("👁 预览")
            pv.setToolTip("渲染一段/一帧本模块的单独效果，用于对比选型")
            pv.clicked.connect(lambda: self.previewRequested.emit(self.inst))
            h.addWidget(pv)
        self.switch = QCheckBox()
        self.switch.setChecked(inst.enabled)
        self.switch.setToolTip("启用/停用（保留参数）")
        self.switch.toggled.connect(lambda v: self.toggled.emit(self.inst, v))
        h.addWidget(self.switch)
        rm = QPushButton("✕")
        rm.setFixedWidth(32)
        rm.setToolTip("从链中移除")
        rm.clicked.connect(lambda: self.removed.emit(self.inst))
        h.addWidget(rm)

    def _open_config(self):
        work = ModuleInstance(self.inst.key, copy.deepcopy(self.inst.params),
                              self.inst.enabled)
        dlg = ModuleParamDialog(work, self.window())
        if dlg.exec() == QDialog.Accepted:
            self.inst.params = work.params
            self.refresh()
            self.configApplied.emit(self.inst)

    def refresh(self):
        self.summary_lab.setText(self.inst.summary())
        self.switch.setChecked(self.inst.enabled)


class ModuleChainList(QWidget):
    """可拖拽排序的模块链 + 添加按钮。"""
    chainChanged = Signal()
    previewRequested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.task: Task | None = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        self.list.setDragDropMode(QListWidget.InternalMove)
        self.list.setDefaultDropAction(Qt.MoveAction)
        self.list.setSpacing(4)
        self.list.model().rowsMoved.connect(self._sync_order)
        v.addWidget(self.list, 1)
        bar = QHBoxLayout()
        self.add_btn = QToolButton()
        self.add_btn.setText("＋ 添加 AI 模块")
        self.add_btn.setPopupMode(QToolButton.InstantPopup)
        self.add_menu = QMenu(self.add_btn)
        self.add_btn.setMenu(self.add_menu)
        bar.addWidget(self.add_btn)
        bar.addStretch()
        v.addLayout(bar)

    def set_task(self, task: Task | None):
        self.task = task
        self.rebuild()

    def rebuild(self):
        self.list.blockSignals(True)
        self.list.clear()
        if self.task:
            for inst in self.task.modules:
                item = QListWidgetItem()
                item.setData(Qt.UserRole, inst)
                row = ModuleRow(inst, has_src=self.task is not None)
                row.previewRequested.connect(self.previewRequested.emit)
                row.configApplied.connect(lambda _: self.chainChanged.emit())
                row.removed.connect(self._remove)
                row.toggled.connect(self._toggle)
                item.setSizeHint(row.sizeHint())
                self.list.addItem(item)
                self.list.setItemWidget(item, row)
        self.list.blockSignals(False)
        self._refresh_menu()

    def _rows(self):
        return [self.list.item(i) for i in range(self.list.count())]

    def _sync_order(self, *args):
        if not self.task:
            return
        self.task.modules = [it.data(Qt.UserRole) for it in self._rows()]
        self.chainChanged.emit()

    def _remove(self, inst: ModuleInstance):
        if not self.task:
            return
        self.task.modules.remove(inst)
        self.rebuild()
        self.chainChanged.emit()

    def _toggle(self, inst: ModuleInstance, on: bool):
        inst.enabled = on
        self.chainChanged.emit()

    def _refresh_menu(self):
        self.add_menu.clear()
        if not self.task:
            return
        present = {m.key for m in self.task.modules}
        for spec in modules.available():
            act = self.add_menu.addAction(spec.name)
            act.setEnabled(True)
            if spec.key in present:
                act.setText(f"{spec.name}（链中已有）")
                act.triggered.connect(lambda _, k=spec.key: self._enable_existing(k))
            else:
                act.triggered.connect(lambda _, k=spec.key: self._add(k))

    def _enable_existing(self, key: str):
        if not self.task:
            return
        for m in self.task.modules:
            if m.key == key:
                m.enabled = True
        self.rebuild()
        self.chainChanged.emit()

    def _add(self, key: str):
        if not self.task:
            return
        inst = ModuleInstance(key)
        # 保持默认链序插入
        self.task.modules.append(inst)
        self.task.modules.sort(key=lambda m: m.spec.order)
        self.rebuild()
        self.chainChanged.emit()


# ---------------------------------------------------------------- 输出设置抽屉

class OutputSettingsDialog(QDialog):
    """详细输出设置：三 tab + 智能提示 + 配置文件（保存/加载/应用/应用到全部）。"""

    def __init__(self, task: Task, all_tasks: list[Task], parent=None):
        super().__init__(parent)
        self.task = task
        self.all_tasks = all_tasks
        self.cs = codec.CodecSettings.from_dict(task.codec_settings.to_dict())  # 编辑副本
        self.setWindowTitle("输出设置")
        self.setMinimumSize(520, 640)

        root = QVBoxLayout(self)
        hint = codec.smart_hint(self.cs, [m.key for m in task.enabled_modules()])
        self.hint_bar = QLabel(hint or "")
        self.hint_bar.setObjectName("hintBar")
        self.hint_bar.setWordWrap(True)
        self.hint_bar.setVisible(bool(hint))
        root.addWidget(self.hint_bar)

        tabs = QTabWidget()
        root.addWidget(tabs, 1)
        tabs.addTab(self._video_tab(), "视频")
        tabs.addTab(self._audio_tab(), "音频")
        tabs.addTab(self._sub_tab(), "字幕")

        # 容器（Remux 行）
        row = QHBoxLayout()
        row.addWidget(QLabel("封装容器"))
        self.container = QComboBox()
        self.container.addItem("MKV（推荐，字幕/多音轨原样）", "mkv")
        self.container.addItem("MP4（兼容性优先）", "mp4")
        self.container.setCurrentIndex(0 if self.cs.container == "mkv" else 1)
        self.container.currentIndexChanged.connect(
            lambda _: self._set("container", self.container.currentData()))
        row.addWidget(self.container)
        row.addStretch()
        root.addLayout(row)

        # 底部按钮
        btns = QHBoxLayout()
        load_btn = QPushButton("加载配置")
        load_btn.clicked.connect(self._load_profile)
        save_btn = QPushButton("保存为配置文件…")
        save_btn.clicked.connect(self._save_profile)
        btns.addWidget(load_btn)
        btns.addWidget(save_btn)
        btns.addStretch()
        apply_btn = QPushButton("应用")
        apply_btn.setObjectName("startBtn")
        apply_btn.clicked.connect(self._apply)
        apply_all_btn = QPushButton("应用到全部")
        apply_all_btn.clicked.connect(lambda: self._apply(all_tasks=True))
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(apply_btn)
        btns.addWidget(apply_all_btn)
        btns.addWidget(close_btn)
        root.addLayout(btns)
        self._refresh_hint()

    # —— 视频页 ——
    def _video_tab(self) -> QWidget:
        w = QWidget()
        g = QGridLayout(w)
        r = 0

        g.addWidget(QLabel("编码器"), r, 0)
        self.encoder = QComboBox()
        for spec in codec.ENCODERS.values():
            self.encoder.addItem(spec.name, spec.key)
        self.encoder.setCurrentIndex(max(0, list(codec.ENCODERS).index(self.cs.encoder)))
        self.encoder.currentIndexChanged.connect(self._on_encoder_change)
        g.addWidget(self.encoder, r, 1, 1, 2)
        r += 1

        g.addWidget(QLabel("质量模式"), r, 0)
        self.qmode = QComboBox()
        self.qmode.addItem("CRF 恒定画质（推荐）", "crf")
        self.qmode.addItem("平均码率", "abr")
        self.qmode.setCurrentIndex(0 if self.cs.quality_mode == "crf" else 1)
        self.qmode.currentIndexChanged.connect(
            lambda _: self._set("quality_mode", self.qmode.currentData()))
        g.addWidget(self.qmode, r, 1, 1, 2)
        r += 1

        self.crf_lab = QLabel()
        g.addWidget(self.crf_lab, r, 0)
        self.crf_slider = QSlider(Qt.Horizontal)
        spec = self.cs.spec()
        self.crf_slider.setRange(spec.crf_range[0], spec.crf_range[1])
        self.crf_slider.setValue(self.cs.crf)
        self.crf_slider.valueChanged.connect(
            lambda v: (self._set("crf", v), self.crf_lab.setText(f"CRF {v}")))
        g.addWidget(self.crf_slider, r, 1, 1, 2)
        r += 1
        self.crf_notes = QLabel(self._crf_notes())
        self.crf_notes.setStyleSheet("color:#8a8f98;font-size:11px;")
        g.addWidget(self.crf_notes, r, 1, 1, 2)
        r += 1

        self.preset_lab = QLabel()
        g.addWidget(self.preset_lab, r, 0)
        self.preset_slider = QSlider(Qt.Horizontal)
        self.preset_slider.setRange(spec.preset_range[0], spec.preset_range[1])
        self.preset_slider.setValue(self.cs.preset)
        self.preset_slider.valueChanged.connect(
            lambda v: (self._set("preset", v), self.preset_lab.setText(f"Preset {v}")))
        g.addWidget(self.preset_slider, r, 1, 1, 2)
        r += 1

        g.addWidget(QLabel("位深"), r, 0)
        self.depth = QComboBox()
        for d in spec.depths:
            self.depth.addItem(f"{d}bit", d)
        self.depth.setCurrentIndex(max(0, list(spec.depths).index(self.cs.bit_depth))
                                   if self.cs.bit_depth in spec.depths else 0)
        self.depth.currentIndexChanged.connect(
            lambda _: self._set("bit_depth", self.depth.currentData()))
        g.addWidget(self.depth, r, 1, 1, 2)
        r += 1

        g.addWidget(QLabel("分辨率"), r, 0)
        h = QHBoxLayout()
        self.res_mode = QComboBox()
        self.res_mode.addItem("与源相同", "source")
        self.res_mode.addItem("自定义", "custom")
        self.res_mode.setCurrentIndex(0 if self.cs.res_mode == "source" else 1)
        self.res_mode.currentIndexChanged.connect(
            lambda _: self._set("res_mode", self.res_mode.currentData()))
        self.res_edit = QLineEdit(self.cs.res_custom)
        self.res_edit.setPlaceholderText("3840x2160")
        self.res_edit.editingFinished.connect(
            lambda: self._set("res_custom", self.res_edit.text().strip().lower()))
        h.addWidget(self.res_mode)
        h.addWidget(self.res_edit, 1)
        g.addLayout(h, r, 1, 1, 2)
        r += 1

        g.addWidget(QLabel("帧率"), r, 0)
        h = QHBoxLayout()
        self.fps_mode = QComboBox()
        self.fps_mode.addItem("与源相同（插帧自动翻倍）", "source")
        self.fps_mode.addItem("自定义", "custom")
        self.fps_mode.setCurrentIndex(0 if self.cs.fps_mode == "source" else 1)
        self.fps_mode.currentIndexChanged.connect(
            lambda _: self._set("fps_mode", self.fps_mode.currentData()))
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(1.0, 480.0)
        self.fps_spin.setDecimals(3)
        self.fps_spin.setValue(self.cs.fps_custom)
        self.fps_spin.valueChanged.connect(
            lambda v: self._set("fps_custom", float(v)))
        h.addWidget(self.fps_mode)
        h.addWidget(self.fps_spin, 1)
        g.addLayout(h, r, 1, 1, 2)
        r += 1

        self.deint = QCheckBox("反交错（bwdif，隔行源才需要）")
        self.deint.setChecked(self.cs.deinterlace)
        self.deint.toggled.connect(lambda v: self._set("deinterlace", v))
        g.addWidget(self.deint, r, 0, 1, 3)
        r += 1

        g.setRowStretch(r, 1)
        g.setColumnMinimumWidth(1, 120)
        self._sync_labels()
        return w

    def _audio_tab(self) -> QWidget:
        w = QWidget()
        g = QGridLayout(w)
        g.addWidget(QLabel("音轨"), 0, 0)
        self.audio_mode = QComboBox()
        self.audio_mode.addItem("原样复制（推荐）", "copy")
        self.audio_mode.addItem("转 AAC", "aac")
        self.audio_mode.setCurrentIndex(0 if self.cs.audio_mode == "copy" else 1)
        self.audio_mode.currentIndexChanged.connect(
            lambda _: self._set("audio_mode", self.audio_mode.currentData()))
        g.addWidget(self.audio_mode, 0, 1)
        g.addWidget(QLabel("AAC 码率"), 1, 0)
        self.audio_br = QSpinBox()
        self.audio_br.setRange(64, 512)
        self.audio_br.setSuffix(" kbps")
        self.audio_br.setValue(self.cs.audio_bitrate_kbps)
        self.audio_br.valueChanged.connect(
            lambda v: self._set("audio_bitrate_kbps", int(v)))
        g.addWidget(self.audio_br, 1, 1)
        note = QLabel("AAC/AC3/EAC3/DTS/FLAC/OPUS 均可直接复制；仅极个别格式需要转码。")
        note.setStyleSheet("color:#8a8f98;font-size:11px;")
        note.setWordWrap(True)
        g.addWidget(note, 2, 0, 1, 2)
        g.setRowStretch(3, 1)
        return w

    def _sub_tab(self) -> QWidget:
        w = QWidget()
        g = QGridLayout(w)
        g.addWidget(QLabel("字幕轨"), 0, 0)
        self.sub_mode = QComboBox()
        self.sub_mode.addItem("复制（MP4 自动转 mov_text）", "copy")
        self.sub_mode.addItem("丢弃", "drop")
        self.sub_mode.setCurrentIndex(0 if self.cs.subtitle_mode == "copy" else 1)
        self.sub_mode.currentIndexChanged.connect(
            lambda _: self._set("subtitle_mode", self.sub_mode.currentData()))
        g.addWidget(self.sub_mode, 0, 1)
        note = QLabel("图形字幕（PGS/VOBSub）无法转 mov_text，含图形字幕的源建议 MKV 容器。")
        note.setStyleSheet("color:#8a8f98;font-size:11px;")
        note.setWordWrap(True)
        g.addWidget(note, 1, 0, 1, 2)
        g.setRowStretch(2, 1)
        return w

    # —— 状态同步 ——
    def _set(self, key: str, value):
        setattr(self.cs, key, value)
        if key in ("encoder", "crf", "quality_mode") or key == "bit_depth":
            self._refresh_hint()

    def _on_encoder_change(self):
        key = self.encoder.currentData()
        self.cs.encoder = key
        # 切编码器：按确认方案，切换即套用该编码器的推荐默认（crf/preset/位深）
        fresh = codec.CodecSettings.for_encoder(key)
        for k in ("crf", "preset", "bit_depth"):
            setattr(self.cs, k, getattr(fresh, k))
        spec = self.cs.spec()
        self.crf_slider.blockSignals(True)
        self.crf_slider.setRange(spec.crf_range[0], spec.crf_range[1])
        self.crf_slider.setValue(self.cs.crf)
        self.crf_slider.blockSignals(False)
        self.preset_slider.blockSignals(True)
        self.preset_slider.setRange(spec.preset_range[0], spec.preset_range[1])
        self.preset_slider.setValue(self.cs.preset)
        self.preset_slider.blockSignals(False)
        self.depth.blockSignals(True)
        self.depth.clear()
        for d in spec.depths:
            self.depth.addItem(f"{d}bit", d)
        self.depth.setCurrentIndex(list(spec.depths).index(self.cs.bit_depth))
        self.depth.blockSignals(False)
        self._sync_labels()
        self._refresh_hint()

    def _sync_labels(self):
        spec = self.cs.spec()
        self.crf_lab.setText(f"CRF {self.cs.crf}")
        self.preset_lab.setText(f"Preset {self.cs.preset}")
        self.crf_notes.setText(self._crf_notes())

    def _crf_notes(self) -> str:
        spec = self.cs.spec()
        if not spec.crf_notes:
            return f"范围 {spec.crf_range[0]}-{spec.crf_range[1]}，越小质量越高"
        return "  ".join(f"{v}={n}" for v, n in sorted(spec.crf_notes.items())) + \
               "（越小质量越高）"

    def _refresh_hint(self):
        text = codec.smart_hint(self.cs, [m.key for m in self.task.enabled_modules()])
        self.hint_bar.setText(text or "")
        self.hint_bar.setVisible(bool(text))

    # —— 配置文件 / 应用 ——
    def _save_profile(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "保存配置文件", "配置名称：",
                                        text=self.task.src.stem)
        if not ok or not name.strip():
            return
        prof = profiles.TaskProfile(name.strip(),
                                    [ModuleInstance(m.key, dict(m.params), m.enabled)
                                     for m in self.task.enabled_modules()] or
                                    [ModuleInstance("sr")],
                                    self.cs)
        profiles.save_profile(prof)
        self.log_and_close_msg = f"配置已保存: {name.strip()}"

    def _load_profile(self):
        files = profiles.list_profiles()
        if not files:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "加载配置", "还没有保存过配置文件")
            return
        from PySide6.QtWidgets import QInputDialog
        names = [f.stem for f in files]
        name, ok = QInputDialog.getItem(self, "加载配置文件", "选择配置：",
                                        names, 0, False)
        if not ok:
            return
        prof = profiles.load_profile(files[names.index(name)])
        self.cs = codec.CodecSettings.from_dict(prof.codec.to_dict())
        # 配置里的模块链也一并恢复（替换当前任务链）
        if prof.modules:
            self.task.modules = [ModuleInstance(m.key, dict(m.params), m.enabled)
                                 for m in prof.modules]
        self.encoder.blockSignals(True)
        self.encoder.setCurrentIndex(max(0, list(codec.ENCODERS).index(self.cs.encoder)))
        self.encoder.blockSignals(False)
        self._on_encoder_change()
        self.chain_dirty = True

    def _apply(self, all_tasks: bool = False):
        targets = self.all_tasks if all_tasks else [self.task]
        for t in targets:
            t.codec_settings = codec.CodecSettings.from_dict(self.cs.to_dict())
        self.accept()


# ---------------------------------------------------------------- 首页

class HomePage(QWidget):
    filesChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(48, 48, 48, 24)
        title = QLabel("把视频拖进来")
        title.setStyleSheet("font-size:22px;font-weight:bold;")
        title.setAlignment(Qt.AlignCenter)
        v.addWidget(title)
        sub = QLabel("超分 · 插帧 · 降噪 · 去压缩 —— 每个任务可自由组合 AI 模块链")
        sub.setStyleSheet("color:#9aa0a8;")
        sub.setAlignment(Qt.AlignCenter)
        v.addWidget(sub)

        self.drop = QPushButton("＋ 拖入视频文件，或点击选择")
        self.drop.setObjectName("dropZone")
        self.drop.setFixedHeight(220)
        self.drop.clicked.connect(self._pick)
        v.addWidget(self.drop)

        self.file_list = QListWidget()
        self.file_list.setVisible(False)
        v.addWidget(self.file_list, 1)

        bar = QHBoxLayout()
        add = QPushButton("添加文件")
        add.clicked.connect(self._pick)
        clear = QPushButton("清空")
        clear.clicked.connect(self._clear)
        bar.addWidget(add)
        bar.addWidget(clear)
        bar.addStretch()
        go = QPushButton("下一步：配置任务 →")
        go.setObjectName("startBtn")
        go.clicked.connect(self.filesChanged.emit)
        bar.addWidget(go)
        v.addLayout(bar)
        self.go_btn = go
        self.setAcceptDrops(True)

    def _pick(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择视频", "", "视频 (*.mp4 *.mkv *.avi *.webm *.mov *.ts *.m2ts *.flv)")
        self.add_files(files)

    def add_files(self, files):
        for f in files:
            p = Path(f)
            if p.suffix.lower() in VIDEO_EXTS and p.exists():
                self.file_list.addItem(str(p))
        self.file_list.setVisible(self.file_list.count() > 0)
        self.go_btn.setEnabled(self.file_list.count() > 0)

    def _clear(self):
        self.file_list.clear()
        self.file_list.setVisible(False)
        self.go_btn.setEnabled(False)

    def paths(self) -> list[Path]:
        return [Path(self.file_list.item(i).text())
                for i in range(self.file_list.count())]

    # 整窗拖拽
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_files([u.toLocalFile() for u in e.mimeData().urls()])


# ---------------------------------------------------------------- 任务页

class TaskPage(QWidget):
    backRequested = Signal()
    startRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tasks: list[Task] = []
        self.current: Task | None = None
        self._building = False

        v = QVBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        v.addWidget(split, 1)

        # 左：任务列表
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QLabel("任务队列"))
        self.task_list = QListWidget()
        self.task_list.currentRowChanged.connect(self._on_select)
        lv.addWidget(self.task_list)
        back = QPushButton("← 添加更多文件")
        back.clicked.connect(self.backRequested.emit)
        lv.addWidget(back)
        split.addWidget(left)

        # 右：任务详情
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        self.header = QFrame()
        self.header.setObjectName("taskHeader")
        hv = QHBoxLayout(self.header)
        head_col = QVBoxLayout()
        self.name_lab = QLabel()
        self.name_lab.setStyleSheet("font-weight:bold;font-size:15px;")
        head_col.addWidget(self.name_lab)
        self.io_lab = QLabel()
        self.io_lab.setStyleSheet("color:#9aa0a8;font-size:12px;")
        head_col.addWidget(self.io_lab)
        hv.addLayout(head_col, 1)
        self.badge = QLabel("就绪")
        self.badge.setStyleSheet("color:#5fd08a;font-weight:bold;")
        hv.addWidget(self.badge)
        del_btn = QPushButton("删除任务")
        del_btn.clicked.connect(self._delete_current)
        hv.addWidget(del_btn)
        rv.addWidget(self.header)

        rv.addWidget(QLabel("AI 模块链（☰ 拖动排序）"))
        self.chain = ModuleChainList()
        self.chain.chainChanged.connect(self._on_chain_changed)
        self.chain.previewRequested.connect(self._open_preview)
        rv.addWidget(self.chain, 1)

        # 底部：输出/开始
        bar = QHBoxLayout()
        self.settings_btn = QPushButton("⚙ 输出设置")
        self.settings_btn.clicked.connect(self._open_settings)
        bar.addWidget(self.settings_btn)
        self.out_lab = QLabel()
        self.out_lab.setStyleSheet("color:#9aa0a8;font-size:12px;")
        bar.addWidget(self.out_lab, 1)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setEnabled(False)
        bar.addWidget(self.stop_btn)
        self.start_btn = QPushButton("开始处理")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self.startRequested.emit)
        bar.addWidget(self.start_btn)
        rv.addLayout(bar)

        self.bar = QProgressBar()
        rv.addWidget(self.bar)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(120)
        rv.addWidget(self.log_box)
        split.addWidget(right)
        split.setSizes([300, 720])

    # —— 任务管理 ——
    def add_tasks(self, paths: list[Path]) -> int:
        n = 0
        for p in paths:
            if any(t.src == p for t in self.tasks):
                continue
            try:
                info = probe(p)
            except Exception as e:  # noqa: BLE001
                self.log_box.append(f"[跳过] {p.name}: 无法读取（{e}）")
                continue
            self.tasks.append(Task(src=p, info=info))
            n += 1
        self.refresh_task_list()
        return n

    def refresh_task_list(self):
        self.task_list.blockSignals(True)
        cur = self.current
        self.task_list.clear()
        for t in self.tasks:
            self.task_list.addItem(QListWidgetItem(t.src.name))
        if cur in self.tasks:
            self.task_list.setCurrentRow(self.tasks.index(cur))
        self.task_list.blockSignals(False)

    def _on_select(self, row):
        if 0 <= row < len(self.tasks):
            self.show_task(self.tasks[row])

    def show_task(self, task: Task):
        self.current = task
        self._building = True
        self.chain.set_task(task)
        self._building = False
        self.refresh_header()
        self.refresh_task_list()

    def refresh_header(self):
        t = self.current
        if not t:
            self.name_lab.setText("（无任务）")
            return
        w, h = t.out_size()
        self.name_lab.setText(t.src.name)
        self.io_lab.setText(
            f"INPUT  {t.info.width}x{t.info.height} · {t.info.fps:.0f}fps · "
            f"{t.info.duration:.0f}s        "
            f"OUTPUT → {w}x{h} · {t.out_fps_value():.0f}fps · "
            f"{t.codec_settings.spec().name.split('（')[0]}")
        self.out_lab.setText(f"输出: {t.out_path()}")

    def _on_chain_changed(self):
        if self._building or not self.current:
            return
        self.refresh_header()

    def _delete_current(self):
        if not self.current:
            return
        idx = self.tasks.index(self.current)
        self.tasks.remove(self.current)
        self.current = None
        self.refresh_task_list()
        if self.tasks:
            self.show_task(self.tasks[min(idx, len(self.tasks) - 1)])
        else:
            self.chain.set_task(None)
            self.refresh_header()

    def _open_preview(self, inst: ModuleInstance):
        from gui_preview import PreviewDialog
        if not self.current:
            return
        dlg = PreviewDialog(self.current.src, inst, self)
        dlg.exec()

    def _open_settings(self):
        if not self.current:
            return
        dlg = OutputSettingsDialog(self.current, self.tasks, self)
        if dlg.exec() == QDialog.Accepted:
            self.refresh_task_list()
            self.refresh_header()

    # —— 运行状态 ——
    def set_running(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.settings_btn.setEnabled(not running)
        self.add_btn_enabled(not running)

    def add_btn_enabled(self, on: bool):
        self.chain.add_btn.setEnabled(on)
        for i in range(self.chain.list.count()):
            w = self.chain.list.itemWidget(self.chain.list.item(i))
            if isinstance(w, ModuleRow):
                w.switch.setEnabled(on)

    def log(self, msg: str):
        self.log_box.append(msg)


# ---------------------------------------------------------------- 主窗口

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1120, 720)
        self.worker: Worker | None = None

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.home = HomePage()
        self.task_page = TaskPage()
        self.stack.addWidget(self.home)
        self.stack.addWidget(self.task_page)
        self.home.filesChanged.connect(self._go_tasks)
        self.task_page.backRequested.connect(
            lambda: self.stack.setCurrentWidget(self.home))
        self.task_page.startRequested.connect(self._start)
        self.task_page.stop_btn.clicked.connect(self._stop)

    def _go_tasks(self):
        paths = self.home.paths()
        if not paths:
            return
        n = self.task_page.add_tasks(paths)
        self.home._clear()
        if self.task_page.current is None and self.task_page.tasks:
            self.task_page.show_task(self.task_page.tasks[0])
        self.stack.setCurrentWidget(self.task_page)
        if n:
            self.task_page.log(f"已添加 {n} 个任务")

    def _start(self):
        if self.worker or not self.task_page.tasks:
            return
        tasks = [t for t in self.task_page.tasks
                 if t.status in ("就绪", "失败")] or self.task_page.tasks
        for t in tasks:
            t.status = "排队中"
        self.task_page.refresh_task_list()
        self.worker = Worker(tasks)
        self.worker.taskProgress.connect(self._on_progress)
        self.worker.log.connect(self.task_page.log)
        self.worker.taskDone.connect(self._on_task_done)
        self.worker.allDone.connect(self._on_all_done)
        self.task_page.bar.setValue(0)
        self.task_page.set_running(True)
        self.worker.start()

    def _stop(self):
        if self.worker:
            self.worker.cancel()

    def _on_progress(self, name: str, done: int, total: int):
        if total:
            self.task_page.bar.setMaximum(total)
            self.task_page.bar.setValue(done)

    def _on_task_done(self, task: Task, ok: bool, msg: str):
        task.status = "完成" if ok else "失败"
        self.task_page.refresh_task_list()

    def _on_all_done(self, ok: bool, msg: str):
        self.task_page.log(f"— {msg} —")
        self.task_page.set_running(False)
        self.task_page.bar.setValue(0)
        self.worker = None


def _excepthook(t, v, tb):
    """双击启动（pythonw 无控制台）时把崩溃留档，否则错误无声消失。"""
    import traceback
    (Path(__file__).parent / "crash.log").write_text(
        "".join(traceback.format_exception(t, v, tb)), encoding="utf-8")
    sys.__stderr__ and sys.__stderr__.write(str(v))


def main():
    sys.excepthook = _excepthook
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
