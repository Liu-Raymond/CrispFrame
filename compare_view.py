"""对比检视器：两路画面（原片 / 渲染结果）全屏对比。

- 槽位选择：视频1/视频2 两个下拉（原片固定 + A/B/C 渲染位），可一键交换
- 视频对比走原生 QVideoWidget（GPU 合成，全帧率全分辨率，60fps/4K 无压力）：
  · 擦除：两路同位叠加，顶层 widget 用 QRegion mask 裁到分割线左侧，
    分割线跟随鼠标（左=视频2，右=视频1；扫到最右全是视频2，最左全是视频1）
  · 并排：左右各一个 QVideoWidget
- 图片对比（渲染此帧）：图形场景像素级对比，支持擦除/并排/缩放
- 视频同步：双 QMediaPlayer 内容偏移对齐（250ms 检查），视频1 出声、视频2 静音；
  混选（一帧一段）时把视频侧取一帧转成图片对比
- 静帧细节对比：暂停后扫动擦除；像素级放大用「渲染此帧」+ 槽位图片对比
- 全屏：按钮 / 双击 / F11，Esc 先退全屏再关对话框
"""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QEventLoop, QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QColor, QPainter, QPen, QPixmap, QRegion, QTransform
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QComboBox, QDialog, QGraphicsLineItem, QGraphicsPixmapItem, QGraphicsScene,
    QGraphicsView, QHBoxLayout, QLabel, QPushButton, QStackedWidget,
    QVBoxLayout, QWidget,
)

SLOT_KEYS = ["A", "B", "C"]          # 渲染位（原片单独占第一位，不占字母）
GAP_RATIO = 0.02                     # 并排模式间隔（相对宽度）
SYNC_INTERVAL_MS = 250
SYNC_TOLERANCE_MS = 300          # 容差放宽 + 连续 2 tick 超阈值才纠正（滞回，防 seek 风暴）


# ---------------------------------------------------------------- 场景项（图片/暂停帧快照）

class _ClipPixmapItem(QGraphicsPixmapItem):
    """可矩形裁剪的图片项（擦除模式把视频2裁到分割线左侧）。"""

    def __init__(self, pixmap: QPixmap):
        super().__init__(pixmap)
        self.clip_rect: QRectF | None = None

    def paint(self, painter, option, widget=None):
        if self.clip_rect is not None:
            painter.save()
            painter.setClipRect(self.clip_rect)
            super().paint(painter, option, widget)
            painter.restore()
        else:
            super().paint(painter, option, widget)


class _CompareView(QGraphicsView):
    """滚轮缩放（光标锚点）+ 左键拖拽平移 + 悬停回调 + 双击全屏。"""
    hoverScene = Signal(object)      # QPointF（仅无按键按下时发射）
    doubleClicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setMouseTracking(True)   # 悬停（无按键）也要收到 mouseMoveEvent
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._zoom = 1.0

    def fit_content(self):
        self._zoom = 1.0
        self.resetTransform()
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, ev):
        if self.scene() is None or self.sceneRect().isEmpty():
            return
        factor = 1.25 if ev.angleDelta().y() > 0 else 0.8
        new_zoom = self._zoom * factor
        if new_zoom < 1.0:
            self.fit_content()
            return
        new_zoom = min(new_zoom, 40.0)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(new_zoom / self._zoom, new_zoom / self._zoom)
        self._zoom = new_zoom

    def mouseMoveEvent(self, ev):
        if not ev.buttons():
            self.hoverScene.emit(self.mapToScene(ev.position().toPoint()))
        super().mouseMoveEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(ev)


class _VideoPage(QWidget):
    """原生视频对比页：擦除（顶层 mask）或并排。悬停驱动分割线，GPU 合成零转换成本。

    实时缩放：滚轮缩放（光标锚点）+ 左键拖拽平移——把 QVideoWidget 的几何
    放大 z 倍并负偏移，父级裁剪溢出，视频内容由 GPU 等比放大，无 CPU 成本。
    播放中即可用，无需暂停（像素级精查走「渲染此帧」槽位图片对比）。
    """
    splitMoved = Signal(float)
    layoutChanged = Signal()             # 遮罩边缘可能移动（缩放/平移/尺寸变化）
    MAX_ZOOM = 8.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.mode = "wipe"
        self.split = 0.5
        self.zoom = 1.0
        self.pan = QPointF(0.0, 0.0)          # 容器像素（当前 zoom 下）
        self._drag = None                     # (press_pos, pan_at_press)
        self._last_layout_key = None
        self.w1 = QVideoWidget(self)          # 视频1（擦除时在底层，显示右半）
        self.w2 = QVideoWidget(self)          # 视频2（擦除时在顶层，mask 裁左半）
        # 鼠标事件全部交给页面（悬停擦除/滚轮缩放/拖拽平移），视频件不消费
        for w in (self.w1, self.w2):
            w.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        # 分割线不再由本页管理：QVideoWidget 的视频呈现面会盖住窗内一切子控件/
        # 兄弟窗口（tool 窗定位簿记也不可信），改由 CompareDialog 创建为顶层
        # 对话框的直接原生子控件（与视频件同一 HWND 树，SetWindowPos z 序可靠），
        # 位置经 mapToGlobal/mapFromGlobal 锚定 w2 遮罩边缘（DPR 感知）。
        self.line = None

    # —— 布局 ——
    def _base_rects(self):
        W, H = max(self.width(), 1), max(self.height(), 1)
        if self.mode == "wipe":
            return QRectF(0, 0, W, H), QRectF(0, 0, W, H)
        gap = round(W * GAP_RATIO)
        left_w = (W - gap) // 2
        return QRectF(0, 0, left_w, H), QRectF(left_w + gap, 0, W - left_w - gap, H)

    def _clamp_pan(self):
        b1, _ = self._base_rects()
        self.pan = QPointF(max(0.0, min(self.pan.x(), b1.width() * (self.zoom - 1))),
                           max(0.0, min(self.pan.y(), b1.height() * (self.zoom - 1))))

    def set_mode(self, mode: str):
        self.mode = mode
        self.w2.clearMask()
        self._clamp_pan()
        self.relayout()

    def set_split(self, ratio: float):
        self.split = max(0.0, min(ratio, 1.0))
        self.relayout()

    def set_zoom(self, z: float, anchor=None):
        """anchor：容器坐标的缩放中心（默认页面中心）。"""
        z = max(1.0, min(z, self.MAX_ZOOM))
        if anchor is None:
            anchor = QPointF(self.width() / 2, self.height() / 2)
        old = self.zoom
        if z == old:
            return
        self.pan = QPointF((anchor.x() + self.pan.x()) * z / old - anchor.x(),
                           (anchor.y() + self.pan.y()) * z / old - anchor.y())
        self.zoom = z
        if self.zoom == 1.0:
            self.pan = QPointF(0, 0)
        self._clamp_pan()
        self.relayout()

    def relayout(self):
        # 布局去重：悬停/微小抖动下 setMask/setGeometry 全部跳过（零开销）
        key = (self.mode, round(self.split * 1000), round(self.zoom * 100),
               round(self.pan.x()), round(self.pan.y()), self.width(), self.height())
        if key == self._last_layout_key:
            return
        self._last_layout_key = key
        b1, b2 = self._base_rects()
        z = self.zoom
        px, py = self.pan.x(), self.pan.y()
        self.w1.setGeometry(round(b1.x() - px), round(b1.y() - py),
                            round(b1.width() * z), round(b1.height() * z))
        self.w2.setGeometry(round(b2.x() - px), round(b2.y() - py),
                            round(b2.width() * z), round(b2.height() * z))
        if self.mode == "wipe":
            split_px = round(self.width() * self.split)
            # mask 坐标 = 容器分割线 − w2 原点，裁到 w2 范围内
            mx = split_px - self.w2.x()
            mw = max(0, min(mx, self.w2.width()))
            if mw > 0:
                self.w2.setMask(QRegion(0, 0, mw, self.w2.height()))
            else:
                self.w2.setMask(QRegion())
        else:
            self.w2.clearMask()
        self.layoutChanged.emit()

    def resizeEvent(self, ev):
        self._clamp_pan()
        self.relayout()

    # —— 交互 ——
    def wheelEvent(self, ev):
        d = ev.angleDelta().y()
        if d:
            self.set_zoom(self.zoom * (1.25 if d > 0 else 0.8), anchor=ev.position())

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and self.zoom > 1.0:
            self._drag = (ev.position(), QPointF(self.pan))

    def mouseMoveEvent(self, ev):
        if self._drag is not None and (ev.buttons() & Qt.LeftButton):
            press, pan0 = self._drag
            self.pan = QPointF(pan0.x() - (ev.position().x() - press.x()),
                               pan0.y() - (ev.position().y() - press.y()))
            self._clamp_pan()
            self.relayout()
        elif self.mode == "wipe":
            self.set_split(ev.position().x() / max(self.width(), 1))
            self.splitMoved.emit(self.split)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._drag = None
        super().mouseReleaseEvent(ev)


# ---------------------------------------------------------------- 对话框

class CompareDialog(QDialog):
    MODE_WIPE = "wipe"
    MODE_SIDE = "side"

    def __init__(self, preview):
        """preview: gui_preview.PreviewDialog（取 src/info/slots/当前时间）。"""
        super().__init__(preview)
        self.preview = preview
        self.setWindowTitle("对比器")
        self.resize(1280, 800)
        self.mode = self.MODE_WIPE
        self._video_mode = False
        self._playing = False
        self._split = 0.5                  # 分割线相对位置 0-1
        self._W = self._H = 0.0            # 图片场景基准尺寸
        self._entries: list[dict] = []
        self._p1: QMediaPlayer | None = None
        self._p2: QMediaPlayer | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # 顶部：槽位选择 + 模式 + 播放 + 全屏
        bar = QHBoxLayout()
        bar.addWidget(QLabel("视频1（擦除右侧）"))
        self.combo1 = QComboBox()
        bar.addWidget(self.combo1)
        swap = QPushButton("⇄")
        swap.setFixedWidth(34)
        swap.setToolTip("交换两路")
        swap.clicked.connect(self._swap)
        bar.addWidget(swap)
        bar.addWidget(QLabel("视频2（擦除左侧）"))
        self.combo2 = QComboBox()
        bar.addWidget(self.combo2)
        self.wipe_btn = QPushButton("擦除叠加")
        self.wipe_btn.setCheckable(True)
        self.wipe_btn.clicked.connect(lambda: self._set_mode(self.MODE_WIPE))
        bar.addWidget(self.wipe_btn)
        self.side_btn = QPushButton("左右并排")
        self.side_btn.setCheckable(True)
        self.side_btn.clicked.connect(lambda: self._set_mode(self.MODE_SIDE))
        bar.addWidget(self.side_btn)
        self.play_btn = QPushButton("▶ 播放")
        self.play_btn.setVisible(False)
        self.play_btn.clicked.connect(self._toggle_play)
        bar.addWidget(self.play_btn)
        self.audio_btn = QPushButton("🔇")
        self.audio_btn.setCheckable(True)
        self.audio_btn.setToolTip("对比默认静音（静音时播放更流畅）；开启后播放带音轨一侧的声音")
        self.audio_btn.setEnabled(False)
        self.audio_btn.clicked.connect(self._toggle_audio)
        bar.addWidget(self.audio_btn)
        self.loop_btn = QPushButton("↻ 循环")
        self.loop_btn.setCheckable(True)
        self.loop_btn.setChecked(True)
        self.loop_btn.setVisible(False)
        self.loop_btn.setToolTip("窗口循环播放（每圈有轻微回卷停顿）；关闭后播完定住最后一帧，"
                                 "按 ▶ 从头重看——完全无停顿")
        self.loop_btn.toggled.connect(lambda v: setattr(self, "loop_enabled", v))
        bar.addWidget(self.loop_btn)
        bar.addStretch()
        self.fs_btn = QPushButton("⛶ 全屏")
        self.fs_btn.clicked.connect(self._toggle_fs)
        bar.addWidget(self.fs_btn)
        close = QPushButton("关闭")
        close.clicked.connect(self.reject)
        bar.addWidget(close)
        root.addLayout(bar)

        # 中央：视频页（原生） / 图形页（图片）
        self.video_page = _VideoPage()
        self.video_page.splitMoved.connect(self._on_video_split)
        self.view = _CompareView()
        self.view.hoverScene.connect(self._on_hover)
        self.view.doubleClicked.connect(self._toggle_fs)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.video_page)   # index 0: 原生视频
        self.stack.addWidget(self.view)         # index 1: 图片场景
        root.addWidget(self.stack, 1)

        # 原生视频模式的分割线 = 无父窗口的置顶 Tool 小窗（每对话框一次）。
        # QVideoWidget 的视频呈现面会盖住同窗树内的一切兄弟/子窗口（z 序抢不过），
        # 唯一可靠的是独立置顶顶层窗口（DWM 保证）；无父窗口 → setGeometry 即全局
        # 逻辑坐标，配合 mapToGlobal 无换算歧义；应用失活/对话框隐藏自动隐藏。
        self._line_overlay = QWidget(None, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self._line_overlay.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self._line_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._line_overlay.setFixedWidth(5)
        self._line_overlay.setStyleSheet(
            "background:#ffffff; border-left:1px solid #000000; border-right:1px solid #000000;")
        self._line_overlay.hide()
        QGuiApplication.instance().applicationStateChanged.connect(self._update_line_window)

        self.hint = QLabel("")
        self.hint.setStyleSheet("color:#8a8f98;font-size:12px;")
        root.addWidget(self.hint)

        self.wipe_btn.setChecked(True)
        self._fill_combos()
        self._setup_compare()

    # ---------------------------------------------------------------- 槽位
    def _fill_combos(self):
        """视频1/视频2 下拉：原片 + A/B/C 渲染位。默认 原片 vs 最新渲染。"""
        def items():
            out = [(self.preview.ORIG_KEY, "原片")]
            for k in SLOT_KEYS:
                s = self.preview.slots.get(k)
                if s:
                    out.append((k, f"[{k}] {s['label'][:26]}"))
            return out

        self._entries = items()
        for combo in (self.combo1, self.combo2):
            combo.blockSignals(True)
            combo.clear()
            for key, disp in self._entries:
                combo.addItem(disp, key)
            combo.blockSignals(False)
        last_render = getattr(self.preview, "last_render_key", None)
        if last_render not in [k for k, _ in self._entries]:
            last_render = next((k for k in reversed(SLOT_KEYS)
                                if k in self.preview.slots), None)
        self.combo1.setCurrentIndex(0)                       # 原片
        idx2 = self.combo2.findData(last_render) if last_render else 0
        self.combo2.setCurrentIndex(max(0, idx2))
        self.combo1.currentIndexChanged.connect(self._on_combo)
        self.combo2.currentIndexChanged.connect(self._on_combo)

    def _slot_entry(self, key: str) -> dict:
        if key == self.preview.ORIG_KEY:
            # 原片 = source：既能当图片（取帧）也能当视频（按另一侧窗口播）
            return {"key": key, "label": "原片", "kind": "source",
                    "payload": None, "t0": self.preview._current_t(),
                    "dur": 0.0, "src": self.preview.src, "lazy": True}
        s = dict(self.preview.slots[key])
        s["key"] = key
        s.setdefault("src", self.preview.src)
        return s

    def _swap(self):
        i1, i2 = self.combo1.currentIndex(), self.combo2.currentIndex()
        self.combo1.setCurrentIndex(i2)
        self.combo2.setCurrentIndex(i1)

    def _on_combo(self):
        self._stop_media()
        self._setup_compare()

    # ---------------------------------------------------------------- 装配
    def _setup_compare(self):
        e1 = self._slot_entry(self.combo1.currentData())
        e2 = self._slot_entry(self.combo2.currentData())
        playable = ("seg", "source")
        # 视频对比：两侧都能播视频（source 恒可）且至少一侧是真渲染段
        self._video_mode = (e1["kind"] in playable and e2["kind"] in playable
                            and "seg" in (e1["kind"], e2["kind"]))
        self.play_btn.setVisible(self._video_mode)
        self._playing = False
        self._over_budget = 0
        self.loop_enabled = self.loop_btn.isChecked()
        self._last_rewind = 0.0
        self.play_btn.setText("▶ 播放")

        self._line = QGraphicsLineItem()
        pen = QPen(QColor(255, 255, 255, 210), 3)
        pen.setCosmetic(True)               # 缩放时线宽恒定
        self._line.setPen(pen)
        self._line.setZValue(10)

        if self._video_mode:
            self._setup_video(e1, e2)
        else:
            self._setup_image(e1, e2)
        self._set_mode(self.mode)

    def _setup_image(self, e1: dict, e2: dict):
        """图片对比：混选时把视频侧取一帧（对齐另一侧的时间点）。"""
        self.audio_btn.setVisible(False)      # 视频专属功能
        self.loop_btn.setVisible(False)
        self.play_btn.setVisible(False)
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            pm1 = self._resolve_pixmap(e1, prefer_t=e2.get("t0", 0.0))
            pm2 = self._resolve_pixmap(e2, prefer_t=e1.get("t0", 0.0))
        finally:
            QApplication.restoreOverrideCursor()
        self._W, self._H = float(pm1.width()), float(pm1.height())
        self.item1 = _ClipPixmapItem(pm1)            # 底层 = 视频1
        self.item2 = _ClipPixmapItem(pm2)            # 顶层 = 视频2（裁剪）
        self.scene = QGraphicsScene(self)
        self.scene.addItem(self.item1)
        self.scene.addItem(self.item2)
        self.scene.addItem(self._line)
        self.view.setScene(self.scene)
        self.stack.setCurrentWidget(self.view)
        self._sync_items_to_rects()

    def _resolve_pixmap(self, e: dict, prefer_t: float) -> QPixmap:
        from gui_preview import grab_frame, frame_to_pixmap
        if e["kind"] == "seg":                    # 混选：从渲染段取一帧
            off = max(0.0, min(prefer_t - e["t0"], max(e["dur"] - 0.05, 0.0)))
            img = grab_frame(Path(e["payload"]), off, max_width=0)
            return frame_to_pixmap(img) if img is not None else QPixmap()
        if e.get("lazy"):                         # 原片帧
            img = grab_frame(Path(e["src"]), e["t0"], max_width=0)
            return frame_to_pixmap(img) if img is not None else QPixmap()
        return e["payload"]

    def _setup_video(self, e1: dict, e2: dict):
        """原生视频对比：每个 QMediaPlayer 绑定独立 QVideoWidget。"""
        self._win1 = self._video_window(e1, e2)
        self._win2 = self._video_window(e2, e1)

        self._p1 = QMediaPlayer(self)
        self._a1 = QAudioOutput(self)
        self._a1.setMuted(True)                   # 对比默认静音：实测音频时钟间隙会
        self._p1.setAudioOutput(self._a1)         # 拖慢视频步进（0.81→0.93 倍速），🔊 可开
        self._p1.setVideoOutput(self.video_page.w1)
        self._p1.setSource(_local_url(self._win1["src"]))
        self._p1.mediaStatusChanged.connect(
            lambda st, p=self._p1, w=self._win1: self._on_media(st, p, w))

        self._p2 = QMediaPlayer(self)
        self._a2 = QAudioOutput(self)
        self._a2.setMuted(True)                   # 渲染段无音轨，恒静音
        self._p2.setAudioOutput(self._a2)
        self._p2.setVideoOutput(self.video_page.w2)
        self._p2.setSource(_local_url(self._win2["src"]))
        self._p2.mediaStatusChanged.connect(
            lambda st, p=self._p2, w=self._win2: self._on_media(st, p, w))

        self.video_page.layoutChanged.connect(self._update_line_window)

        self.audio_btn.setEnabled(True)
        self.audio_btn.setChecked(False)
        self.audio_btn.setText("🔇")
        self.audio_btn.setVisible(True)
        self.loop_btn.setVisible(True)

        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(SYNC_INTERVAL_MS)
        self._sync_timer.timeout.connect(self._align_playback)
        self._sync_timer.start()
        self.stack.setCurrentWidget(self.video_page)
        self.video_page.relayout()
        self._update_line_window()

    def _video_window(self, e: dict, other: dict) -> dict:
        """播放窗口（映射到文件内毫秒）。

        段文件的内容从源 t0 开始 → 文件时间 [0, dur]，seek=0；
        原片 → 文件时间 [t0, t0+dur]。内容对齐用 seek 作锚点。
        """
        if e["kind"] == "seg":
            dur = max(float(e.get("dur") or 5.0), 0.1)
            return {"src": Path(e["payload"]), "seek": 0, "end": int(dur * 1000)}
        t0 = float(other.get("t0", 0.0))
        dur = float(other.get("dur") or 5.0)
        max_dur = self.preview.info.duration or (t0 + dur)
        end = min(t0 + dur, max_dur)
        return {"src": Path(e["src"]), "seek": int(t0 * 1000), "end": int(end * 1000)}

    def _on_media(self, status, player, win):
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            player.setPosition(win["seek"])
        elif (status == QMediaPlayer.MediaStatus.EndOfMedia
              and getattr(self, "_playing", False)):
            # 窗口循环已在 _align_playback 统一回卷（双方一起）；这里只兜底
            # 处理"窗口短于文件"的孤立 EndOfMedia。窗口回卷后的 400ms 内跳过，
            # 避免同一边界触发两次 seek（解码器重启两次 = 可见停顿加重）。
            now = time.monotonic()
            if now - getattr(self, "_last_rewind", 0.0) < 0.4:
                return
            self._last_rewind = now
            player.setPosition(win["seek"])
            player.play()

    def _align_playback(self):
        """内容对齐：p2 = 自己的 seek 锚点 + p1 已播的内容偏移；
        任一侧到达窗口末端 → 两路一起回卷（窗口循环，单次 seek 对）。"""
        if not (self._p1 and self._p2 and self._playing):
            return
        playing = QMediaPlayer.PlaybackState.PlayingState
        if (self._p1.playbackState() != playing
                or self._p2.playbackState() != playing):
            # 单侧 EndOfMedia 由 _on_media 逐播放器自己回卷。
            # 这里绝不主动 seek：短段每几秒就 EndOfMedia 一次，若任一侧瞬时的
            # 非 Playing 状态就重置两路，会造成每 250ms 一次的 seek 风暴
            # （解码器永远追不上 → 卡顿 + 半解码损坏帧，2026-10-11 实锤）
            return
        if self.loop_enabled:
            if (self._p1.position() >= self._win1["end"]
                    or self._p2.position() >= self._win2["end"]):
                self._last_rewind = time.monotonic()
                self._p1.setPosition(self._win1["seek"])
                self._p2.setPosition(self._win2["seek"])
                return
        else:
            # 不循环：任一侧到窗口末端 → 暂停定住最后一帧，按 ▶ 从头再看
            if (self._p1.position() >= self._win1["end"]
                    or self._p2.position() >= self._win2["end"]):
                self._toggle_play()
                return
        want = self._win2["seek"] + self._p1.position() - self._win1["seek"]
        if abs(self._p2.position() - want) > SYNC_TOLERANCE_MS:
            self._over_budget += 1
            if self._over_budget >= 2:          # 连续 2 tick 超阈值才纠正（滞回）
                self._p2.setPosition(want)
                self._over_budget = 0
        else:
            self._over_budget = 0
    def _toggle_play(self):
        if not self._video_mode:
            return
        if self._playing:
            self._p1.pause()
            self._p2.pause()
            self._playing = False
            self.play_btn.setText("▶ 播放")
        else:
            if (self._video_mode and self._p1.position() >= self._win1["end"] - 2):
                # 已停在窗口末端：按播放 = 从窗口起点重看
                self._p1.setPosition(self._win1["seek"])
                self._p2.setPosition(self._win2["seek"])
            self._p1.play()
            self._p2.play()
            self._playing = True
            self.play_btn.setText("⏸ 暂停")

    def _toggle_audio(self):
        """🔊 开关：有音轨的一侧出声（原片通常有；渲染段 -an 无音轨保持静音）。"""
        on = self.audio_btn.isChecked()
        self.audio_btn.setText("🔊 开" if on else "🔇")
        for p, a in ((self._p1, self._a1), (self._p2, self._a2)):
            if p and a:
                a.setMuted(not (on and p.hasAudio()))

    def _stop_media(self):
        if getattr(self, "_sync_timer", None):
            self._sync_timer.stop()
        for p in (self._p1, self._p2):
            if p:
                p.stop()
                p.deleteLater()
        self._p1 = self._p2 = None
        self._playing = False

    # ---------------------------------------------------------------- 布局/模式
    def _set_mode(self, mode: str):
        self.mode = mode
        self.wipe_btn.setChecked(mode == self.MODE_WIPE)
        self.side_btn.setChecked(mode == self.MODE_SIDE)
        if self._video_mode:
            self.video_page.set_mode(mode)
            self.stack.setCurrentWidget(self.video_page)
        else:
            self._line.setVisible(mode == self.MODE_WIPE)
            if hasattr(self, "item1"):
                self._sync_items_to_rects()
                self.view.fit_content()
        self._update_line_window()
        self._update_hint()

    def _update_line_window(self):
        """原生视频擦除模式的分割线 = 无父窗口的置顶 Tool 小窗（浮于一切视频之上）。
        位置 = 遮罩边缘的 mapToGlobal（全局逻辑坐标）直接 setGeometry——
        无父窗口 → 无任何相对坐标系歧义；应用失活/对话框隐藏时自动隐藏。
        仅在"视频模式 + 擦除 + 视频页可见"时显示。"""
        if getattr(self, "_line_overlay", None) is None:
            return                          # 对话框已关闭（overlay 已销毁）
        vp = self.video_page
        lw = self._line_overlay
        active = bool(QGuiApplication.applicationState() & Qt.ApplicationActive)
        show = (self._video_mode
                and self.mode == self.MODE_WIPE
                and self.stack.currentWidget() is vp and vp.isVisible()
                and self.isVisible() and active)
        if not show:
            lw.hide()
            return
        edge_global = vp.w2.mapToGlobal(
            QPoint(vp.w2.mask().boundingRect().right(), 0))
        lw.setGeometry(edge_global.x() - 2, vp.mapToGlobal(QPoint(0, 0)).y(),
                       5, vp.height())
        lw.show()
        lw.raise_()

    def _sync_items_to_rects(self):
        """图片场景：擦除=两路同位叠加（视频2裁左）；并排=视频2 左、视频1 右。"""
        if not hasattr(self, "item1") or self._W <= 0:
            return
        W, H = self._W, self._H
        r1 = QRectF(0, 0, W, H)
        if self.mode == self.MODE_WIPE:
            r2 = QRectF(0, 0, W, H)
            self._place(self.item1, r1)
            self._place(self.item2, r2)
            self.scene.setSceneRect(r1)
            self._apply_wipe_split()
        else:
            aspect2 = (self._pixmap_aspect(self.item2) or W / H)
            w2 = H * aspect2
            r2 = QRectF(0, 0, w2, H)
            r1 = QRectF(w2 + W * GAP_RATIO, 0, W, H)
            self.item2.clip_rect = None      # 擦除→并排切换：清除擦除期的裁剪残留
            self._place(self.item2, r2)
            self._place(self.item1, r1)
            self.scene.setSceneRect(r2.united(r1))
            self._update_hint()

    @staticmethod
    def _place(item: _ClipPixmapItem, rect: QRectF):
        pm = item.pixmap()
        if pm.width() > 0:
            item.setTransform(QTransform.fromScale(
                rect.width() / pm.width(), rect.height() / pm.height()))
        item.setPos(rect.topLeft())

    @staticmethod
    def _pixmap_aspect(item: _ClipPixmapItem) -> float | None:
        pm = item.pixmap()
        if pm.height() > 0:
            return pm.width() / pm.height()
        return None

    def _apply_wipe_split(self):
        """图片场景：按 _split 更新视频2 裁剪与分割线（高频路径，只动裁剪）。"""
        if not hasattr(self, "item2") or self._W <= 0:
            return
        split_x = self._W * self._split
        self.item2.clip_rect = QRectF(0, 0, split_x, self._H)
        self._line.setLine(split_x, 0, split_x, self._H)
        self._update_hint()

    # ---------------------------------------------------------------- 交互
    def showEvent(self, ev):
        super().showEvent(ev)
        # 构造期的 fitInView 算在未显示的 viewport 上，显示后重适配一次
        QTimer.singleShot(0, self._refit_if_unzoomed)
        self._update_line_window()

    def moveEvent(self, ev):
        super().moveEvent(ev)
        self._update_line_window()      # 顶层分割线小窗跟随对话框移动

    def hideEvent(self, ev):
        super().hideEvent(ev)
        if getattr(self, "_line_overlay", None) is not None:
            self._line_overlay.hide()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._refit_if_unzoomed()
        self._update_line_window()

    def _refit_if_unzoomed(self):
        view = getattr(self, "view", None)
        if view is not None and self._W > 0 and view._zoom <= 1.0 \
                and self.stack.currentWidget() is self.view:
            view.fit_content()

    def _on_hover(self, pos):
        if self.mode != self.MODE_WIPE or self._W <= 0 or pos is None:
            return
        x = max(0.0, min(pos.x(), self._W))
        self._split = x / self._W
        self._apply_wipe_split()
        # 页面遮罩同步（单一事实来源：回到视频模式时两侧分割位置一致）
        if self.video_page.split != self._split:
            self.video_page.set_split(self._split)

    def _on_video_split(self, ratio: float):
        # 单一事实来源：无论谁发起（页面鼠标 / 程序），mask、线窗、提示全部同步到同一值
        self._split = ratio
        if self.video_page.split != ratio:
            self.video_page.set_split(ratio)   # 页面遮罩同步（relayout 有 memo，未变时零开销）
        self._update_line_window()
        self._update_hint()

    def _toggle_fs(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_F11:
            self._toggle_fs()
            return
        if ev.key() == Qt.Key_Escape and self.isFullScreen():
            self.showNormal()
            return
        super().keyPressEvent(ev)

    def _update_hint(self):
        n1 = self.combo1.currentText()
        n2 = self.combo2.currentText()
        if self._video_mode:
            pos = f"分割 {self._split * 100:.0f}%" if self.mode == self.MODE_WIPE else ""
            self.hint.setText(
                f"左＝{n2} · 右＝{n1}　|　原生全帧率播放 · 移动鼠标擦除 · "
                f"暂停后扫动可静帧对比 · 双击/F11 全屏　|　{pos}")
        elif self.mode == self.MODE_WIPE:
            pos = f"分割 {self._split * 100:.0f}%"
            self.hint.setText(
                f"左＝{n2} · 右＝{n1}　|　移动鼠标擦除 · 滚轮放大细节（分割线锚定画面）· "
                f"左键拖动平移 · 双击/F11 全屏　|　{pos}")
        else:
            self.hint.setText(
                f"左＝{n2} · 右＝{n1}　|　滚轮同步放大细节 · 左键拖动平移 · 双击/F11 全屏")

    def done(self, r):
        self._stop_media()
        if self._line_overlay is not None:
            self._line_overlay.hide()
            self._line_overlay.deleteLater()
            self._line_overlay = None
        try:
            QGuiApplication.instance().applicationStateChanged.disconnect(
                self._update_line_window)
        except (RuntimeError, TypeError):
            pass
        super().done(r)


def _local_url(path: Path):
    from PySide6.QtCore import QUrl
    return QUrl.fromLocalFile(str(path))
