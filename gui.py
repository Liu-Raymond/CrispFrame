"""CrispFrame GUI —— Topaz 风格深色界面。

布局：左侧任务队列（拖拽添加）· 右侧参数面板（档位+专家折叠）· 底部进度与日志。
管线在 QThread 中运行，进度经信号槽回 UI；--extreme 档走子进程轮询。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QProgressBar, QPushButton, QSlider, QSpinBox, QSplitter, QTextEdit,
    QVBoxLayout, QWidget,
)

from crispframe import registry
from crispframe.ffutils import probe
from crispframe.pipeline import EncodeConfig, run

APP_TITLE = "CrispFrame — 让每一帧都锐利"

DARK_QSS = """
QWidget { background:#1e1f22; color:#d7d8db; font-size:13px; }
QListWidget, QTextEdit { background:#17181a; border:1px solid #2b2d31; border-radius:6px; }
QPushButton { background:#2f3136; border:1px solid #3f4147; border-radius:6px;
              padding:7px 14px; }
QPushButton:hover { background:#3a3c42; }
QPushButton:disabled { color:#6a6c72; }
QPushButton#startBtn { background:#3b82f6; border:none; font-weight:bold; padding:10px; }
QPushButton#startBtn:hover { background:#2f6fe0; }
QPushButton#stopBtn  { background:#8b3a3a; border:none; padding:10px; }
QGroupBox { border:1px solid #2b2d31; border-radius:8px; margin-top:12px;
            padding-top:8px; font-weight:bold; }
QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 4px; color:#9aa0a8; }
QProgressBar { background:#17181a; border:1px solid #2b2d31; border-radius:6px;
               text-align:center; height:20px; }
QProgressBar::chunk { background:#3b82f6; border-radius:5px; }
QComboBox, QSpinBox, QDoubleSpinBox { background:#17181a; border:1px solid #2b2d31;
                                      border-radius:4px; padding:3px 6px; }
QSlider::groove:horizontal { height:4px; background:#2b2d31; border-radius:2px; }
QSlider::handle:horizontal { width:14px; margin:-6px 0; background:#3b82f6; border-radius:7px; }
"""


class Worker(QThread):
    progress = Signal(int, int)
    log = Signal(str)
    done = Signal(bool, str)  # ok, message

    def __init__(self, jobs: list[dict], settings: dict):
        super().__init__()
        self.jobs = jobs
        self.settings = settings
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        ok_all = True
        for job in self.jobs:
            if self._cancel:
                break
            try:
                self._run_one(job)
            except Exception as e:  # noqa: BLE001
                self.log.emit(f"[错误] {job['path'].name}: {e}")
                ok_all = False
        self.done.emit(ok_all and not self._cancel,
                       "已取消" if self._cancel else ("全部完成" if ok_all else "有任务失败"))

    def _run_one(self, job: dict):
        src: Path = job["path"]
        s = self.settings
        self.log.emit(f"▶ {src.name}")
        info = probe(src)
        scale = 2 if s["content"] == "anime" else 2  # 快速档均为 2x 系
        out_w, out_h = s["target_wh"]
        tags = [s["model"], "tm" if s["tone_match"] else "", f"s{int(s['sharpen']*10)}"]
        out_path = src.with_name(f"{src.stem}.cf.{'+'.join(x for x in tags if x)}.mp4")

        stages = []
        if s["quality"] == "fast":
            from crispframe.sr import SuperRes
            from crispframe.gpu_post import GpuPost
            stages.append(SuperRes(registry.resolve(s["model"]), use_amp=not s["fp32"],
                                   batch_size=s["batch"], compile_model=False, use_trt=not s["no_trt"]))
            stages.append(GpuPost(out_w, out_h, sharpen=s["sharpen"]))
        else:  # extreme：SeedVR2 走子进程，见 extreme.py；此处单独处理
            from crispframe.extreme import run_seedvr2
            result = run_seedvr2(src, out_path.parent, resolution=s["extreme_res"])
            result.replace(out_path)
            self.progress.emit(1, 1)
            self.log.emit(f"✓ {out_path.name}（极致档）")
            return

        t0 = time.perf_counter()
        run(info, out_path, stages=stages,
            encode=EncodeConfig(encoder=s["encoder"], crf=s["crf"], preset=s["preset"]),
            batch_size=s["batch"],
            progress_callback=lambda d, t: self.progress.emit(d, t),
            cancel_check=lambda: self._cancel)
        dt = time.perf_counter() - t0
        size_mb = out_path.stat().st_size / 1e6
        self.log.emit(f"✓ {out_path.name}  {info.fps:.0f}fps→{out_h}p  "
                      f"{dt:.0f}s / {size_mb:.1f}MB")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1060, 640)
        self.worker: Worker | None = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        split = QSplitter(Qt.Horizontal)
        layout.addWidget(split, 1)

        # 左：任务队列
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("任务队列（拖入视频文件）"))
        self.file_list = QListWidget()
        self.file_list.setAcceptDrops(True)
        self.file_list.setDragDropMode(QListWidget.DropOnly)
        self.file_list.setAcceptDrops(True)
        self.file_list.dragEnterEvent = self._drag_enter
        self.file_list.dragMoveEvent = lambda e: e.acceptProposedAction()
        self.file_list.dropEvent = self._drop
        lv.addWidget(self.file_list)
        btns = QHBoxLayout()
        add = QPushButton("添加文件")
        add.clicked.connect(self._add_files)
        rm = QPushButton("移除选中")
        rm.clicked.connect(lambda: [self.file_list.takeItem(self.file_list.row(i)) for i in self.file_list.selectedItems()])
        btns.addWidget(add)
        btns.addWidget(rm)
        btns.addStretch()
        lv.addLayout(btns)
        split.addWidget(left)

        # 右：参数
        right = QWidget()
        rv = QVBoxLayout(right)

        g1 = QGroupBox("画质档位")
        g1v = QVBoxLayout(g1)
        self.quality = QComboBox()
        self.quality.addItems(["快速（推荐，~12 帧/s @4K）", "极致 SeedVR2（质量最高，~0.2 帧/s）"])
        g1v.addWidget(self.quality)
        self.quality.currentIndexChanged.connect(
            lambda i: self.extreme_box.setEnabled(i == 1))
        self.extreme_box = QGroupBox("极致档选项")
        ev = QVBoxLayout(self.extreme_box)
        ev.addWidget(QLabel("输出高度"))
        self.extreme_res = QSpinBox()
        self.extreme_res.setRange(720, 4320)
        self.extreme_res.setValue(2160)
        ev.addWidget(self.extreme_res)
        self.extreme_box.setEnabled(False)
        g1v.addWidget(self.extreme_box)
        rv.addWidget(g1)

        g2 = QGroupBox("内容与模型（快速档）")
        g2v = QVBoxLayout(g2)
        self.content = QComboBox()
        self.content.addItems(["真人（2xNomosUni）", "动漫（AnimeJaNai）"])
        g2v.addWidget(self.content)
        g2v.addWidget(QLabel("锐化补偿（0-2，默认 0.6）"))
        self.sharpen = QSlider(Qt.Horizontal)
        self.sharpen.setRange(0, 20)
        self.sharpen.setValue(6)
        self.sharpen_lab = QLabel("0.6")
        self.sharpen.valueChanged.connect(
            lambda v: self.sharpen_lab.setText(f"{v/10:.1f}"))
        row = QHBoxLayout()
        row.addWidget(self.sharpen)
        row.addWidget(self.sharpen_lab)
        g2v.addLayout(row)
        rv.addWidget(g2)

        g3 = QGroupBox("输出")
        g3v = QVBoxLayout(g3)
        g3v.addWidget(QLabel("目标分辨率"))
        self.target = QComboBox()
        self.target.addItems(["3840x2160（4K）", "2560x1440（2K）", "1920x1080（1080p）"])
        g3v.addWidget(self.target)
        g3v.addWidget(QLabel("编码器"))
        self.encoder = QComboBox()
        self.encoder.addItems(["av1_nvenc", "hevc_nvenc"])
        g3v.addWidget(self.encoder)
        self.tone = QCheckBox("色调对齐源片")
        self.tone.setChecked(True)
        g3v.addWidget(self.tone)
        rv.addWidget(g3)
        rv.addStretch()
        split.addWidget(right)
        split.setSizes([520, 480])

        # 底部：进度与日志
        self.bar = QProgressBar()
        layout.addWidget(self.bar)
        ctrl = QHBoxLayout()
        self.start_btn = QPushButton("开始处理")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(lambda: self.worker and self.worker.cancel())
        ctrl.addWidget(self.start_btn)
        ctrl.addWidget(self.stop_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(140)
        layout.addWidget(self.log_box)

    # —— 拖拽与文件 ——
    def _drag_enter(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def _drop(self, e):
        exts = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".ts"}
        for url in e.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.suffix.lower() in exts:
                self.file_list.addItem(QListWidgetItem(str(p)))

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择视频", "", "视频 (*.mp4 *.mkv *.avi *.webm *.mov *.ts)")
        for f in files:
            self.file_list.addItem(QListWidgetItem(f))

    def _jobs(self) -> list[dict]:
        return [{"path": Path(self.file_list.item(i).text())}
                for i in range(self.file_list.count())]

    def _settings(self) -> dict:
        tw, _, th = self.target.currentText().partition("x")
        th = th.split("（")[0]
        return {
            "quality": "fast" if self.quality.currentIndex() == 0 else "extreme",
            "content": "real" if self.content.currentIndex() == 0 else "anime",
            "model": "anime" if self.content.currentIndex() == 1 else "nomosuni",
            "sharpen": self.sharpen.value() / 10,
            "tone_match": self.tone.isChecked(),
            "target_wh": (int(tw), int(th)),
            "encoder": self.encoder.currentText(),
            "crf": 28,
            "preset": "p5",
            "batch": 0,
            "fp32": False,
            "no_trt": False,
            "extreme_res": self.extreme_res.value(),
        }

    # —— 启停 ——
    def _start(self):
        jobs = self._jobs()
        if not jobs or self.worker:
            return
        self.worker = Worker(jobs, self._settings())
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._on_log)
        self.worker.done.connect(self._on_done)
        self.bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker.start()

    def _on_progress(self, done, total):
        if total:
            self.bar.setMaximum(total)
            self.bar.setValue(done)

    def _on_log(self, msg):
        self.log_box.append(msg)

    def _on_done(self, ok, msg):
        self.log_box.append(f"— {msg} —")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker = None


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
