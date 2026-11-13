"""存档压缩页：SVT-AV1 分段直编（断点续跑/暂停）+ 编码后三项验证。

与旧增强界面完全分离：画质只有 CRF 一个旋钮，编码固定 SVT-AV1 preset4
（定版存档基线），容器可选（VIDEO_CONTAINERS 全清单），位深自动跟随源片，
输出到源文件所在目录。分段编排（断点续跑/暂停/合并）在 crispframe.archive_job；
逐核相关性调度在 crispframe.affinity（任务管理器式对话框，编码中即时生效）。

引擎复用 crispframe.pipeline.run_passthrough（单进程直编，色彩/时基元数据
passthrough）与 crispframe.verify（体积/元数据/全量解码查错）。
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QListWidget, QPushButton, QProgressBar, QSlider,
                               QTextEdit, QVBoxLayout, QWidget)

from crispframe import affinity
from crispframe.archive_job import parts_dir, run_segmented
from crispframe.codec import (VIDEO_CONTAINERS, available_containers,
                              archive_pix_fmt, CodecSettings)
from crispframe.ffutils import SUBPROC_FLAGS, VideoInfo, probe
from crispframe.pipeline import PassthroughCanceled
from crispframe.verify import human_size, probe_stream_meta, verify_output

_CRF_NOTES = {25: "稳", 27: "稳", 30: "甜点", 32: "省"}


class ProbeWorker(QThread):
    """轻量探测：只取 VideoInfo。"""
    probed = Signal(str, object, str)   # path, VideoInfo | None, err

    def __init__(self, paths: list[Path]):
        super().__init__()
        self.paths = paths

    def run(self):
        for p in self.paths:
            try:
                self.probed.emit(str(p), probe(p), "")
            except Exception as e:  # noqa: BLE001
                self.probed.emit(str(p), None, f"{p.name}: {e}")


class ArchiveWorker(QThread):
    taskProgress = Signal(str, int, int)   # 文件名, done, total（编码/验证统一为秒）
    fpsTick = Signal(float)                # 整片实时帧率（2s 节流）
    log = Signal(str)
    fileDone = Signal(str, bool, str)      # 文件名, 成功, 消息
    allDone = Signal(bool, str)

    def __init__(self, items: list[tuple[Path, VideoInfo]], crf: int,
                 container: str = "matroska", affinity_ids: list[int] | None = None):
        super().__init__()
        self.items = items
        self.crf = crf
        self.container = container
        self.affinity_ids = affinity_ids   # None=不干预；list=逐核勾选集合
        self._cancel = False               # 放弃：终止当前段并删分片
        self._pause = False                # 暂停：当前段跑完（或被立即终止）后保留分片
        self._proc = None                  # 当前段 ffmpeg 进程（相关性切换用）

    def cancel(self):
        self._cancel = True

    def pause(self):
        self._pause = True
        # 立即终止当前段 ffmpeg：run_passthrough 收到 rc≠0+零 stderr →
        # SilentTermination → archive_job 的 pause 分支（残段丢、完成段留）
        if self._proc is not None:
            try:
                self._proc.terminate()
            except OSError:
                pass

    def set_affinity(self, ids: list[int]) -> None:
        """编码中即时切换相关性（GUI 线程调用；Win32 调用线程安全）。"""
        self.affinity_ids = ids
        if self._proc is not None and ids:
            if affinity.apply(self._proc, ids):
                self.log.emit(f"  相关性已切换：CPU {', '.join(map(str, sorted(ids)))}")
            else:
                self.log.emit("  ⚠ 相关性切换失败（本机不支持）")

    def run(self):
        ok_all = True
        for src, info in self.items:
            if self._cancel:
                self.fileDone.emit(src.name, False, "已放弃")
                ok_all = False
                continue
            try:
                if not self._process_one(src, info):
                    ok_all = False
            except Exception as e:  # noqa: BLE001
                self.log.emit(f"[错误] {src.name}: {e}")
                self.fileDone.emit(src.name, False, str(e))
                ok_all = False
        if self._cancel:
            self.allDone.emit(False, "已放弃")
        else:
            self.allDone.emit(ok_all, "全部完成" if ok_all else "有文件未完成存档")

    def _on_proc(self, proc):
        self._proc = proc
        if self.affinity_ids:
            affinity.apply(proc, self.affinity_ids)

    def _process_one(self, src: Path, info: VideoInfo) -> bool:
        """单文件：预检 → 分段直编（断点续跑）→ 三项验证。失败/放弃清分片。"""
        cs = CodecSettings()              # 定版基线：svtav1 / preset4 / matroska / 10bit
        cs.crf = self.crf
        cs.container = self.container
        ext = VIDEO_CONTAINERS[self.container]["ext"]
        out = src.with_name(f"{src.stem}.cf.archive-crf{self.crf}{ext}")
        if out.exists():
            # 已验证的存档成品不做静默覆盖；失败产物在失败点即删，拦到的只剩真成品
            self.log.emit(f"[跳过] {src.name}: 输出已存在（不覆盖，请先处理已有文件）")
            self.fileDone.emit(src.name, False, "输出已存在")
            return False

        # 像素格式映射（严格元数据一致只保证表内格式）：4:2:2/4:4:4 等
        # 不支持源在此预检拒绝，不空转编码
        pix = archive_pix_fmt(probe_stream_meta(src)["pix_fmt"])
        if pix is None:
            self.log.emit(f"[跳过] {src.name}: 源像素格式超出严格一致支持范围"
                          f"（仅 yuv420p / yuv420p10le 可保持输出元数据一致），"
                          f"请用 ffmpeg 直编或先转 4:2:0")
            self.fileDone.emit(src.name, False, "源像素格式不支持")
            return False
        cs.bit_depth, _ = pix
        free = shutil.disk_usage(out.parent).free
        if free < src.stat().st_size / 4:
            self.log.emit(f"⚠ {src.name}: 输出盘剩余 {human_size(free)}，"
                          f"建议至少留源片体积 25% 的余量")

        cont_label = VIDEO_CONTAINERS[self.container]["label"]
        self.log.emit(f"▶ {src.name}: {info.width}x{info.height}@{info.fps:.3f} "
                      f"SVT-AV1 crf{cs.crf}/preset{cs.preset} → {cont_label.split('（')[0]}"
                      f"（分段直编，每段 {run_seg_secs()}s，全流保留，色彩/时基跟随源），"
                      f"完成后自动验证")
        t0 = time.perf_counter()
        try:
            result = run_segmented(
                info, src, out, cs,
                cancel_check=lambda: self._cancel,
                pause_check=lambda: self._pause,
                progress_cb=lambda d, t: self.taskProgress.emit(src.name, int(d), int(t)),
                fps_cb=lambda fps: self.fpsTick.emit(fps),
                log_cb=self.log.emit,
                on_process=self._on_proc)
        except PassthroughCanceled:
            self.log.emit(f"  {src.name}: 已放弃，分片与残件已删除")
            self.fileDone.emit(src.name, False, "已放弃")
            return False
        self._proc = None
        if result == "pause":
            self.log.emit(f"  {src.name}: 已暂停——分片保留，可随时关闭软件；"
                          f"重新加入该源片点「继续存档」即可续跑")
            self.fileDone.emit(src.name, False, "已暂停（可续跑）")
            return False
        if result != "ok":
            self.log.emit("  存档失败：分片与残件已删除（源片保留）")
            self.fileDone.emit(src.name, False, "存档失败")
            return False
        self.log.emit(f"✓ {out.name}  编码 {time.perf_counter() - t0:.0f}s / "
                      f"{human_size(out.stat().st_size)}")

        verdict = self._verify(src, out)
        pd = parts_dir(src, cs)
        if verdict == "pass":
            shutil.rmtree(pd, ignore_errors=True)   # 验证通过才清分片
            self.fileDone.emit(src.name, True, out.name)
            return True
        out.unlink(missing_ok=True)       # 未过验证 = 不可归档，删除待重跑
        shutil.rmtree(pd, ignore_errors=True)
        self.log.emit("  未通过验证的成品已删除（源片保留），可重新开始存档")
        self.fileDone.emit(src.name, False,
                           "已放弃" if verdict == "cancel" else "验证未通过")
        return False

    def _verify(self, src: Path, out: Path) -> str:
        """返回 'pass' / 'fail' / 'cancel'。"""
        self.log.emit("  验证：体积 / 元数据 / 全量解码查错（约数分钟到半小时，可放弃）...")
        vr = verify_output(
            out, src, strict_metadata=True,
            progress_callback=lambda dn, t: self.taskProgress.emit(src.name, dn, t),
            cancel_check=lambda: self._cancel)
        for line in vr.lines + vr.failures:
            self.log.emit("  " + line)
        if vr.canceled:
            return "cancel"
        if vr.ok:
            self.log.emit("  结论: ✓ 验证全部通过，可归档")
            return "pass"
        self.log.emit("  ⚠ 验证未通过，不要删除源片！")
        return "fail"


def run_seg_secs() -> int:
    from crispframe.archive_job import SEG_SECS
    return SEG_SECS


VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".ts", ".m2ts", ".flv"}


class AffinityDialog(QDialog):
    """任务管理器式逐核相关性设置：每个逻辑处理器独立勾选。"""

    def __init__(self, parent, current_ids: list[int] | None):
        super().__init__(parent)
        self.setWindowTitle("设置 CPU 相关性（编码中即时生效）")
        self.selected_ids: list[int] | None = None
        try:
            p_cores, e_cores = affinity.topology()
        except Exception as e:  # noqa: BLE001
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "设置相关性", f"本机拓扑读取失败：{e}")
            self._all: list[int] = []
            return
        p_set, e_set = set(p_cores), set(e_cores)
        self._all = sorted(p_set | e_set)
        self._boxes: dict[int, QCheckBox] = {}
        v = QVBoxLayout(self)

        tip = QLabel("勾选允许参与编码的逻辑处理器；编码进行中确认后立即生效。"
                     "全选 = 不干预（系统默认调度）。")
        tip.setWordWrap(True)
        v.addWidget(tip)

        rows = QHBoxLayout()
        for group_name, ids in (("大核（P）", sorted(p_set)),
                                ("小核（E）", sorted(e_set))):
            gb = QGroupBox(group_name)
            gv = QGridLayout(gb)
            for r, i in enumerate(ids):
                cb = QCheckBox(f"CPU {i}")
                cb.setChecked(current_ids is None or i in current_ids)
                cb.stateChanged.connect(self._sync_ok)
                self._boxes[i] = cb
                gv.addWidget(cb, r // 2, r % 2)
            rows.addWidget(gb)
        v.addLayout(rows)

        pb = QHBoxLayout()
        for text, ids in (("全选", self._all),
                          ("全大核", p_cores),
                          ("全小核", e_cores),
                          ("反选", None)):
            b = QPushButton(text)
            b.clicked.connect(lambda _, ids=ids: self._preset(ids))
            pb.addWidget(b)
        pb.addStretch()
        v.addLayout(pb)

        bb = QHBoxLayout()
        bb.addStretch()
        ok = QPushButton("确定")
        ok.setObjectName("startBtn")
        ok.clicked.connect(self.accept)
        bb.addWidget(ok)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        bb.addWidget(cancel)
        v.addLayout(bb)
        self._sync_ok()

    def _preset(self, ids: list[int] | None):
        if ids is None:                       # 反选
            for i, cb in self._boxes.items():
                cb.setChecked(i not in self.selected_now())
            return
        for i, cb in self._boxes.items():
            cb.setChecked(i in ids)
        self._sync_ok()

    def selected_now(self) -> list[int]:
        return [i for i, cb in self._boxes.items() if cb.isChecked()]

    def _sync_ok(self):
        for b in self.findChildren(QPushButton):
            if b.objectName() == "startBtn":
                b.setEnabled(bool(self.selected_now()))

    def accept(self):
        sel = self.selected_now()
        if not sel:
            return
        self.selected_ids = sel
        super().accept()


class ArchivePage(QWidget):
    """存档压缩页：拖入视频 + CRF 旋钮 + 暂停/放弃 + 逐核相关性 + 进度与验证日志。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[tuple[Path, VideoInfo]] = []
        self._probing: set[Path] = set()
        self._probe_workers: list[ProbeWorker] = []
        self._gen = 0                  # 清空即换代：在途探测的回填按代丢弃
        self.worker: ArchiveWorker | None = None
        self._affinity_ids: list[int] | None = None   # None=不干预
        self.setAcceptDrops(True)

        v = QVBoxLayout(self)
        v.setContentsMargins(48, 32, 48, 24)
        title = QLabel("存档压缩")
        title.setStyleSheet("font-size:22px;font-weight:bold;")
        v.addWidget(title)
        sub = QLabel("SVT-AV1 恒定画质分段直编 · 暂停后可关闭软件、重新打开自动续跑 · "
                     "保留全部音轨/字幕/章节 · 位深/色彩/时基跟随源片 · 容器可选 · "
                     "编码后自动验证完整性")
        sub.setStyleSheet("color:#9aa0a8;")
        sub.setWordWrap(True)
        v.addWidget(sub)

        self.drop = QPushButton("＋ 拖入视频文件，或点击选择")
        self.drop.setObjectName("dropZone")
        self.drop.setFixedHeight(120)
        self.drop.clicked.connect(self._pick)
        v.addWidget(self.drop)

        self.file_list = QListWidget()
        self.file_list.setVisible(False)
        v.addWidget(self.file_list, 1)

        crow = QHBoxLayout()
        crow.addWidget(QLabel("输出容器"))
        self.container_box = QComboBox()
        self._containers = available_containers("svtav1")
        for key, spec in self._containers.items():
            self.container_box.addItem(f"{spec['label']} — {spec['note']}", key)
        self.container_box.setCurrentIndex(0)     # 默认 MKV
        self.container_box.currentIndexChanged.connect(
            lambda _: (self.tip.setText(self._tip_text()),
                       self._sync_crf(self.crf_slider.value())))
        crow.addWidget(self.container_box, 1)
        self.aff_btn = QPushButton("设置相关性…")
        self.aff_btn.setToolTip("逐逻辑处理器勾选（大核/小核分组），编码中确认即时生效；"
                                "默认不干预")
        self.aff_btn.clicked.connect(self._open_affinity)
        crow.addWidget(self.aff_btn)
        self.shutdown_cb = QCheckBox("完成后关机")
        self.shutdown_cb.setToolTip("全部任务成功完成后 60 秒自动关机（挂机过夜用；"
                                    "期间可点「取消关机」）")
        crow.addWidget(self.shutdown_cb)
        v.addLayout(crow)

        row = QHBoxLayout()
        row.addWidget(QLabel("画质 CRF"))
        self.crf_slider = QSlider(Qt.Horizontal)
        self.crf_slider.setRange(22, 35)
        self.crf_slider.setValue(30)
        self.crf_slider.valueChanged.connect(self._sync_crf)
        row.addWidget(self.crf_slider, 1)
        self.crf_lab = QLabel("CRF 30")
        self.crf_lab.setMinimumWidth(64)
        row.addWidget(self.crf_lab)
        v.addLayout(row)
        self.crf_notes = QLabel(self._notes_text())
        self.crf_notes.setStyleSheet("color:#8a8f98;font-size:11px;")
        v.addWidget(self.crf_notes)
        self.tip = QLabel(self._tip_text())
        self.tip.setStyleSheet("color:#8a8f98;font-size:11px;")
        self.tip.setWordWrap(True)
        v.addWidget(self.tip)

        bar = QHBoxLayout()
        clear = QPushButton("清空列表")
        clear.clicked.connect(self._clear)
        bar.addWidget(clear)
        rm = QPushButton("移除选中")
        rm.clicked.connect(self._remove_selected)
        bar.addWidget(rm)
        self._list_btns = (clear, rm)
        bar.addStretch()
        self.aff_live_btn = QPushButton("相关性…")
        self.aff_live_btn.setVisible(False)
        self.aff_live_btn.clicked.connect(self._open_affinity)
        bar.addWidget(self.aff_live_btn)
        self.abandon_btn = QPushButton("放弃")
        self.abandon_btn.setVisible(False)
        self.abandon_btn.setToolTip("终止并存档：删除分片与残件，从头再来")
        self.abandon_btn.clicked.connect(self._abandon)
        bar.addWidget(self.abandon_btn)
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setVisible(False)
        self.pause_btn.setToolTip("立即暂停：当前段作废、已完成段保留，"
                                  "可随时关闭软件；重新加入源片即可续跑")
        self.pause_btn.clicked.connect(self._pause)
        bar.addWidget(self.pause_btn)
        self.start_btn = QPushButton("开始存档")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start)
        bar.addWidget(self.start_btn)
        self.cancel_sd_btn = QPushButton("取消关机")
        self.cancel_sd_btn.setVisible(False)
        self.cancel_sd_btn.setObjectName("stopBtn")
        self.cancel_sd_btn.clicked.connect(self._cancel_shutdown)
        bar.addWidget(self.cancel_sd_btn)
        v.addLayout(bar)
        self._sd_timer: QTimer | None = None

        self.bar = QProgressBar()
        v.addWidget(self.bar)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(180)
        v.addWidget(self.log_box)

    # —— 文件管理 ——
    def _pick(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择视频", "", "视频 (*.mp4 *.mkv *.avi *.webm *.mov *.ts *.m2ts *.flv)")
        self.set_files([Path(f) for f in files])

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        e.acceptProposedAction()

    def dropEvent(self, e):
        files = [Path(u.toLocalFile()) for u in e.mimeData().urls()]
        self.set_files([f for f in files if f.suffix.lower() in VIDEO_EXTS])

    def set_files(self, paths: list[Path]):
        """拖入/选择入口：入队探测，完成一个亮一个。"""
        if paths:
            self.file_list.setVisible(True)
        gen = self._gen
        for p in paths:
            if (p in self._probing or any(s == p for s, _ in self._items)
                    or self.file_list.findItems(str(p), Qt.MatchExactly)):
                continue
            self._probing.add(p)
            self.file_list.addItem(str(p))
            w = ProbeWorker([p])
            self._probe_workers.append(w)
            w.probed.connect(lambda ps, info, err, g=gen: self._on_probed(ps, info, err, g))
            w.finished.connect(
                lambda w=w: self._probe_workers.remove(w) if w in self._probe_workers else None)
            w.start()

    def _on_probed(self, path_str: str, info, err: str, gen: int = -1):
        p = Path(path_str)
        self._probing.discard(p)
        if gen != self._gen:
            return                     # 列表已被清空（换代）：丢弃在途回填
        if err:
            self._take_item(p)
            self.log_box.append(f"[跳过] {err}")
            return
        self._items.append((p, info))
        self._sync_start_text()
        # 断点探测：同 CRF/容器设置的分片目录存在 → 提示续跑
        cs = CodecSettings()
        cs.crf = self.crf_slider.value()
        cs.container = self._container_key()
        if parts_dir(p, cs).exists():
            self.log_box.append(f"[续跑] {p.name}: 发现未完成的存档分片，"
                                f"点「继续存档」将从断点继续（CRF/容器需与上次一致）")
        self._sync_buttons()

    def _take_item(self, p: Path):
        for it in self.file_list.findItems(str(p), Qt.MatchExactly):
            self.file_list.takeItem(self.file_list.row(it))

    def _remove_selected(self):
        for it in self.file_list.selectedItems():
            p = Path(it.text())
            if p in self._probing:
                continue    # 探测中的无法取消，等回填后再移除
            self._items = [x for x in self._items if x[0] != p]
            self.file_list.takeItem(self.file_list.row(it))
        self._sync_buttons()

    def _clear(self):
        self._gen += 1                 # 在途探测的回填按代丢弃，防"隐形"文件
        self._items.clear()
        self._probing.clear()
        self.file_list.clear()
        self.file_list.setVisible(False)
        self._sync_buttons()
        self.drop.setVisible(True)

    def _sync_buttons(self):
        busy = self.worker is not None
        self.start_btn.setEnabled(bool(self._items) and not busy)
        for b in self._list_btns:
            b.setEnabled(bool(self._items) and not busy)
        self.pause_btn.setVisible(busy)
        self.pause_btn.setEnabled(busy)
        self.aff_live_btn.setVisible(busy)
        self.abandon_btn.setVisible(busy)
        self.abandon_btn.setEnabled(busy)

    # —— 设置 ——
    def _notes_text(self) -> str:
        return "  ".join(f"{v}={n}" for v, n in sorted(_CRF_NOTES.items())) + \
               "（越小质量越高、体积越大；默认 30 是挂机存档甜点）"

    def _tip_text(self) -> str:
        key = self._container_key()
        ext = self._containers.get(key, {"ext": ".mkv"})["ext"]
        return (f"输出到源文件所在目录（*.cf.archive-crfN{ext}）。长片的全量解码验证"
                f"可能需要数十分钟；验证通过前请勿删除源片。")

    def _container_key(self) -> str:
        return self.container_box.currentData() or "matroska"

    def _sync_crf(self, v: int):
        self.crf_lab.setText(f"CRF {v}")

    # —— 相关性 ——
    def _open_affinity(self):
        dlg = AffinityDialog(self, self._affinity_ids)
        if dlg.exec() and dlg.selected_ids:
            self._affinity_ids = dlg.selected_ids
            self.log_box.append(f"相关性：CPU {', '.join(map(str, sorted(self._affinity_ids)))}"
                                f"（{'已即时生效' if self.worker else '下一段起生效'}）")
            if self.worker:
                self.worker.set_affinity(self._affinity_ids)

    # —— 运行 ——
    def _start(self):
        if self.worker or not self._items:
            return
        self.worker = ArchiveWorker(list(self._items), self.crf_slider.value(),
                                    self._container_key(),
                                    affinity_ids=self._affinity_ids)
        self.worker.taskProgress.connect(
            lambda name, dn, t: (self.bar.setMaximum(t),
                                 self.bar.setValue(dn)) if t else None)
        self.worker.fpsTick.connect(
            lambda fps: self.setWindowTitle(
                f"存档压缩 — 实时 {fps:.1f} 帧/s"))
        self.worker.log.connect(self.log_box.append)
        self.worker.fileDone.connect(self._on_file_done)
        self.worker.allDone.connect(self._on_all_done)
        self._cancel_shutdown(silent=True)
        self._sync_buttons()
        self.bar.setValue(0)
        self.worker.start()

    def _on_file_done(self, name: str, ok: bool, msg: str):
        if ok:
            # 完成的文件移出列表（产物已落盘），未完成的留在列表便于重跑
            p = Path(name)
            self._items = [x for x in self._items if x[0] != p]
            self._take_item(p)
        elif "暂停" in msg or "放弃" in msg:
            self._sync_start_text()

    def _on_all_done(self, ok: bool, msg: str):
        self.log_box.append(f"— {msg} —")
        self.worker = None
        self._sync_buttons()
        self.bar.setValue(0)
        self.setWindowTitle("存档压缩")
        self.start_btn.setText("开始存档")
        if ok and self.shutdown_cb.isChecked():
            self._begin_shutdown_countdown()

    # —— 自动关机 ——
    def _begin_shutdown_countdown(self, seconds: int = 60):
        self._shutdown_left = seconds
        self.log_box.append(f"全部完成：{seconds} 秒后自动关机"
                            f"（点「取消关机」可中止）")
        self.cancel_sd_btn.setVisible(True)
        self._sync_shutdown_btn()
        self._sd_timer = QTimer(self)
        self._sd_timer.setInterval(1000)
        self._sd_timer.timeout.connect(self._shutdown_tick)
        self._sd_timer.start()

    def _sync_shutdown_btn(self):
        self.cancel_sd_btn.setText(f"取消关机（{self._shutdown_left}s）")

    def _shutdown_tick(self):
        self._shutdown_left -= 1
        if self._shutdown_left <= 0:
            self._cancel_shutdown(silent=True)
            self.log_box.append("正在关机…")
            subprocess.run(["shutdown", "/s", "/t", "0"],
                           capture_output=True, **SUBPROC_FLAGS)
            return
        self._sync_shutdown_btn()

    def _cancel_shutdown(self, silent: bool = False):
        if self._sd_timer is not None:
            self._sd_timer.stop()
            self._sd_timer = None
        self.cancel_sd_btn.setVisible(False)
        if not silent:
            self.log_box.append("已取消自动关机")

    def _sync_start_text(self):
        """有任一源片带分片 → 按钮变「继续存档」。"""
        cs = CodecSettings()
        cs.crf = self.crf_slider.value()
        cs.container = self._container_key()
        has_parts = any(parts_dir(p, cs).exists() for p, _ in self._items)
        self.start_btn.setText("继续存档" if has_parts else "开始存档")

    def _pause(self):
        if self.worker:
            self.worker.pause()
            self.pause_btn.setEnabled(False)
            self.log_box.append("暂停中：当前段已终止作废（已完成段保留）…")

    def _abandon(self):
        if self.worker:
            self.worker.cancel()
            self.abandon_btn.setEnabled(False)
            self.log_box.append("放弃中：当前段将被终止，分片与残件一并删除…")
        else:
            cs = CodecSettings()
            cs.crf = self.crf_slider.value()
            cs.container = self._container_key()
            for p, _ in self._items:
                d = parts_dir(p, cs)
                if d.exists():
                    for f in d.iterdir():
                        f.unlink(missing_ok=True)
                    d.rmdir()
                    self.log_box.append(f"已清除 {p.name} 的未完成分片")
            self._sync_start_text()
