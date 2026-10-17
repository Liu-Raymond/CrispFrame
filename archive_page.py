"""存档压缩页：SVT-AV1 单进程直编 + 编码后三项验证的独立简洁界面。

与任务页（AI 增强）完全分离——没有模块链、没有输出设置抽屉：
画质只有 CRF 一个旋钮，编码固定 SVT-AV1 preset4（定版存档基线），容器可选
（VIDEO_CONTAINERS 全清单），位深自动跟随源片，输出到源文件所在目录。

引擎复用 crispframe.pipeline.run_passthrough（单进程直编，SVT 全核线程、
色彩/时基元数据 passthrough）与 crispframe.verify（体积/元数据/全量解码
查错），本文件只负责界面与编排。容器清单按 AV1 实测兼容集过滤
（available_containers("svtav1")）。
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QComboBox,
                               QPushButton, QProgressBar, QSlider, QTextEdit,
                               QVBoxLayout, QWidget)

from crispframe.codec import (VIDEO_CONTAINERS, available_containers,
                              archive_pix_fmt, CodecSettings,
                              build_encode_config_kwargs)
from crispframe.ffutils import VideoInfo, probe
from crispframe.pipeline import EncodeConfig, PassthroughCanceled, run_passthrough
from crispframe.verify import human_size, probe_stream_meta, verify_output

_CRF_NOTES = {25: "稳", 27: "稳", 30: "甜点", 32: "省"}


class ProbeWorker(QThread):
    """轻量探测：只取 VideoInfo（不建 Task，与增强任务页无关）。"""
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
    log = Signal(str)
    fileDone = Signal(str, bool, str)      # 文件名, 成功, 消息
    allDone = Signal(bool, str)

    def __init__(self, items: list[tuple[Path, VideoInfo]], crf: int, container: str = "matroska"):
        super().__init__()
        self.items = items
        self.crf = crf
        self.container = container
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        ok_all = True
        for src, info in self.items:
            if self._cancel:
                self.fileDone.emit(src.name, False, "已取消")
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
            self.allDone.emit(False, "已取消")
        else:
            self.allDone.emit(ok_all, "全部完成" if ok_all else "有文件未完成存档")

    def _process_one(self, src: Path, info: VideoInfo) -> bool:
        """单文件：预检 → 单进程直编（全流）→ 三项验证。失败/取消删产物。"""
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

        encode = EncodeConfig(**build_encode_config_kwargs(
            cs, info.fps, False, map_all=True))
        cont_label = VIDEO_CONTAINERS[self.container]["label"]
        self.log.emit(f"▶ {src.name}: {info.width}x{info.height}@{info.fps:.3f} "
                      f"SVT-AV1 crf{cs.crf}/preset4 → {cont_label.split('（')[0]}"
                      f"（单进程直编，全流保留，色彩/时基跟随源），完成后自动验证")
        t0 = time.perf_counter()
        try:
            run_passthrough(info, out, encode=encode,
                            progress_callback=lambda dn, t: self.taskProgress.emit(
                                src.name, int(dn), int(t)),
                            cancel_check=lambda: self._cancel)
        except PassthroughCanceled:
            out.unlink(missing_ok=True)
            self.log.emit(f"  {src.name}: 已取消，残缺产物已删除")
            self.fileDone.emit(src.name, False, "已取消")
            return False
        except Exception as e:
            # 编码失败：残件顶着最终存档命名落盘会卡死重跑（覆盖跳过），删除之
            out.unlink(missing_ok=True)
            self.log.emit(f"[错误] {src.name}: {e}")
            self.log.emit("  编码失败的残缺产物已删除（源片保留）")
            self.fileDone.emit(src.name, False, f"编码失败: {str(e)[:120]}")
            return False
        self.log.emit(f"✓ {out.name}  编码 {time.perf_counter() - t0:.0f}s / "
                      f"{human_size(out.stat().st_size)}")

        verdict = self._verify(src, out)
        if verdict == "pass":
            self.fileDone.emit(src.name, True, out.name)
            return True
        out.unlink(missing_ok=True)       # 未过验证 = 不可归档，删除待重跑
        self.log.emit("  未通过验证的产物已删除（源片保留），可重新开始存档")
        self.fileDone.emit(src.name, False,
                           "已取消" if verdict == "cancel" else "验证未通过")
        return False

    def _verify(self, src: Path, out: Path) -> str:
        """返回 'pass' / 'fail' / 'cancel'。"""
        self.log.emit("  验证：体积 / 元数据 / 全量解码查错（约数分钟到半小时，可停止）...")
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


class ArchivePage(QWidget):
    """存档压缩独立界面：文件列表 + 一个 CRF 旋钮 + 开始/停止 + 进度与验证日志。"""
    backRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[tuple[Path, VideoInfo]] = []
        self._probing: set[Path] = set()
        self._probe_workers: list[ProbeWorker] = []
        self._gen = 0                  # 清空即换代：在途探测的回填按代丢弃
        self.worker: ArchiveWorker | None = None

        v = QVBoxLayout(self)
        v.setContentsMargins(48, 32, 48, 24)
        title = QLabel("存档压缩")
        title.setStyleSheet("font-size:22px;font-weight:bold;")
        v.addWidget(title)
        sub = QLabel("SVT-AV1 恒定画质单进程直编（CPU 全核） · 保留全部音轨/字幕/章节 · "
                     "位深/色彩/时基跟随源片 · 容器可选 · 编码后自动验证完整性")
        sub.setStyleSheet("color:#9aa0a8;")
        sub.setWordWrap(True)
        v.addWidget(sub)

        self.file_list = QListWidget()
        self.file_list.setVisible(False)
        v.addWidget(self.file_list, 1)

        crow = QHBoxLayout()
        crow.addWidget(QLabel("输出容器"))
        self.container_box = QComboBox()
        # 存档固定 SVT-AV1：容器下拉按 AV1 的实测兼容集过滤（muxer 层会硬拒
        # 其余组合），动态检测本机 ffmpeg 支持
        self._containers = available_containers("svtav1")
        for key, spec in self._containers.items():
            self.container_box.addItem(f"{spec['label']} — {spec['note']}", key)
        self.container_box.setCurrentIndex(0)     # 默认 MKV
        self.container_box.currentIndexChanged.connect(
            lambda _: (self.tip.setText(self._tip_text()),
                       self._sync_crf(self.crf_slider.value())))
        crow.addWidget(self.container_box, 1)
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
        back = QPushButton("← 返回首页")
        back.clicked.connect(self.backRequested.emit)
        bar.addWidget(back)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        bar.addWidget(self.stop_btn)
        self.start_btn = QPushButton("开始存档")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start)
        bar.addWidget(self.start_btn)
        v.addLayout(bar)

        self.bar = QProgressBar()
        v.addWidget(self.bar)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(180)
        v.addWidget(self.log_box)

    # —— 文件管理 ——
    def set_files(self, paths: list[Path]):
        """首页分流入口：入队探测，完成一个亮一个。"""
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
            w.finished.connect(lambda ww=w: self._probe_workers.remove(ww)
                               if ww in self._probe_workers else None)
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

    def _sync_buttons(self):
        busy = self.worker is not None
        self.start_btn.setEnabled(bool(self._items) and not busy)
        for b in self._list_btns:
            b.setEnabled(bool(self._items) and not busy)   # 列表即任务清单
        self.stop_btn.setEnabled(busy)

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

    # —— 运行 ——
    def _start(self):
        if self.worker or not self._items:
            return
        self.worker = ArchiveWorker(list(self._items), self.crf_slider.value(),
                                    self._container_key())
        self.worker.taskProgress.connect(
            lambda name, dn, t: (self.bar.setMaximum(t),
                                 self.bar.setValue(dn)) if t else None)
        self.worker.log.connect(self.log_box.append)
        self.worker.fileDone.connect(self._on_file_done)
        self.worker.allDone.connect(self._on_all_done)
        self._sync_buttons()
        self.bar.setValue(0)
        self.worker.start()

    def _on_file_done(self, name: str, ok: bool, msg: str):
        if ok:
            # 完成的文件移出列表（产物已落盘），未完成的留在列表便于重跑
            p = Path(name)
            self._items = [x for x in self._items if x[0] != p]
            self._take_item(p)

    def _on_all_done(self, ok: bool, msg: str):
        self.log_box.append(f"— {msg} —")
        self.worker = None
        self._sync_buttons()
        self.bar.setValue(0)

    def _stop(self):
        if self.worker:
            self.worker.cancel()
