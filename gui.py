"""CrispFrame GUI —— 视频归档压缩工具。

启动直达存档页：拖入视频 → 选 CRF/容器 → 开始存档（SVT-AV1 直编 + 三项验证）。
任务运行中关闭窗口 = 缩到系统托盘保活（后台继续跑）；空闲时关闭 = 正常退出。
全部业务逻辑在 crispframe.codec / pipeline / verify / archive_job / archive_page，
本文件只做应用壳（窗口/托盘/样式/崩溃日志/--cli 透传）。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QApplication, QMainWindow, QMenu, QMessageBox,
                               QSystemTrayIcon)

APP_TITLE = "CrispFrame — 视频归档压缩"

DARK_QSS = """
QWidget { background:#1e1f22; color:#d7d8db; font-size:13px; }
QListWidget, QTextEdit { background:#17181a; border:1px solid #2b2d31; border-radius:6px; }
QPushButton { background:#2f3136; border:1px solid #3f4147; border-radius:6px;
              padding:6px 12px; }
QPushButton:hover { background:#3a3c42; }
QPushButton:disabled { color:#6a6c72; }
QPushButton:checked { background:#2563eb; border-color:#3b82f6; color:#ffffff; }
QPushButton:checked:hover { background:#1d4fd7; }
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


def _excepthook(t, v, tb):
    """双击启动（无控制台）时把崩溃留档，否则错误无声消失。"""
    import traceback
    from crispframe.layout import ROOT
    (ROOT / "crash.log").write_text(
        "".join(traceback.format_exception(t, v, tb)), encoding="utf-8")
    sys.__stderr__ and sys.__stderr__.write(str(v))


def _dispatch_worker_modes() -> int | None:
    """隐藏模式：--cli <args...> 命令行透传到 crispframe.cli（自动化测试用）。
    返回退出码表示已处理；None = 继续 GUI。"""
    argv = sys.argv[1:]
    if "--cli" in argv:
        i = argv.index("--cli")
        from crispframe.cli import main as cli_main
        return cli_main(argv[i + 1:])
    return None


class TrayMainWindow(QMainWindow):
    """主窗口：任务运行中关窗 = 隐藏保活（托盘/任务栏），后台继续跑；空闲关窗 = 正常退出。

    托盘单击回主窗口；菜单「退出」在任务中为软停——当前段结束后停止，
    已完成分片保留（可续跑），不会硬杀丢进度。"""

    def __init__(self, page):
        super().__init__()
        self.page = page
        self._quit_pending = False
        self._tray: QSystemTrayIcon | None = None
        self._last_tip = 0.0                # 托盘 tooltip 节流（1Hz；Shell 调用非免费）
        self.setCentralWidget(page)
        self._setup_tray()
        # 页面状态上浮：进度 → 托盘 tooltip；结束 → 气泡通知 + 待退出则退出
        page.taskProgress.connect(self._on_progress)
        page.allDone.connect(self._on_page_done)

    # —— 托盘 ——
    def _setup_tray(self):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return                      # 无托盘环境：关窗退化为最小化（仍保活）
        icon = QIcon(str(Path(__file__).resolve().parent / "crispframe" / "appicon.ico"))
        if icon.isNull():
            icon = self.windowIcon()
        menu = QMenu()
        menu.addAction("显示主窗口", self._show_window)
        self._act_quit = menu.addAction("退出")
        self._act_quit.triggered.connect(self._tray_quit)
        tray = QSystemTrayIcon(icon, self)
        tray.setContextMenu(menu)
        tray.setToolTip(APP_TITLE)
        tray.activated.connect(self._on_tray_activated)
        tray.show()
        self._tray = tray

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.Trigger:          # 左键单击
            self._show_window()

    def _show_window(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def _on_progress(self, _name: str, done: int, total: int):
        # tooltip = Shell_NotifyIcon 调用（非免费）——1Hz 节流
        now = time.perf_counter()
        if self._tray is not None and total and now - self._last_tip >= 1.0:
            self._last_tip = now
            self._tray.setToolTip(f"CrispFrame 存档中 {done * 100 // max(total, 1)}%")

    def _on_page_done(self, ok: bool, msg: str):
        if self._tray is not None:
            self._tray.setToolTip(APP_TITLE)
            self._tray.showMessage(
                "CrispFrame", msg or ("存档完成" if ok else "存档结束"),
                QSystemTrayIcon.Information, 5000)
        if self._quit_pending:
            QApplication.quit()

    # —— 关窗/退出 ——
    def _tray_quit(self):
        if self.page.is_busy():
            r = QMessageBox.question(
                self, "退出 CrispFrame",
                "存档任务进行中：将在当前分段结束后停止\n"
                "（已完成分段保留，重新打开可续跑）。\n\n确定退出？")
            if r != QMessageBox.StandardButton.Yes:
                return
            self._quit_pending = True
            self.page.halt()           # 段级软停；页面 allDone 后 quit()
            self._act_quit.setText("等待当前分段结束…")
            self._act_quit.setEnabled(False)
            return
        self._quit_pending = True
        QApplication.quit()

    def closeEvent(self, ev):
        if self._quit_pending or not self.page.is_busy():
            ev.accept()
            return
        # 任务运行中：隐藏保活（有托盘 → 托盘图标；无托盘 → 最小化留任务栏）
        ev.ignore()
        self.hide()
        if self._tray is not None:
            self._tray.showMessage(
                "CrispFrame", "存档任务继续在后台运行；点托盘图标回到窗口",
                QSystemTrayIcon.Information, 4000)
        else:
            self.showMinimized()


def main():
    sys.excepthook = _excepthook
    code = _dispatch_worker_modes()
    if code is not None:
        raise SystemExit(code)
    app = QApplication(sys.argv)
    # 窗口/任务栏图标（exe 图标由 launcher 资源提供；此处管 Qt 窗口这层）
    app_icon = Path(__file__).resolve().parent / "crispframe" / "appicon.ico"
    if app_icon.exists():
        app.setWindowIcon(QIcon(str(app_icon)))
    app.setStyleSheet(DARK_QSS)
    from archive_page import ArchivePage
    page = ArchivePage()
    win = TrayMainWindow(page)
    win.setWindowTitle(APP_TITLE)
    win.resize(1120, 760)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
