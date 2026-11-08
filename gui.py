"""CrispFrame GUI —— 视频归档压缩工具。

启动直达存档页：拖入视频 → 选 CRF/容器 → 开始存档（SVT-AV1 直编 + 三项验证）。
全部业务逻辑在 crispframe.codec / pipeline / verify / archive_page，
本文件只做应用壳（窗口/样式/崩溃日志/--cli 透传）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow

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


def main():
    sys.excepthook = _excepthook
    code = _dispatch_worker_modes()
    if code is not None:
        raise SystemExit(code)
    app = QApplication(sys.argv)
    # 窗口/任务栏图标（exe 图标由 launcher 资源提供；此处管 Qt 窗口这层）
    app_icon = Path(__file__).resolve().parent / "crispframe" / "appicon.ico"
    if app_icon.exists():
        from PySide6.QtGui import QIcon
        app.setWindowIcon(QIcon(str(app_icon)))
    app.setStyleSheet(DARK_QSS)
    from archive_page import ArchivePage
    win = QMainWindow()
    win.setWindowTitle(APP_TITLE)
    win.resize(1120, 760)
    win.setCentralWidget(ArchivePage())
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
