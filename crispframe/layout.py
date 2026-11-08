"""包根定位：crispframe/ 包目录的上级（确定值，不依赖任何标记目录）。

源码布局：crispframe/ 的上级即仓库根；便携包布局：runtime/app/crispframe 的
上级 runtime/app（随包 ffmpeg 在再上一级包根 ffmpeg/，由 ffutils 搜索基覆盖）。
crash.log 等可写文件一律经 ROOT 派生，两种布局下无需任何代码差异。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
