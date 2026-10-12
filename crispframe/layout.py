"""包根定位：向上找含 models/ 的祖先目录。

源码布局：crispframe/ 的上级即仓库根；便携包布局：runtime/app/crispframe 向上
两级到包根。所有数据路径（models/config/crash.log）一律经 ROOT 派生，
两种布局下无需任何代码差异。
"""
from pathlib import Path


def _find_root() -> Path:
    here = Path(__file__).resolve().parent
    for cand in here.parents:
        if (cand / "models").is_dir():
            return cand
    return here.parent  # 兜底：源码布局假定（crispframe/ 的上级）


ROOT = _find_root()
