"""CPU 亲和性调度：混合架构（大小核）的拓扑枚举与进程级逐核绑定。

无第三方依赖（ctypes 直调 Win32）。拓扑经 GetLogicalProcessorInformationEx
的 EfficiencyClass 区分大核（P，>0）/小核（E，=0）；仅支持单处理器组
（≤64 逻辑处理器——混合架构消费级 CPU 均在此范围）。非 Windows 平台为 no-op。
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

if sys.platform == "win32":
    import ctypes.wintypes as wt

    class _GROUP_AFFINITY(ctypes.Structure):
        _fields_ = [("Mask", ctypes.c_size_t), ("Group", ctypes.c_uint16),
                    ("Reserved", ctypes.c_uint16 * 3)]

    class _PROCESSOR_RELATIONSHIP(ctypes.Structure):
        _fields_ = [("Flags", ctypes.c_byte), ("EfficiencyClass", ctypes.c_byte),
                    ("Reserved", ctypes.c_byte * 20), ("GroupCount", ctypes.c_uint16),
                    ("GroupMask", _GROUP_AFFINITY * 1)]

    class _SLPIE(ctypes.Structure):
        class _U(ctypes.Union):
            _fields_ = [("proc", _PROCESSOR_RELATIONSHIP)]
        _anonymous_ = ("U",)
        _fields_ = [("Relationship", ctypes.c_int), ("Size", ctypes.c_uint32),
                    ("U", _U)]

_k32 = ctypes.windll.kernel32 if sys.platform == "win32" else None
_topology_cache: tuple[list[int], list[int]] | None = None


def topology() -> tuple[list[int], list[int]]:
    """返回 (大核逻辑处理器列表, 小核逻辑处理器列表)。结果缓存。

    非混合架构（全部 EfficiencyClass=0）时大核表为空、小核表=全部；
    非 Windows 平台两表均为空。"""
    global _topology_cache
    if _topology_cache is not None:
        return _topology_cache
    if sys.platform != "win32":
        _topology_cache = ([], [])
        return _topology_cache
    size = wt.DWORD(0)
    _k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))  # 设计如此：FALSE+填充 size
    if size.value == 0:
        raise OSError("GetLogicalProcessorInformationEx 未返回拓扑大小")
    buf = (ctypes.c_char * size.value)()
    if not _k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
        raise OSError("GetLogicalProcessorInformationEx 失败")
    p_cores: list[int] = []
    e_cores: list[int] = []
    off = 0
    while off < size.value:
        e = ctypes.cast(ctypes.byref(buf, off), ctypes.POINTER(_SLPIE)).contents
        if e.Relationship == 0:   # RelationProcessorCore
            mask = e.proc.GroupMask[0].Mask
            bits = [i for i in range(64) if mask & (1 << i)]
            (p_cores if e.proc.EfficiencyClass > 0 else e_cores).extend(bits)
        off += e.Size
    _topology_cache = (p_cores, e_cores)
    return _topology_cache


def _mask_for(cpu_ids: list[int]) -> int:
    mask = 0
    for i in cpu_ids:
        if not 0 <= i < 64:
            raise ValueError(f"逻辑处理器编号超界（仅支持单组 0-63）: {i}")
        mask |= 1 << i
    if mask == 0:
        raise ValueError("亲和性掩码为空——至少选择一个逻辑处理器")
    return mask


def apply(proc, cpu_ids: list[int]) -> bool:
    """对运行中的进程设置亲和掩码（勾选的逻辑处理器集合）。

    proc: subprocess.Popen（用其 _handle）。成功 True；失败 False（调用方 log）。
    """
    if sys.platform != "win32":
        return False
    mask = _mask_for(cpu_ids)
    return bool(_k32.SetProcessAffinityMask(int(proc._handle), ctypes.c_size_t(mask)))


def current_mask(proc) -> int:
    """回读进程当前亲和掩码（测试用）。"""
    if sys.platform != "win32":
        return 0
    m = ctypes.c_size_t(0)
    pm = ctypes.c_size_t(0)
    if not _k32.GetProcessAffinityMask(int(proc._handle), ctypes.byref(m), ctypes.byref(pm)):
        raise OSError("GetProcessAffinityMask 失败")
    return m.value
