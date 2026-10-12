"""输出配置文件：模块链 + 编码设置的 JSON 存取。

支撑输出设置抽屉的"保存配置文件 / 应用 / 应用到全部"。
存放于项目 config/profiles/ 下，文件名即配置名。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .codec import CodecSettings
from .layout import ROOT
from .modules import ModuleInstance

PROFILE_DIR = ROOT / "config" / "profiles"


@dataclass
class TaskProfile:
    """一份可复用的任务配置：模块链 + 编码设置。"""
    name: str
    modules: list[ModuleInstance]
    codec: CodecSettings

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "name": self.name,
            "modules": [m.to_dict() for m in self.modules],
            "codec": self.codec.to_dict(),
        }

    @staticmethod
    def from_dict(d: dict) -> "TaskProfile":
        return TaskProfile(
            name=str(d.get("name") or "未命名"),
            modules=[ModuleInstance.from_dict(m) for m in d.get("modules") or []],
            codec=CodecSettings.from_dict(d.get("codec") or {}),
        )


def save_profile(profile: TaskProfile) -> Path:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = PROFILE_DIR / f"{profile.name}.json"
    path.write_text(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def load_profile(path: Path) -> TaskProfile:
    return TaskProfile.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def list_profiles() -> list[Path]:
    if not PROFILE_DIR.exists():
        return []
    return sorted(PROFILE_DIR.glob("*.json"))
