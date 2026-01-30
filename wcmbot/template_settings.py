"""Template configuration loader for WCMBot."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_CONFIG_PATH = BASE_DIR / "media" / "templates" / "templates.json"


@dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    label: str
    template_path: Path
    rows: int
    cols: int
    crop_x: int = 0
    crop_y: int = 0
    default_rotation: int = 0
    auto_align_default: bool = True
    export_width_cm: float = 66.0
    piece_dirs: Tuple[Path, ...] = field(default_factory=tuple)
    matcher_overrides: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TemplateRegistry:
    templates: Dict[str, TemplateSpec]
    default_template_id: str

    def choices(self) -> List[Tuple[str, str]]:
        return [(spec.label, spec.template_id) for spec in self.templates.values()]

    def get(self, template_id: str) -> TemplateSpec:
        return self.templates.get(template_id, self.templates[self.default_template_id])


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def load_template_registry(
    config_path: Path = TEMPLATE_CONFIG_PATH,
) -> TemplateRegistry:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Template config not found at {config_path}. "
            "Please add a templates.json file."
        )
    data = json.loads(config_path.read_text(encoding="utf-8"))
    default_id = data.get("default_template")
    templates_data = data.get("templates", [])
    if not templates_data:
        raise ValueError("Template config must include a non-empty templates list.")

    templates: Dict[str, TemplateSpec] = {}
    for entry in templates_data:
        template_id = entry["id"]
        label = entry.get("label", template_id)
        template_path = _resolve_path(entry["template_path"])
        piece_dirs = tuple(_resolve_path(path) for path in entry.get("piece_dirs", []))
        matcher_overrides = entry.get("matcher_overrides", {}) or {}
        spec = TemplateSpec(
            template_id=template_id,
            label=label,
            template_path=template_path,
            rows=int(entry["rows"]),
            cols=int(entry["cols"]),
            crop_x=int(entry.get("crop_x", 0)),
            crop_y=int(entry.get("crop_y", 0)),
            default_rotation=int(entry.get("default_rotation", 0)),
            auto_align_default=bool(entry.get("auto_align_default", True)),
            export_width_cm=float(entry.get("export_width_cm", 66.0)),
            piece_dirs=piece_dirs,
            matcher_overrides=matcher_overrides,
        )
        templates[template_id] = spec

    if default_id is None or default_id not in templates:
        default_id = templates_data[0]["id"]

    return TemplateRegistry(templates=templates, default_template_id=default_id)
