from __future__ import annotations

from typing import Any, Dict, Optional

from .constants import (
    ATOM_LABEL_TYPES,
    COLOR_BYS,
    COLOR_TYPES,
    MATERIAL_TYPES,
    MODEL_STYLE_MAP,
    RADIUS_TYPES,
)
from .tool_helpers import (
    WeasToolResult,
    _canonical_style_key,
    _parse_boundary,
    _parse_enum,
    _parse_model_style,
)


def build_style_tools(viewer: Any):
    from langchain_core.tools import tool

    @tool
    def list_style_options(topic: Optional[str] = None) -> Dict[str, Any]:
        """
        List supported visualization/style controls and their allowed values.

        Topics:
        - "viewer": model_style, boundary, color_type, color_ramp,
                    radius_type, material_type, atom_label_type
        - "bond": show_bonded_atoms, hide_long_bonds, show_hydrogen_bonds, show_out_boundary_bonds
        - "cell": cell.* settings such as cell.showCell, cell.cellColor, cell.axisColors, ...
        """
        schema: Dict[str, Any] = {
            "viewer": {
                "model_style": {"type": "int|str", "values": MODEL_STYLE_MAP},
                "boundary": {
                    "type": "3x2 float list",
                    "example": [[-0.1, 1.1], [-0.1, 1.1], [-0.1, 1.1]],
                },
                "color_by": {
                    "type": "str",
                    "values": list(COLOR_BYS),
                    "note": "You may also set color_by to any per-atom attribute name present in the structure.",
                },
                "color_type": {"type": "str", "values": list(COLOR_TYPES)},
                "color_ramp": {
                    "type": "list[str]",
                    "example": ["red", "yellow", "blue"],
                },
                "radius_type": {"type": "str", "values": list(RADIUS_TYPES)},
                "material_type": {"type": "str", "values": list(MATERIAL_TYPES)},
                "atom_label_type": {"type": "str", "values": list(ATOM_LABEL_TYPES)},
                "show_atom_legend": {"type": "bool"},
                "continuous_update": {"type": "bool"},
            },
            "bond": {
                "show_bonded_atoms": {"type": "bool"},
                "hide_long_bonds": {"type": "bool"},
                "show_hydrogen_bonds": {"type": "bool"},
                "show_out_boundary_bonds": {"type": "bool"},
            },
            "cell": {
                "cell.showCell": {"type": "bool"},
                "cell.showAxes": {"type": "bool"},
                "cell.cellColor": {"type": "int", "example": 0x000000},
                "cell.cellLineWidth": {"type": "int|float", "example": 2},
                "cell.axisColors": {
                    "type": "dict",
                    "example": {"a": 0xFF0000, "b": 0x00FF00, "c": 0x0000FF},
                },
                "cell.axisRadius": {"type": "float", "example": 0.15},
                "cell.axisConeHeight": {"type": "float", "example": 0.8},
                "cell.axisConeRadius": {"type": "float", "example": 0.3},
                "cell.axisSphereRadius": {"type": "float", "example": 0.3},
            },
        }

        if topic is None:
            return WeasToolResult("OK", summary=schema).to_dict()
        t = str(topic).strip().lower()
        if t in schema:
            return WeasToolResult("OK", summary={t: schema[t]}).to_dict()
        return WeasToolResult(
            f"Unknown topic '{topic}'. Use one of: {sorted(schema)}.",
            summary=schema,
        ).to_dict()

    @tool
    def get_style(key: Optional[str] = None) -> Dict[str, Any]:
        """Get current visualization/style settings. If key is provided, returns that setting only."""
        out: Dict[str, Any] = {
            "model_style": int(getattr(viewer.avr, "model_style", 0)),
            "boundary": getattr(viewer.avr, "boundary", None),
            "color_by": getattr(viewer.avr, "color_by", None),
            "color_type": getattr(viewer.avr, "color_type", None),
            "color_ramp": getattr(viewer.avr, "color_ramp", None),
            "radius_type": getattr(viewer.avr, "radius_type", None),
            "material_type": getattr(viewer.avr, "material_type", None),
            "atom_label_type": getattr(viewer.avr, "atom_label_type", None),
            "show_bonded_atoms": bool(getattr(viewer.avr, "show_bonded_atoms", False)),
            "hide_long_bonds": bool(getattr(viewer.avr, "hide_long_bonds", True)),
            "show_hydrogen_bonds": bool(
                getattr(viewer.avr, "show_hydrogen_bonds", False)
            ),
            "show_out_boundary_bonds": bool(
                getattr(viewer.avr, "show_out_boundary_bonds", False)
            ),
            "show_atom_legend": bool(getattr(viewer.avr, "show_atom_legend", False)),
            "continuous_update": bool(getattr(viewer.avr, "continuous_update", True)),
            "cell": dict(getattr(viewer.avr.cell, "settings", {})),
        }
        if key is None:
            return WeasToolResult("OK", summary=out).to_dict()
        k = _canonical_style_key(key)
        if k.startswith("cell."):
            cell_key = k.split(".", 1)[1]
            return WeasToolResult(
                "OK", summary={k: out["cell"].get(cell_key)}
            ).to_dict()
        return WeasToolResult("OK", summary={k: out.get(k)}).to_dict()

    @tool
    def set_style(key: str, value: Any) -> Dict[str, Any]:
        """
        Set visualization/style settings.

        Examples:
        - set_style("model_style", "Polyhedra")
        - set_style("color_type", "VESTA")
        - set_style("boundary", [[-0.1,1.1],[-0.1,1.1],[-0.1,1.1]])
        - set_style("cell.showCell", True)
        - set_style("atom_label_type", "Index")
        """
        k = _canonical_style_key(key)

        if k == "model_style":
            viewer.avr.model_style = _parse_model_style(value)
            return WeasToolResult(
                f"Set model_style to {viewer.avr.model_style}."
            ).to_dict()

        if k == "boundary":
            viewer.avr.boundary = _parse_boundary(value)
            return WeasToolResult("Set boundary.").to_dict()

        if k == "color_type":
            viewer.avr.color_type = _parse_enum(value, COLOR_TYPES, name="color_type")
            return WeasToolResult(
                f"Set color_type to {viewer.avr.color_type}."
            ).to_dict()

        if k == "radius_type":
            viewer.avr.radius_type = _parse_enum(
                value, RADIUS_TYPES, name="radius_type"
            )
            return WeasToolResult(
                f"Set radius_type to {viewer.avr.radius_type}."
            ).to_dict()

        if k == "material_type":
            viewer.avr.material_type = _parse_enum(
                value, MATERIAL_TYPES, name="material_type"
            )
            return WeasToolResult(
                f"Set material_type to {viewer.avr.material_type}."
            ).to_dict()

        if k == "atom_label_type":
            viewer.avr.atom_label_type = _parse_enum(
                value, ATOM_LABEL_TYPES, name="atom_label_type"
            )
            return WeasToolResult(
                f"Set atom_label_type to {viewer.avr.atom_label_type}."
            ).to_dict()

        if k in {
            "show_bonded_atoms",
            "hide_long_bonds",
            "show_hydrogen_bonds",
            "show_out_boundary_bonds",
        }:
            if not isinstance(value, (bool, int)):
                raise ValueError(f"{k} must be a boolean.")
            setattr(viewer.avr, k, bool(value))
            return WeasToolResult(
                f"Set {k} to {bool(getattr(viewer.avr, k))}."
            ).to_dict()

        if k in {"show_atom_legend", "continuous_update"}:
            if not isinstance(value, (bool, int)):
                raise ValueError(f"{k} must be a boolean.")
            setattr(viewer.avr, k, bool(value))
            return WeasToolResult(
                f"Set {k} to {bool(getattr(viewer.avr, k))}."
            ).to_dict()

        if k.startswith("cell."):
            cell_key = k.split(".", 1)[1]
            viewer.avr.cell.settings[cell_key] = value
            return WeasToolResult(f"Set cell.{cell_key}.").to_dict()

        raise ValueError(
            f"Unknown style key '{key}'. Use list_style_options() to see supported keys."
        )

    return [list_style_options, get_style, set_style]
