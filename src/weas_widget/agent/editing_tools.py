from typing import Any, Dict, List, Optional, Union

from .tool_helpers import (
    WeasToolResult,
    _get_current_ase_atoms,
    _get_current_atoms_and_use_indices,
    _normalize_indices,
)


def build_editing_tools(viewer: Any):
    from langchain_core.tools import tool

    def _apply_fixed_xyz(
        indices: List[int], axes: Optional[Union[List[str], str]]
    ) -> Dict[str, Any]:
        atoms_payload = getattr(viewer._widget, "atoms", {})
        if isinstance(atoms_payload, list):
            frame = int(getattr(viewer.avr, "current_frame", 0))
            frame = max(0, min(frame, len(atoms_payload) - 1))
            atoms_payload = atoms_payload[frame]
        if not isinstance(atoms_payload, dict):
            atoms_payload = {}
        attributes = atoms_payload.get("attributes", {})
        atom_attrs = attributes.get("atom", {})
        atoms, *_ = _get_current_ase_atoms(viewer)
        n_atoms = len(atoms)
        fixed_xyz = atom_attrs.get("fixed_xyz")
        if not isinstance(fixed_xyz, list) or len(fixed_xyz) != n_atoms:
            fixed_xyz = [[False, False, False] for _ in range(n_atoms)]
        else:
            cleaned = []
            for i in range(n_atoms):
                entry = fixed_xyz[i] if i < len(fixed_xyz) else None
                if isinstance(entry, (list, tuple)) and len(entry) == 3:
                    cleaned.append([bool(x) for x in entry])
                else:
                    cleaned.append([False, False, False])
            fixed_xyz = cleaned

        if axes is None:
            axis_mask = [True, True, True]
        else:
            if isinstance(axes, str):
                axis_tokens = set(axes.lower().replace(",", "").replace(" ", ""))
            else:
                axis_tokens = {str(a).lower().strip() for a in axes}
            axis_mask = [
                "x" in axis_tokens,
                "y" in axis_tokens,
                "z" in axis_tokens,
            ]
            if not any(axis_mask):
                raise ValueError("axes must include at least one of 'x', 'y', or 'z'.")

        for idx in indices:
            for axis_idx, enabled in enumerate(axis_mask):
                if enabled:
                    fixed_xyz[idx][axis_idx] = True

        viewer.avr.set_attribute("fixed_xyz", fixed_xyz, domain="atom")
        fixed_indices = [i for i, mask in enumerate(fixed_xyz) if any(mask)]
        viewer.avr.highlight.settings["fixed"] = {
            "type": "crossView",
            "indices": fixed_indices,
            "scale": 1.0,
            "color": "black",
        }
        viewer.avr.draw()
        axis_names = [
            name for name, enabled in zip(("x", "y", "z"), axis_mask) if enabled
        ]
        return {
            "fixed_xyz": fixed_xyz,
            "axis_names": axis_names,
            "fixed_indices": fixed_indices,
        }

    @tool
    def add_atom(symbol: str, x: float, y: float, z: float) -> Dict[str, Any]:
        """Add a single atom at (x, y, z) in Angstrom."""
        viewer.ops.atoms.add_atom(
            symbol=symbol, position={"x": float(x), "y": float(y), "z": float(z)}
        )
        return WeasToolResult(f"Added {symbol} at ({x}, {y}, {z}).").to_dict()

    @tool
    def delete_atoms(indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """Delete atoms by indices; if omitted, deletes the current selection."""
        _, _, _, _, use = _get_current_atoms_and_use_indices(viewer, indices)
        if not use:
            return WeasToolResult(
                "Nothing to delete (no indices and empty selection)."
            ).to_dict()
        viewer.ops.object.delete(indices=use)
        return WeasToolResult(f"Deleted {len(use)} atoms.").to_dict()

    @tool
    def copy_atoms(indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """Copy atoms by indices; if omitted, uses the current selection."""
        _, _, _, _, use = _get_current_atoms_and_use_indices(viewer, indices)
        if not use:
            return WeasToolResult(
                "Nothing to copy (no indices and empty selection)."
            ).to_dict()
        viewer.ops.object.copy(indices=use)
        return WeasToolResult(f"Copied {len(use)} atoms.").to_dict()

    @tool
    def replace_atoms(
        symbol: str, indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Replace atoms (change element) by indices; if omitted, uses the current selection."""
        if indices is not None:
            viewer.avr.selected_atoms_indices = list(indices)
        else:
            indices = viewer.avr.selected_atoms_indices or []
        viewer.ops.atoms.replace(symbol=symbol, indices=indices)
        return WeasToolResult(
            f"Replaced {len(indices)} atoms with '{symbol}'."
        ).to_dict()

    @tool
    def color_by_attribute(
        attribute: str = "Element",
        color1: Optional[str] = "#ff0000",
        color2: Optional[str] = "#0000ff",
    ):
        """Color selected atoms by a given atomic attribute (e.g., 'magmom', 'charge');
        if indices omitted, uses selection."""
        viewer.ops.atoms.color_by_attribute(
            attribute=attribute, color1=color1, color2=color2
        )
        return WeasToolResult(
            f"Colored atoms by attribute '{attribute}', from {color1} to {color2}."
        ).to_dict()

    @tool
    def translate(
        vector: List[float], indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Translate atoms by (dx, dy, dz) in Angstrom; if indices omitted, uses selection."""
        if len(vector) != 3:
            raise ValueError("vector must be a length-3 list like [dx, dy, dz].")
        if indices is not None:
            viewer.avr.selected_atoms_indices = list(indices)
        else:
            indices = viewer.avr.selected_atoms_indices or []
        viewer.ops.transform.translate(vector=list(vector))
        return WeasToolResult(
            f"Translated {len(indices)} atoms by ({vector[0]}, {vector[1]}, {vector[2]})."
        ).to_dict()

    @tool
    def rotate(
        axis: List[float], angle: float, indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Rotate atoms around an axis by angle degrees; indices omitted => selection; about='com' or 'origin'."""
        if len(axis) != 3:
            raise ValueError("axis must be a length-3 list like [ax, ay, az].")
        if indices is not None:
            viewer.avr.selected_atoms_indices = list(indices)
        else:
            indices = viewer.avr.selected_atoms_indices or []
        viewer.ops.transform.rotate(axis=list(axis), angle=float(angle))
        return WeasToolResult(
            f"Rotated {len(indices)} atoms by {angle} degrees."
        ).to_dict()

    @tool
    def fix_atoms(
        indices: Optional[List[int]] = None,
        axes: Optional[Union[List[str], str]] = None,
    ) -> Dict[str, Any]:
        """Fix atoms by indices (0-based); if omitted, uses the current selection."""
        atoms, *_ = _get_current_ase_atoms(viewer)
        n_atoms = len(atoms)
        selected = getattr(viewer.avr, "selected_atoms_indices", []) or []
        use = _normalize_indices(indices, selected=selected, n_atoms=n_atoms)
        if not use:
            return WeasToolResult(
                "Nothing to fix (no indices and empty selection)."
            ).to_dict()

        info = _apply_fixed_xyz(use, axes)
        return WeasToolResult(
            f"Fixed {len(use)} atoms on axes {info['axis_names']}."
        ).to_dict()

    return [
        add_atom,
        delete_atoms,
        copy_atoms,
        replace_atoms,
        color_by_attribute,
        translate,
        rotate,
        fix_atoms,
    ]
