from typing import Any, Dict, List, Optional

import numpy as np

from .tool_helpers import (
    WeasToolResult,
    _get_current_ase_atoms,
    _get_current_atoms_and_use_indices,
    _group_layers_by_z,
)


def build_editing_tools(viewer: Any):
    from langchain_core.tools import tool

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
    def get_top_layer_indices(
        symbol: Optional[str] = None,
        n_layers: int = 1,
        tolerance: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Return atom indices for the top-most layer(s) along +z.

        Parameters
        ----------
        symbol
            If provided, only consider atoms of this element.
        n_layers
            Number of top layers to include.
        tolerance
            Z-distance threshold (Angstrom) to group atoms into the same layer.
        """
        if n_layers <= 0:
            raise ValueError("n_layers must be >= 1.")
        atoms, *_ = _get_current_ase_atoms(viewer)
        symbols = np.array(atoms.get_chemical_symbols(), dtype=object)
        z_values = atoms.get_positions()[:, 2]

        if symbol is not None:
            mask = symbols == str(symbol)
            if not np.any(mask):
                return WeasToolResult(
                    f"No atoms found for element '{symbol}'.",
                    summary={"indices": []},
                ).to_dict()
            indices = np.where(mask)[0]
            z_slice = z_values[indices]
            layers = _group_layers_by_z(z_slice, float(tolerance))
            top_layers = layers[: int(n_layers)]
            selected = [int(indices[i]) for layer in top_layers for i in layer]
        else:
            layers = _group_layers_by_z(z_values, float(tolerance))
            top_layers = layers[: int(n_layers)]
            selected = [int(i) for layer in top_layers for i in layer]

        return WeasToolResult(
            f"Found {len(selected)} atoms in top {int(n_layers)} layer(s).",
            summary={"indices": selected},
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

    return [
        add_atom,
        delete_atoms,
        copy_atoms,
        replace_atoms,
        get_top_layer_indices,
        color_by_attribute,
        translate,
        rotate,
    ]
