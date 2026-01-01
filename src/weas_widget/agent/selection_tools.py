from typing import Any, Dict, List, Optional

from .tool_helpers import (
    WeasToolResult,
    _ensure_group_attribute,
    _get_current_ase_atoms,
    _get_current_atoms_data,
    _normalize_indices,
    _set_current_atoms_data,
)


def build_selection_tools(viewer: Any):
    from langchain_core.tools import tool

    @tool
    def get_selected_atoms() -> Dict[str, Any]:
        """Get the current selection as atom indices (0-based)."""
        selected = getattr(viewer.avr, "selected_atoms_indices", []) or []
        return WeasToolResult(
            "OK", summary={"selected_atoms_indices": list(selected)}
        ).to_dict()

    @tool
    def select_atoms(indices: List[int]) -> Dict[str, Any]:
        """Set the current selection by atom indices (0-based)."""
        atoms, *_ = _get_current_ase_atoms(viewer)
        _normalize_indices(indices, selected=[], n_atoms=len(atoms))
        viewer.avr.selected_atoms_indices = list(indices)
        return WeasToolResult(f"Selected {len(indices)} atoms.").to_dict()

    @tool
    def select_atoms_by_element(elements: List[str]) -> Dict[str, Any]:
        """Select atoms by element symbols (e.g., ['C', 'O'])."""
        atoms, *_ = _get_current_ase_atoms(viewer)
        elements_set = set(str(e).strip() for e in elements)
        selected = [
            i for i, atom in enumerate(atoms) if str(atom.symbol) in elements_set
        ]
        viewer.avr.selected_atoms_indices = selected
        return WeasToolResult(
            f"Selected {len(selected)} atoms with elements {sorted(elements_set)}."
        ).to_dict()

    @tool
    def list_groups() -> Dict[str, Any]:
        """List all atom groups defined on the current structure."""
        atoms_data, *_ = _get_current_atoms_data(viewer)
        attributes = atoms_data.get("attributes", {})
        atom_attrs = attributes.get("atom", {})
        groups = atom_attrs.get("groups")
        if not isinstance(groups, list):
            return WeasToolResult(
                "No groups defined.", summary={"groups": []}
            ).to_dict()
        names = sorted(
            {str(name) for entry in groups if isinstance(entry, list) for name in entry}
        )
        return WeasToolResult("OK", summary={"groups": names}).to_dict()

    @tool
    def add_atoms_to_group(
        group: str, indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Add atoms to a named group; indices omitted => current selection."""
        atoms_data, is_traj, frame, traj = _get_current_atoms_data(viewer)
        selected = getattr(viewer.avr, "selected_atoms_indices", []) or []
        n_atoms = len(atoms_data.get("symbols") or [])
        use = _normalize_indices(indices, selected=selected, n_atoms=n_atoms)
        if not use:
            return WeasToolResult(
                "Nothing to group (no indices and empty selection)."
            ).to_dict()
        groups = _ensure_group_attribute(atoms_data)
        name = str(group)
        for idx in use:
            if name not in groups[idx]:
                groups[idx].append(name)
        _set_current_atoms_data(
            viewer, atoms_data, is_trajectory=is_traj, frame=frame, trajectory=traj
        )
        return WeasToolResult(
            f"Added {len(use)} atoms to group '{name}'.",
            summary={"group": name, "indices": use},
        ).to_dict()

    @tool
    def remove_atoms_from_group(
        group: str, indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Remove atoms from a named group; indices omitted => current selection."""
        atoms_data, is_traj, frame, traj = _get_current_atoms_data(viewer)
        selected = getattr(viewer.avr, "selected_atoms_indices", []) or []
        n_atoms = len(atoms_data.get("symbols") or [])
        use = _normalize_indices(indices, selected=selected, n_atoms=n_atoms)
        if not use:
            return WeasToolResult(
                "Nothing to ungroup (no indices and empty selection)."
            ).to_dict()
        groups = _ensure_group_attribute(atoms_data)
        name = str(group)
        for idx in use:
            groups[idx] = [entry for entry in groups[idx] if entry != name]
        _set_current_atoms_data(
            viewer, atoms_data, is_trajectory=is_traj, frame=frame, trajectory=traj
        )
        return WeasToolResult(
            f"Removed {len(use)} atoms from group '{name}'.",
            summary={"group": name, "indices": use},
        ).to_dict()

    @tool
    def clear_group(group: str) -> Dict[str, Any]:
        """Remove a group from all atoms."""
        atoms_data, is_traj, frame, traj = _get_current_atoms_data(viewer)
        groups = _ensure_group_attribute(atoms_data)
        name = str(group)
        removed = 0
        for i, entry in enumerate(groups):
            new_entry = [g for g in entry if g != name]
            removed += len(entry) - len(new_entry)
            groups[i] = new_entry
        _set_current_atoms_data(
            viewer, atoms_data, is_trajectory=is_traj, frame=frame, trajectory=traj
        )
        return WeasToolResult(
            f"Cleared group '{name}' from {removed} memberships.",
            summary={"group": name, "removed": removed},
        ).to_dict()

    @tool
    def select_atoms_by_group(group: str) -> Dict[str, Any]:
        """Select atoms by group name."""
        atoms_data, *_ = _get_current_atoms_data(viewer)
        groups = _ensure_group_attribute(atoms_data)
        name = str(group)
        selected = [i for i, entry in enumerate(groups) if name in entry]
        viewer.avr.selected_atoms_indices = selected
        return WeasToolResult(
            f"Selected {len(selected)} atoms in group '{name}'.",
            summary={"group": name, "indices": selected},
        ).to_dict()

    return [
        get_selected_atoms,
        select_atoms,
        select_atoms_by_element,
        list_groups,
        add_atoms_to_group,
        remove_atoms_from_group,
        clear_group,
        select_atoms_by_group,
    ]
