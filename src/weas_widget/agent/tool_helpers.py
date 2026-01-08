from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..utils import group_layers_by_coordinate

from .constants import MODEL_STYLE_MAP


def _to_jsonable(value: Any) -> Any:
    """Convert common scientific Python types to JSON-serializable structures."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _get_current_ase_atoms(viewer: Any):
    atoms_or_traj = viewer.to_ase()
    if isinstance(atoms_or_traj, list):
        frame = int(getattr(viewer.avr, "current_frame", 0))
        frame = max(0, min(frame, len(atoms_or_traj) - 1))
        return atoms_or_traj[frame].copy(), True, frame, atoms_or_traj
    return atoms_or_traj.copy(), False, None, None


def _set_current_ase_atoms(
    viewer: Any,
    atoms: Any,
    *,
    is_trajectory: bool,
    frame: Optional[int],
    trajectory: Optional[list],
):
    if is_trajectory:
        assert trajectory is not None and frame is not None
        trajectory = list(trajectory)
        trajectory[frame] = atoms
        viewer.from_ase(trajectory)
    else:
        viewer.from_ase(atoms)


def _get_current_atoms_data(
    viewer: Any,
) -> Tuple[Dict[str, Any], bool, Optional[int], Optional[list]]:
    atoms_data = deepcopy(getattr(viewer, "_widget").atoms)
    if isinstance(atoms_data, list):
        frame = int(getattr(viewer.avr, "current_frame", 0))
        frame = max(0, min(frame, len(atoms_data) - 1))
        return deepcopy(atoms_data[frame]), True, frame, atoms_data
    return atoms_data, False, None, None


def _set_current_atoms_data(
    viewer: Any,
    atoms_data: Dict[str, Any],
    *,
    is_trajectory: bool,
    frame: Optional[int],
    trajectory: Optional[list],
):
    if is_trajectory:
        assert trajectory is not None and frame is not None
        trajectory = list(trajectory)
        trajectory[frame] = atoms_data
        viewer._widget.atoms = trajectory
    else:
        viewer._widget.atoms = atoms_data


def _ensure_group_attribute(atoms_data: Dict[str, Any]) -> List[List[str]]:
    symbols = atoms_data.get("symbols") or []
    attributes = atoms_data.setdefault("attributes", {})
    atom_attrs = attributes.setdefault("atom", {})
    groups = atom_attrs.get("groups")
    if not isinstance(groups, list):
        groups = [[] for _ in range(len(symbols))]
        atom_attrs["groups"] = groups
    if len(groups) < len(symbols):
        groups.extend([[] for _ in range(len(symbols) - len(groups))])
    elif len(groups) > len(symbols):
        groups = list(groups[: len(symbols)])
        atom_attrs["groups"] = groups
    for i, entry in enumerate(groups):
        if not isinstance(entry, list):
            groups[i] = []
        else:
            groups[i] = [str(name) for name in entry]
    return groups


def _normalize_indices(
    indices: Optional[Sequence[int]], selected: Sequence[int], n_atoms: int
) -> List[int]:
    use = list(selected) if indices is None else list(indices)
    use = [int(i) for i in use]
    bad = [i for i in use if i < 0 or i >= n_atoms]
    if bad:
        raise ValueError(
            f"Atom indices out of range: {bad}. Valid range is [0, {n_atoms - 1}]."
        )
    return use


def _get_current_atoms_and_use_indices(
    viewer: Any, indices: Optional[Sequence[int]]
) -> Tuple[Any, bool, Optional[int], Optional[list], List[int]]:
    atoms, is_traj, frame, traj = _get_current_ase_atoms(viewer)
    selected = getattr(viewer.avr, "selected_atoms_indices", []) or []
    use = _normalize_indices(indices, selected=selected, n_atoms=len(atoms))
    return atoms, is_traj, frame, traj, use


def _canonical_style_key(key: str) -> str:
    key = str(key).strip()
    key = key.replace("-", "_")
    lower = key.lower().replace(" ", "")

    if lower.startswith("cell."):
        return "cell." + key.split(".", 1)[1]

    aliases = {
        "modelstyle": "model_style",
        "model_style": "model_style",
        "boundary": "boundary",
        "colorby": "color_by",
        "color_by": "color_by",
        "colortype": "color_type",
        "color_type": "color_type",
        "colorramp": "color_ramp",
        "color_ramp": "color_ramp",
        "radiustype": "radius_type",
        "radius_type": "radius_type",
        "materialtype": "material_type",
        "material_type": "material_type",
        "atomlabeltype": "atom_label_type",
        "atom_label_type": "atom_label_type",
        "showbondedatoms": "show_bonded_atoms",
        "show_bonded_atoms": "show_bonded_atoms",
        "hidelongbonds": "hide_long_bonds",
        "hide_long_bonds": "hide_long_bonds",
        "showhydrogenbonds": "show_hydrogen_bonds",
        "show_hydrogen_bonds": "show_hydrogen_bonds",
        "showoutboundarybonds": "show_out_boundary_bonds",
        "show_out_boundary_bonds": "show_out_boundary_bonds",
        "showatomlegend": "show_atom_legend",
        "show_atom_legend": "show_atom_legend",
        "atom_label": "atom_label_type",
        "label": "atom_label_type",
        "continuousupdate": "continuous_update",
        "continuous_update": "continuous_update",
    }
    return aliases.get(lower, key)


def _parse_model_style(value: Union[int, str]) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if not isinstance(value, str):
        raise ValueError(
            f"model_style must be an int or one of {sorted(MODEL_STYLE_MAP)}."
        )
    v = value.strip()
    if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
        return int(v)
    folded = v.lower().replace(" ", "")
    for name, code in MODEL_STYLE_MAP.items():
        if folded == name.lower().replace(" ", ""):
            return int(code)
    raise ValueError(
        f"Unknown model_style '{value}'. Supported: {sorted(MODEL_STYLE_MAP)} or an int."
    )


def _parse_enum(value: Any, options: Tuple[str, ...], *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be one of {list(options)}.")
    v = value.strip()
    for opt in options:
        if v.lower() == opt.lower():
            return opt
    raise ValueError(f"Unknown {name} '{value}'. Supported: {list(options)}.")


def _parse_boundary(value: Any) -> List[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(
            "boundary must be a 3x2 list like [[xmin,xmax],[ymin,ymax],[zmin,zmax]]."
        )
    out: List[List[float]] = []
    for axis in value:
        if not isinstance(axis, (list, tuple)) or len(axis) != 2:
            raise ValueError(
                "boundary must be a 3x2 list like [[xmin,xmax],[ymin,ymax],[zmin,zmax]]."
            )
        out.append([float(axis[0]), float(axis[1])])
    return out


def _cell_center_xy(atoms: Any) -> np.ndarray:
    cell = np.asarray(atoms.get_cell().array, dtype=float)
    if cell.shape != (3, 3):
        return np.zeros(3, dtype=float)
    return 0.5 * (cell[0] + cell[1])


def _group_layers_by_z(z_values: np.ndarray, tolerance: float) -> List[List[int]]:
    return group_layers_by_coordinate(z_values, tolerance, descending=True)


@dataclass(frozen=True)
class WeasToolResult:
    message: str
    summary: Optional[Dict[str, Any]] = None
    confirmation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"message": self.message}
        if self.summary is not None:
            out["summary"] = self.summary
        if self.confirmation is not None:
            out["confirmation"] = self.confirmation
        return out
