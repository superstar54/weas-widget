from __future__ import annotations

from dataclasses import dataclass
import inspect
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

MODEL_STYLE_MAP: Dict[str, int] = {
    "Ball": 0,
    "Ball + Stick": 1,
    "Polyhedra": 2,
    "Stick": 3,
    "Line": 4,
}

COLOR_TYPES: Tuple[str, ...] = ("JMOL", "VESTA", "CPK")
COLOR_BYS: Tuple[str, ...] = ("Element", "Index", "Random", "Uniform")
RADIUS_TYPES: Tuple[str, ...] = ("Covalent", "VDW")
MATERIAL_TYPES: Tuple[str, ...] = ("Standard", "Phong", "Basic")
ATOM_LABEL_TYPES: Tuple[str, ...] = ("None", "Symbol", "Index")


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


def _rotation_matrix(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("Rotation axis must be non-zero.")
    axis = axis / norm
    angle = np.deg2rad(float(angle_degrees))
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class WeasToolResult:
    message: str
    summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"message": self.message}
        if self.summary is not None:
            out["summary"] = self.summary
        return out


class WeasToolkit:
    """
    Build a collection of LangChain-compatible tools for driving a WeasWidget.

    Extensions
    ----------
    You can extend the toolkit in two ways:

    1) Register directly:

        toolkit = WeasToolkit(viewer)
        toolkit.register(my_tool)  # a langchain_core.tools.BaseTool
        toolkit.register(lambda viewer: [tool_a(viewer), tool_b(viewer)])  # factories are supported

    2) Plugin entry points:
       Provide entry points under the group ``weas_widget.tools``.
       Each entry point should resolve to either:
         - a tool instance, or
         - a callable returning a tool or list of tools (optionally accepting ``viewer``).
    """

    def __init__(
        self,
        viewer: Any,
        *,
        extra_tools: Optional[Sequence[Any]] = None,
        load_entry_points: bool = True,
        entry_point_group: str = "weas_widget.tools",
    ) -> None:
        self.viewer = viewer
        self._tools = self._build_tools()
        if load_entry_points:
            self._extend_from_entry_points(entry_point_group)
        if extra_tools:
            self.extend(extra_tools)

    @property
    def tools(self):
        return list(self._tools)

    def extend(
        self,
        tools: Sequence[Any],
        *,
        on_conflict: Literal["skip", "replace", "error"] = "skip",
    ) -> None:
        resolved = list(_resolve_tool_specs(tools, self.viewer))
        self._merge_tools(resolved, on_conflict=on_conflict, source="extra_tools")

    def register(
        self, tool: Any, *, on_conflict: Literal["skip", "replace", "error"] = "skip"
    ) -> None:
        self.extend([tool], on_conflict=on_conflict)

    def _merge_tools(
        self,
        new_tools: Sequence[Any],
        *,
        on_conflict: Literal["skip", "replace", "error"],
        source: str,
    ) -> None:
        existing = {getattr(t, "name", None): i for i, t in enumerate(self._tools)}
        for t in new_tools:
            name = getattr(t, "name", None)
            if not name:
                raise TypeError(f"Invalid tool from {source}: missing .name")
            if name in existing:
                if on_conflict == "skip":
                    warnings.warn(
                        f"Skipping duplicate tool name {name!r} from {source}.",
                        RuntimeWarning,
                    )
                    continue
                if on_conflict == "error":
                    raise ValueError(f"Duplicate tool name {name!r} from {source}.")
                self._tools[existing[name]] = t
            else:
                existing[name] = len(self._tools)
                self._tools.append(t)

    def _extend_from_entry_points(self, group: str) -> None:
        for spec in _load_tool_entry_points(group):
            try:
                resolved = list(_resolve_tool_specs([spec], self.viewer))
            except Exception as e:
                warnings.warn(
                    f"Failed to load tool plugin from entry point group {group!r}: {e}",
                    RuntimeWarning,
                )
                continue
            self._merge_tools(
                resolved, on_conflict="skip", source=f"entry_points:{group}"
            )

    def _build_tools(self):
        try:
            from langchain_core.tools import tool
        except Exception as e:
            raise ImportError(
                "Missing optional dependencies for the agent. Install with `pip install weas-widget[agent]`."
            ) from e

        viewer = self.viewer

        @tool
        def list_style_options(topic: Optional[str] = None) -> Dict[str, Any]:
            """
            List supported visualization/style controls and their allowed values.

            Topics:
            - "viewer": model_style, boundary, color_by, color_type, color_ramp,
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
                    "atom_label_type": {
                        "type": "str",
                        "values": list(ATOM_LABEL_TYPES),
                    },
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
                "show_bonded_atoms": bool(
                    getattr(viewer.avr, "show_bonded_atoms", False)
                ),
                "hide_long_bonds": bool(getattr(viewer.avr, "hide_long_bonds", True)),
                "show_hydrogen_bonds": bool(
                    getattr(viewer.avr, "show_hydrogen_bonds", False)
                ),
                "show_out_boundary_bonds": bool(
                    getattr(viewer.avr, "show_out_boundary_bonds", False)
                ),
                "show_atom_legend": bool(
                    getattr(viewer.avr, "show_atom_legend", False)
                ),
                "continuous_update": bool(
                    getattr(viewer.avr, "continuous_update", True)
                ),
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
            - set_style("color_by", "Element") or set_style("color_by", "Force")  # attribute coloring
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
                viewer.avr.color_type = _parse_enum(
                    value, COLOR_TYPES, name="color_type"
                )
                return WeasToolResult(
                    f"Set color_type to {viewer.avr.color_type}."
                ).to_dict()

            if k == "color_by":
                if not isinstance(value, str):
                    raise ValueError("color_by must be a string.")
                viewer.avr.color_by = str(value)
                return WeasToolResult(
                    f"Set color_by to {viewer.avr.color_by}."
                ).to_dict()

            if k == "color_ramp":
                if not isinstance(value, list) or not all(
                    isinstance(x, str) for x in value
                ):
                    raise ValueError(
                        "color_ramp must be a list of color strings, e.g. ['red','blue']."
                    )
                viewer.avr.color_ramp = list(value)
                return WeasToolResult("Set color_ramp.").to_dict()

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

        @tool
        def get_selected_atoms() -> Dict[str, Any]:
            """Get the current selection as atom indices (0-based)."""
            selected = getattr(viewer.avr, "selected_atoms_indices", []) or []
            return WeasToolResult(
                "OK", summary={"selected_atoms_indices": list(selected)}
            ).to_dict()

        @tool
        def get_structure_summary() -> Dict[str, Any]:
            """Return a compact summary of the current structure in the viewer."""
            atoms, is_traj, frame, traj = _get_current_ase_atoms(viewer)
            cell = atoms.get_cell().array
            pbc = atoms.get_pbc().tolist()
            summary = {
                "n_atoms": int(len(atoms)),
                "formula": atoms.get_chemical_formula(),
                "pbc": pbc,
                "cell": cell.tolist(),
                "is_trajectory": bool(is_traj),
                "frame": int(frame) if is_traj else None,
            }
            return WeasToolResult("OK", summary=summary).to_dict()

        @tool
        def select_atoms(indices: List[int]) -> Dict[str, Any]:
            """Set the current selection by atom indices (0-based)."""
            atoms, *_ = _get_current_ase_atoms(viewer)
            _normalize_indices(indices, selected=[], n_atoms=len(atoms))
            viewer.avr.selected_atoms_indices = list(indices)
            return WeasToolResult(f"Selected {len(indices)} atoms.").to_dict()

        @tool
        def load_molecule(name: str) -> Dict[str, Any]:
            """Load an ASE molecule by name/formula (e.g., 'H2O', 'CO2', 'benzene') into the viewer."""
            from ase.build import molecule

            atoms = molecule(name)
            viewer.from_ase(atoms)
            viewer.avr.selected_atoms_indices = []
            return WeasToolResult(
                f"Loaded molecule '{name}' with {len(atoms)} atoms."
            ).to_dict()

        @tool
        def load_bulk(
            symbol: str,
            crystalstructure: str = "fcc",
            a: Optional[float] = None,
            cubic: bool = True,
            repeat: Optional[List[int]] = None,
        ) -> Dict[str, Any]:
            """Load a bulk crystal using ASE `bulk` and optionally repeat it."""
            from ase.build import bulk

            atoms = bulk(symbol, crystalstructure, a=a, cubic=cubic)
            if repeat is not None:
                if len(repeat) != 3:
                    raise ValueError(
                        "repeat must be a length-3 list like [nx, ny, nz]."
                    )
                atoms = atoms * tuple(int(x) for x in repeat)
            viewer.from_ase(atoms)
            viewer.avr.selected_atoms_indices = []
            return WeasToolResult(
                f"Loaded bulk {symbol} ({crystalstructure}) with {len(atoms)} atoms."
            ).to_dict()

        @tool
        def load_fcc_surface(
            symbol: str = "Pt",
            miller: List[int] = [1, 1, 1],
            size: List[int] = [3, 3, 4],
            vacuum: float = 10.0,
            a: Optional[float] = None,
            orthogonal: bool = True,
        ) -> Dict[str, Any]:
            """
            Build and load an FCC surface slab (e.g., Pt(111)).

            Supports any Miller index. For common low-index surfaces (111/100/110),
            ASE's specialized builders are used. For other indices, a general
            surface is built via ``ase.build.surface(ase.build.bulk(...), ...)``.

            Parameters
            ----------
            symbol
                Element symbol, e.g. "Pt".
            miller
                Miller index, e.g. [1,1,1] for (111).
            size
                [nx, ny, nlayers] for in-plane repetition and number of layers.
            vacuum
                Vacuum thickness in Angstrom added along z.
            a
                Lattice constant (optional, ASE will use a default if omitted).
            orthogonal
                Whether to build an orthogonal cell (recommended for slabs).
            """
            from ase.build import bulk, fcc100, fcc110, fcc111, surface

            if len(miller) != 3:
                raise ValueError("miller must be a length-3 list like [1,1,1].")
            if len(size) != 3:
                raise ValueError("size must be a length-3 list like [nx, ny, nlayers].")
            nx, ny, nlayers = (int(size[0]), int(size[1]), int(size[2]))
            h, k, l = (int(miller[0]), int(miller[1]), int(miller[2]))  # noqa: E741

            if (h, k, l) == (1, 1, 1):
                slab = fcc111(
                    symbol,
                    size=(nx, ny, nlayers),
                    a=a,
                    vacuum=float(vacuum),
                    orthogonal=orthogonal,
                )
            elif (h, k, l) == (1, 0, 0):
                slab = fcc100(
                    symbol,
                    size=(nx, ny, nlayers),
                    a=a,
                    vacuum=float(vacuum),
                    orthogonal=orthogonal,
                )
            elif (h, k, l) == (1, 1, 0):
                slab = fcc110(
                    symbol,
                    size=(nx, ny, nlayers),
                    a=a,
                    vacuum=float(vacuum),
                    orthogonal=orthogonal,
                )
            else:
                lattice = bulk(symbol, crystalstructure="fcc", a=a, cubic=True)
                slab = surface(
                    lattice,
                    indices=(h, k, l),
                    layers=nlayers,
                    vacuum=float(vacuum),
                    periodic=True,
                )
                slab = slab.repeat((nx, ny, 1))

            viewer.from_ase(slab)
            viewer.avr.selected_atoms_indices = []
            return WeasToolResult(
                f"Loaded {symbol}({h}{k}{l}) slab with {len(slab)} atoms.",
                summary={
                    "symbol": symbol,
                    "miller": [h, k, l],
                    "size": [nx, ny, nlayers],
                },
            ).to_dict()

        @tool
        def load_surface(
            symbol: str,
            crystalstructure: str = "fcc",
            miller: List[int] = [1, 1, 1],
            size: List[int] = [3, 3, 4],
            vacuum: float = 10.0,
            a: Optional[float] = None,
            cubic: bool = True,
        ) -> Dict[str, Any]:
            """
            Build and load a general surface slab using ASE.

            This is the general counterpart to `load_fcc_surface` and supports any
            crystal structure accepted by `ase.build.bulk` and any Miller index.
            """
            from ase.build import bulk, surface

            if len(miller) != 3:
                raise ValueError("miller must be a length-3 list like [1,1,1].")
            if len(size) != 3:
                raise ValueError("size must be a length-3 list like [nx, ny, nlayers].")
            nx, ny, nlayers = (int(size[0]), int(size[1]), int(size[2]))
            h, k, l = (int(miller[0]), int(miller[1]), int(miller[2]))  # noqa: E741

            lattice = bulk(
                symbol, crystalstructure=str(crystalstructure), a=a, cubic=bool(cubic)
            )
            slab = surface(
                lattice,
                indices=(h, k, l),
                layers=nlayers,
                vacuum=float(vacuum),
                periodic=True,
            )
            slab = slab.repeat((nx, ny, 1))

            viewer.from_ase(slab)
            viewer.avr.selected_atoms_indices = []
            return WeasToolResult(
                f"Loaded {symbol} {crystalstructure}({h}{k}{l}) slab with {len(slab)} atoms.",
                summary={
                    "symbol": symbol,
                    "crystalstructure": str(crystalstructure),
                    "miller": [h, k, l],
                    "size": [nx, ny, nlayers],
                },
            ).to_dict()

        @tool
        def append_molecule(
            name: str,
            position: Optional[List[float]] = None,
            select_added: bool = True,
        ) -> Dict[str, Any]:
            """
            Append an ASE molecule (e.g., "H2O") to the current structure.

            If position is omitted and a structure already exists, the molecule is placed above the current top surface.
            """
            from ase.build import molecule

            atoms, is_traj, frame, traj = _get_current_ase_atoms(viewer)
            before_n = int(len(atoms))
            mol = molecule(name)

            if position is not None:
                if len(position) != 3:
                    raise ValueError("position must be a length-3 list like [x,y,z].")
                target = np.array(
                    [float(position[0]), float(position[1]), float(position[2])],
                    dtype=float,
                )
            else:
                pos = atoms.get_positions()
                z_top = float(np.max(pos[:, 2])) if len(pos) else 0.0
                center = _cell_center_xy(atoms)
                target = np.array([center[0], center[1], z_top + 5.0], dtype=float)

            mol_pos = mol.get_positions()
            mol_com = mol_pos.mean(axis=0)
            mol.translate(target - mol_com)
            atoms += mol
            _set_current_ase_atoms(
                viewer, atoms, is_trajectory=is_traj, frame=frame, trajectory=traj
            )

            added = list(range(before_n, before_n + len(mol)))
            if select_added:
                viewer.avr.selected_atoms_indices = added
            return WeasToolResult(
                f"Appended molecule '{name}' ({len(mol)} atoms).",
                summary={"added_indices": added, "selected_added": bool(select_added)},
            ).to_dict()

        @tool
        def place_selected_on_top(
            clearance: float = 2.0,
            surface_indices: Optional[List[int]] = None,
            align_xy: Literal["cell_center", "surface_com", "none"] = "cell_center",
        ) -> Dict[str, Any]:
            """
            Move the currently selected atoms so they sit above a surface/slab by a given clearance along +z.

            - If surface_indices is omitted, uses all non-selected atoms as the surface.
            - align_xy controls optional x/y alignment (recommended: "cell_center").
            """
            from ase.build import add_adsorbate

            atoms, is_traj, frame, traj = _get_current_ase_atoms(viewer)
            selected = getattr(viewer.avr, "selected_atoms_indices", []) or []
            sel = _normalize_indices(list(selected), selected=[], n_atoms=len(atoms))
            if not sel:
                raise ValueError(
                    "No selected atoms. Select the molecule/fragment first."
                )

            if surface_indices is None:
                sel_set = set(sel)
                surf = [i for i in range(len(atoms)) if i not in sel_set]
            else:
                surf = _normalize_indices(
                    surface_indices, selected=[], n_atoms=len(atoms)
                )
            if not surf:
                raise ValueError("Surface atom set is empty.")

            pos = atoms.get_positions()

            dxy = np.zeros(2, dtype=float)
            if align_xy == "cell_center":
                center = _cell_center_xy(atoms)
                target_xy = np.array([center[0], center[1]], dtype=float)
                sel_xy = pos[sel, :2].mean(axis=0)
                dxy = target_xy - sel_xy
            elif align_xy == "surface_com":
                target_xy = pos[surf, :2].mean(axis=0)
                sel_xy = pos[sel, :2].mean(axis=0)
                dxy = target_xy - sel_xy
            elif align_xy != "none":
                raise ValueError(
                    "align_xy must be one of: cell_center, surface_com, none."
                )

            slab = atoms[surf].copy()
            slab.set_cell(atoms.get_cell())
            slab.set_pbc(atoms.get_pbc())
            ads = atoms[sel].copy()
            old_ads_pos = ads.get_positions().copy()
            mol_index = int(np.argmin(old_ads_pos[:, 2]))

            if align_xy == "none":
                position_xy = tuple(float(x) for x in old_ads_pos[mol_index, :2])
            else:
                position_xy = tuple(
                    float(x) for x in (old_ads_pos[mol_index, :2] + dxy)
                )

            add_adsorbate(
                slab,
                ads,
                height=float(clearance),
                position=position_xy,
                mol_index=mol_index,
            )
            new_ads_pos = slab.get_positions()[-len(ads) :]
            shift = (new_ads_pos - old_ads_pos)[0]
            pos[sel] = pos[sel] + shift
            atoms.set_positions(pos)
            _set_current_ase_atoms(
                viewer, atoms, is_trajectory=is_traj, frame=frame, trajectory=traj
            )
            return WeasToolResult(
                f"Moved {len(sel)} selected atoms on top of the surface (clearance={float(clearance)} Å).",
                summary={"shift": shift.tolist(), "selected": sel, "surface": surf},
            ).to_dict()

        @tool
        def add_atom(symbol: str, x: float, y: float, z: float) -> Dict[str, Any]:
            """Add a single atom at (x, y, z) in Angstrom."""

            viewer._widget.send_js_task(
                {
                    "name": "avr.addAtom",
                    "kwargs": {
                        "element": symbol,
                        "position": {"x": float(x), "y": float(y), "z": float(z)},
                    },
                }
            )
            viewer._widget.send_js_task({"name": "avr.drawModels"})

            return WeasToolResult(f"Added {symbol} at ({x}, {y}, {z}).").to_dict()

        @tool
        def delete_atoms(indices: Optional[List[int]] = None) -> Dict[str, Any]:
            """Delete atoms by indices; if omitted, deletes the current selection."""

            _, _, _, _, use = _get_current_atoms_and_use_indices(viewer, indices)
            if not use:
                return WeasToolResult(
                    "Nothing to delete (no indices and empty selection)."
                ).to_dict()

            viewer._widget.send_js_task(
                {"name": "avr.deleteSelectedAtoms", "kwargs": {"indices": use}}
            )
            return WeasToolResult(f"Deleted {len(use)} atoms.").to_dict()

        @tool
        def replace_atoms(
            symbol: str, indices: Optional[List[int]] = None
        ) -> Dict[str, Any]:
            """Replace atoms (change element) by indices; if omitted, uses the current selection."""

            _, _, _, _, use = _get_current_atoms_and_use_indices(viewer, indices)
            if not use:
                return WeasToolResult(
                    "Nothing to replace (no indices and empty selection)."
                ).to_dict()
            viewer._widget.send_js_task(
                {
                    "name": "avr.replaceSelectedAtoms",
                    "kwargs": {"element": symbol, "indices": use},
                }
            )
            return WeasToolResult(
                f"Replaced {len(use)} atoms with '{symbol}'."
            ).to_dict()

        @tool
        def translate(
            vector: List[float], indices: Optional[List[int]] = None
        ) -> Dict[str, Any]:
            """Translate atoms by (dx, dy, dz) in Angstrom; if indices omitted, uses selection."""

            _, _, _, _, use = _get_current_atoms_and_use_indices(viewer, indices)
            if not use:
                return WeasToolResult(
                    "Nothing to translate (no indices and empty selection)."
                ).to_dict()
            if len(vector) != 3:
                raise ValueError("vector must be a length-3 list like [dx, dy, dz].")
            viewer._widget.send_js_task(
                {
                    "name": "avr.translateSelectedAtoms",
                    "kwargs": {"translateVector": vector, "indices": use},
                }
            )
            viewer._widget.send_js_task({"name": "avr.drawModels"})
            return WeasToolResult(
                f"Translated {len(use)} atoms by ({vector[0]}, {vector[1]}, {vector[2]})."
            ).to_dict()

        @tool
        def rotate(
            axis: List[float],
            angle_degrees: float,
            indices: Optional[List[int]] = None,
            about: str = "com",
        ) -> Dict[str, Any]:
            """Rotate atoms around an axis by angle degrees; indices omitted => selection; about='com' or 'origin'."""

            _, _, _, _, use = _get_current_atoms_and_use_indices(viewer, indices)
            if not use:
                return WeasToolResult(
                    "Nothing to rotate (no indices and empty selection)."
                ).to_dict()
            if len(axis) != 3:
                raise ValueError("axis must be a length-3 list like [ax, ay, az].")
            if about not in ("com", "origin"):
                raise ValueError("about must be 'com' or 'origin'.")

            axis_obj = {"x": float(axis[0]), "y": float(axis[1]), "z": float(axis[2])}
            args: List[Any] = [axis_obj, float(angle_degrees), use]
            if about == "origin":
                args.append({"x": 0.0, "y": 0.0, "z": 0.0})
            viewer._widget.send_js_task(
                {"name": "avr.rotateSelectedAtoms", "args": args}
            )
            return WeasToolResult(
                f"Rotated {len(use)} atoms by {angle_degrees} degrees."
            ).to_dict()

        @tool
        def undo() -> Dict[str, Any]:
            """Undo the last operation (equivalent to clicking the GUI undo)."""
            ops = getattr(viewer, "ops", None)
            if ops is None:
                raise AttributeError(
                    "Viewer does not have .ops; expected a WeasWidget."
                )
            ops.undo()
            return WeasToolResult("Undid last operation.").to_dict()

        @tool
        def redo() -> Dict[str, Any]:
            """Redo the last undone operation (equivalent to clicking the GUI redo)."""
            ops = getattr(viewer, "ops", None)
            if ops is None:
                raise AttributeError(
                    "Viewer does not have .ops; expected a WeasWidget."
                )
            ops.redo()
            return WeasToolResult("Redid last operation.").to_dict()

        @tool
        def select_all_atoms() -> Dict[str, Any]:
            """Select all atoms using the operation system (undoable)."""
            ops = getattr(viewer, "ops", None)
            if ops is None:
                raise AttributeError(
                    "Viewer does not have .ops; expected a WeasWidget."
                )
            ops.selection.select_all()
            return WeasToolResult("Selected all atoms.").to_dict()

        @tool
        def invert_selection() -> Dict[str, Any]:
            """Invert the atom selection (undoable)."""
            ops = getattr(viewer, "ops", None)
            if ops is None:
                raise AttributeError(
                    "Viewer does not have .ops; expected a WeasWidget."
                )
            ops.selection.invert_selection()
            return WeasToolResult("Inverted selection.").to_dict()

        @tool
        def select_atoms_inside_selected_objects() -> Dict[str, Any]:
            """
            Select atoms inside currently selected objects/meshes (undoable).

            This corresponds to the "Select inside" operation (docs: operation.rst).
            """
            ops = getattr(viewer, "ops", None)
            if ops is None:
                raise AttributeError(
                    "Viewer does not have .ops; expected a WeasWidget."
                )
            ops.selection.inside_selection()
            return WeasToolResult("Selected atoms inside selected objects.").to_dict()

        @tool
        def op_translate(
            vector: List[float], indices: Optional[List[int]] = None
        ) -> Dict[str, Any]:
            """Translate selection via the operation system (undoable)."""
            if len(vector) != 3:
                raise ValueError("vector must be a length-3 list like [dx, dy, dz].")
            if indices is not None:
                atoms, *_ = _get_current_ase_atoms(viewer)
                _normalize_indices(indices, selected=[], n_atoms=len(atoms))
                viewer.avr.selected_atoms_indices = list(indices)
            ops = getattr(viewer, "ops", None)
            if ops is None:
                raise AttributeError(
                    "Viewer does not have .ops; expected a WeasWidget."
                )
            ops.transform.translate(vector=list(vector))
            return WeasToolResult("Translated selection (undoable).").to_dict()

        @tool
        def op_rotate(
            axis: List[float],
            angle_degrees: float,
            indices: Optional[List[int]] = None,
        ) -> Dict[str, Any]:
            """Rotate selection via the operation system (undoable)."""
            if len(axis) != 3:
                raise ValueError("axis must be a length-3 list like [ax, ay, az].")
            if indices is not None:
                atoms, *_ = _get_current_ase_atoms(viewer)
                _normalize_indices(indices, selected=[], n_atoms=len(atoms))
                viewer.avr.selected_atoms_indices = list(indices)
            ops = getattr(viewer, "ops", None)
            if ops is None:
                raise AttributeError(
                    "Viewer does not have .ops; expected a WeasWidget."
                )
            ops.transform.rotate(axis=list(axis), angle=float(angle_degrees))
            return WeasToolResult("Rotated selection (undoable).").to_dict()

        @tool
        def op_scale(scale: List[float]) -> Dict[str, Any]:
            """Scale selected objects via the operation system (undoable)."""
            if len(scale) != 3:
                raise ValueError("scale must be a length-3 list like [sx, sy, sz].")
            ops = getattr(viewer, "ops", None)
            if ops is None:
                raise AttributeError(
                    "Viewer does not have .ops; expected a WeasWidget."
                )
            ops.transform.scale(scale=list(scale))
            return WeasToolResult("Scaled selection (undoable).").to_dict()

        @tool
        def delete_selected_objects() -> Dict[str, Any]:
            """Delete selected objects/meshes via the operation system (undoable)."""
            ops = getattr(viewer, "ops", None)
            if ops is None:
                raise AttributeError(
                    "Viewer does not have .ops; expected a WeasWidget."
                )
            ops.object.delete()
            return WeasToolResult("Deleted selected objects.").to_dict()

        @tool
        def duplicate_selected_objects() -> Dict[str, Any]:
            """Duplicate selected objects/meshes via the operation system (undoable)."""
            ops = getattr(viewer, "ops", None)
            if ops is None:
                raise AttributeError(
                    "Viewer does not have .ops; expected a WeasWidget."
                )
            ops.object.copy()
            return WeasToolResult("Duplicated selected objects.").to_dict()

        @tool
        def add_mesh_primitive(
            kind: Literal[
                "cube",
                "plane",
                "cylinder",
                "icosahedron",
                "cone",
                "sphere",
                "torus",
                "arrow",
            ],
            params: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """Add a mesh primitive via the operation system (undoable)."""
            ops = getattr(viewer, "ops", None)
            if ops is None:
                raise AttributeError(
                    "Viewer does not have .ops; expected a WeasWidget."
                )
            params = {} if params is None else dict(params)
            meth = getattr(ops.mesh, f"add_{kind}", None)
            if meth is None:
                raise ValueError(f"Unsupported mesh primitive kind: {kind!r}.")
            meth(**_to_jsonable(params))
            return WeasToolResult(f"Added mesh primitive '{kind}'.").to_dict()

        @tool
        def set_instanced_mesh_primitives(
            settings: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            """Set instanced mesh primitives (viewer.imp.settings)."""
            imp = getattr(viewer, "imp", None)
            if imp is None:
                raise AttributeError(
                    "Viewer does not have .imp; expected a WeasWidget."
                )
            imp.settings = _to_jsonable(settings)
            return WeasToolResult(
                f"Set {len(settings)} instanced mesh primitive groups."
            ).to_dict()

        @tool
        def clear_instanced_mesh_primitives() -> Dict[str, Any]:
            """Clear instanced mesh primitives."""
            imp = getattr(viewer, "imp", None)
            if imp is None:
                raise AttributeError(
                    "Viewer does not have .imp; expected a WeasWidget."
                )
            imp.settings = []
            return WeasToolResult("Cleared instanced mesh primitives.").to_dict()

        @tool
        def set_any_mesh(settings: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Set custom meshes (viewer.any_mesh.settings)."""
            any_mesh = getattr(viewer, "any_mesh", None)
            if any_mesh is None:
                raise AttributeError(
                    "Viewer does not have .any_mesh; expected a WeasWidget."
                )
            any_mesh.settings = _to_jsonable(settings)
            return WeasToolResult(f"Set {len(settings)} custom meshes.").to_dict()

        @tool
        def clear_any_mesh() -> Dict[str, Any]:
            """Clear custom meshes."""
            any_mesh = getattr(viewer, "any_mesh", None)
            if any_mesh is None:
                raise AttributeError(
                    "Viewer does not have .any_mesh; expected a WeasWidget."
                )
            any_mesh.settings = []
            return WeasToolResult("Cleared custom meshes.").to_dict()

        @tool
        def get_camera(
            key: Optional[Literal["zoom", "position", "look_at", "setting"]] = None
        ) -> Dict[str, Any]:
            """Get camera settings (zoom/position/look_at/setting)."""
            cam = getattr(viewer, "camera", None)
            if cam is None:
                raise AttributeError(
                    "Viewer does not have .camera; expected a WeasWidget."
                )
            data = {
                "zoom": getattr(cam, "zoom", None),
                "position": getattr(cam, "position", None),
                "look_at": getattr(cam, "look_at", None),
                "setting": getattr(cam, "setting", None),
            }
            if key is None:
                return WeasToolResult("OK", summary=_to_jsonable(data)).to_dict()
            return WeasToolResult(
                "OK", summary={key: _to_jsonable(data.get(key))}
            ).to_dict()

        @tool
        def set_camera(
            key: Literal["zoom", "position", "look_at", "setting"], value: Any
        ) -> Dict[str, Any]:
            """Set camera settings (zoom/position/look_at/setting)."""
            cam = getattr(viewer, "camera", None)
            if cam is None:
                raise AttributeError(
                    "Viewer does not have .camera; expected a WeasWidget."
                )
            if key == "zoom":
                cam.zoom = float(value)
            elif key in {"position", "look_at"}:
                if not isinstance(value, (list, tuple)) or len(value) != 3:
                    raise ValueError(f"{key} must be a length-3 list like [x,y,z].")
                setattr(cam, key, [float(value[0]), float(value[1]), float(value[2])])
            elif key == "setting":
                if not isinstance(value, dict):
                    raise ValueError(
                        "setting must be a dict, e.g. {'direction':[0,5,1],'distance':50,'zoom':2}."
                    )
                cam.setting = _to_jsonable(value)
            else:
                raise ValueError(f"Unknown camera key: {key!r}.")
            return WeasToolResult(f"Set camera.{key}.").to_dict()

        @tool
        def set_current_frame(frame: int) -> Dict[str, Any]:
            """Set the current frame (for trajectories)."""
            atoms_or_traj = viewer.to_ase()
            if not isinstance(atoms_or_traj, list):
                return WeasToolResult(
                    "Not a trajectory; current_frame unchanged."
                ).to_dict()
            n = len(atoms_or_traj)
            f = max(0, min(int(frame), n - 1))
            viewer.avr.current_frame = f
            return WeasToolResult(
                f"Set current frame to {f}.", summary={"n_frames": n}
            ).to_dict()

        @tool
        def get_animation_info() -> Dict[str, Any]:
            """Get trajectory/animation info (n_frames/current_frame)."""
            atoms_or_traj = viewer.to_ase()
            if not isinstance(atoms_or_traj, list):
                return WeasToolResult("OK", summary={"is_trajectory": False}).to_dict()
            return WeasToolResult(
                "OK",
                summary={
                    "is_trajectory": True,
                    "n_frames": len(atoms_or_traj),
                    "current_frame": int(getattr(viewer.avr, "current_frame", 0)),
                },
            ).to_dict()

        @tool
        def measure(indices: Optional[List[int]] = None) -> Dict[str, Any]:
            """
            Add a measurement overlay for 0-4 atoms.

            If indices is omitted, uses the current selection. Passing [] clears measurements.
            """
            atoms, *_ = _get_current_ase_atoms(viewer)
            selected = getattr(viewer.avr, "selected_atoms_indices", []) or []
            use = (
                _normalize_indices(indices, selected=selected, n_atoms=len(atoms))
                if indices is not None
                else list(selected)
            )
            if indices is None and not use:
                use = []
            viewer._widget.send_js_task(
                {"name": "avr.Measurement.measure", "args": [use]}
            )
            return WeasToolResult(
                "Measurement updated.", summary={"indices": use}
            ).to_dict()

        @tool
        def clear_measurements() -> Dict[str, Any]:
            """Clear all measurement overlays."""
            viewer._widget.send_js_task({"name": "avr.Measurement.reset", "kwargs": {}})
            return WeasToolResult("Cleared measurements.").to_dict()

        @tool
        def set_volumetric_data(values: Any) -> Dict[str, Any]:
            """
            Set volumetric data (shared by isosurface and volume slice).

            `values` should be a 3D array-like (z/y/x or x/y/z depending on your cube reader).
            """
            viewer.avr.iso.volumetric_data = {"values": _to_jsonable(values)}
            return WeasToolResult("Set volumetric data.").to_dict()

        @tool
        def set_isosurface_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
            """Set isosurface settings (viewer.avr.iso.settings)."""
            viewer.avr.iso.settings = _to_jsonable(settings)
            return WeasToolResult("Set isosurface settings.").to_dict()

        @tool
        def set_volume_slice_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
            """Set volume slice settings (viewer.avr.volume_slice.settings)."""
            viewer.avr.volume_slice.settings = _to_jsonable(settings)
            return WeasToolResult("Set volume slice settings.").to_dict()

        @tool
        def set_vector_field_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
            """Set vector field settings (viewer.avr.vf.settings)."""
            viewer.avr.vf.settings = _to_jsonable(settings)
            return WeasToolResult("Set vector field settings.").to_dict()

        @tool
        def set_show_vector_field(show: bool) -> Dict[str, Any]:
            """Toggle vector field visibility."""
            viewer._widget.showVectorField = bool(show)
            return WeasToolResult(f"Set show_vector_field to {bool(show)}.").to_dict()

        @tool
        def set_phonon_setting(setting: Dict[str, Any]) -> Dict[str, Any]:
            """Set phonon visualization settings (viewer.avr.phonon_setting)."""
            viewer.avr.phonon_setting = _to_jsonable(setting)
            return WeasToolResult("Set phonon setting.").to_dict()

        @tool
        def get_bond_pair(key: str) -> Dict[str, Any]:
            """Get a bond pair setting by key (e.g., 'Ti-O')."""
            data = dict(getattr(viewer.avr.bond, "settings", {}))
            return WeasToolResult(
                "OK", summary={key: _to_jsonable(data.get(key))}
            ).to_dict()

        @tool
        def update_bond_pair(key: str, updates: Dict[str, Any]) -> Dict[str, Any]:
            """Update a bond pair setting (e.g., set max/color)."""
            if not isinstance(updates, dict):
                raise ValueError("updates must be a dict.")
            settings = viewer.avr.bond.settings
            if key not in settings:
                raise KeyError(f"Bond pair {key!r} not found.")
            cur = settings[key]
            if isinstance(cur, dict):
                cur.update(_to_jsonable(updates))
            else:
                settings[key] = _to_jsonable(updates)
            return WeasToolResult(f"Updated bond pair {key}.").to_dict()

        @tool
        def delete_bond_pair(key: str) -> Dict[str, Any]:
            """Delete a bond pair setting (e.g., 'Ti-Ca')."""
            settings = viewer.avr.bond.settings
            if key in settings:
                del settings[key]
                return WeasToolResult(f"Deleted bond pair {key}.").to_dict()
            return WeasToolResult(
                f"Bond pair {key} not found; nothing deleted."
            ).to_dict()

        @tool
        def add_bond_pair(
            species1: str,
            species2: Optional[str] = None,
            min: float = 0.0,
            max: Optional[float] = None,
            color1: Optional[Any] = None,
            color2: Optional[Any] = None,
            type: int = 0,
        ) -> Dict[str, Any]:
            """Add a bond pair (wrapper over viewer.avr.bond.add_bond_pair)."""
            viewer.avr.bond.add_bond_pair(
                species1,
                species2,
                min=float(min),
                max=float(max) if max is not None else None,
                color1=_to_jsonable(color1) if color1 is not None else None,
                color2=_to_jsonable(color2) if color2 is not None else None,
                type=int(type),
            )
            key = f"{species1}-{species1 if species2 is None else species2}"
            return WeasToolResult(f"Added bond pair {key}.").to_dict()

        @tool
        def set_highlight_item(
            name: str,
            indices: List[int],
            type: Literal["sphere", "box", "cross"] = "sphere",
            color: Any = "yellow",
            scale: float = 1.1,
        ) -> Dict[str, Any]:
            """Add/update a highlight item (viewer.avr.highlight.settings[name])."""
            atoms, *_ = _get_current_ase_atoms(viewer)
            _normalize_indices(indices, selected=[], n_atoms=len(atoms))
            viewer.avr.highlight.settings[str(name)] = {
                "type": str(type),
                "indices": list(indices),
                "color": _to_jsonable(color),
                "scale": float(scale),
            }
            viewer.avr.draw()
            return WeasToolResult(
                f"Set highlight '{name}' for {len(indices)} atoms."
            ).to_dict()

        @tool
        def delete_highlight_item(name: str) -> Dict[str, Any]:
            """Delete a highlight item."""
            settings = viewer.avr.highlight.settings
            if name in settings:
                del settings[name]
                viewer.avr.draw()
                return WeasToolResult(f"Deleted highlight '{name}'.").to_dict()
            return WeasToolResult(
                f"Highlight '{name}' not found; nothing deleted."
            ).to_dict()

        @tool
        def add_lattice_plane_from_miller(
            name: str,
            miller: List[int],
            distance: float = 1.0,
            color: Optional[List[float]] = None,
            scale: float = 1.0,
            width: float = 0.1,
        ) -> Dict[str, Any]:
            """Add a lattice plane from Miller indices and rebuild planes."""
            if len(miller) != 3:
                raise ValueError("miller must be a length-3 list like [h,k,l].")
            kwargs: Dict[str, Any] = {
                "distance": float(distance),
                "scale": float(scale),
                "width": float(width),
            }
            if color is not None:
                if len(color) != 4:
                    raise ValueError("color must be RGBA like [r,g,b,a].")
                kwargs["color"] = [float(x) for x in color]
            viewer.avr.lp.add_plane_from_indices(
                str(name), [int(x) for x in miller], **kwargs
            )
            viewer.avr.lp.build_plane()
            return WeasToolResult(
                f"Added lattice plane '{name}' and rebuilt planes."
            ).to_dict()

        @tool
        def add_lattice_plane_from_selected_atoms(
            name: str,
            color: Optional[List[float]] = None,
            scale: float = 1.0,
            width: float = 0.1,
        ) -> Dict[str, Any]:
            """Add a lattice plane from 3 selected atoms and rebuild planes."""
            kwargs: Dict[str, Any] = {"scale": float(scale), "width": float(width)}
            if color is not None:
                if len(color) != 4:
                    raise ValueError("color must be RGBA like [r,g,b,a].")
                kwargs["color"] = [float(x) for x in color]
            viewer.avr.lp.add_plane_from_selected_atoms(str(name), **kwargs)
            viewer.avr.lp.build_plane()
            return WeasToolResult(
                f"Added lattice plane '{name}' from selection and rebuilt planes."
            ).to_dict()

        return [
            list_style_options,
            get_style,
            set_style,
            get_selected_atoms,
            get_structure_summary,
            load_molecule,
            load_bulk,
            load_fcc_surface,
            load_surface,
            append_molecule,
            place_selected_on_top,
            select_atoms,
            add_atom,
            delete_atoms,
            replace_atoms,
            translate,
            rotate,
            undo,
            redo,
            select_all_atoms,
            invert_selection,
            select_atoms_inside_selected_objects,
            op_translate,
            op_rotate,
            op_scale,
            delete_selected_objects,
            duplicate_selected_objects,
            add_mesh_primitive,
            set_instanced_mesh_primitives,
            clear_instanced_mesh_primitives,
            set_any_mesh,
            clear_any_mesh,
            get_camera,
            set_camera,
            set_current_frame,
            get_animation_info,
            measure,
            clear_measurements,
            set_volumetric_data,
            set_isosurface_settings,
            set_volume_slice_settings,
            set_vector_field_settings,
            set_show_vector_field,
            set_phonon_setting,
            get_bond_pair,
            update_bond_pair,
            delete_bond_pair,
            add_bond_pair,
            set_highlight_item,
            delete_highlight_item,
            add_lattice_plane_from_miller,
            add_lattice_plane_from_selected_atoms,
        ]


def create_weas_tools(
    viewer: Any,
    *,
    extra_tools: Optional[Sequence[Any]] = None,
    load_entry_points: bool = True,
    entry_point_group: str = "weas_widget.tools",
):
    """Helper returning `WeasToolkit(...).tools` with optional extensions."""
    return WeasToolkit(
        viewer,
        extra_tools=extra_tools,
        load_entry_points=load_entry_points,
        entry_point_group=entry_point_group,
    ).tools


def _is_tool_instance(obj: Any) -> bool:
    name = getattr(obj, "name", None)
    if not isinstance(name, str) or not name:
        return False
    return any(
        callable(getattr(obj, attr, None))
        for attr in ("invoke", "ainvoke", "run", "arun")
    )


def _call_maybe_with_viewer(factory: Callable[..., Any], viewer: Any) -> Any:
    try:
        return factory(viewer)
    except TypeError as e:
        try:
            sig = inspect.signature(factory)
        except Exception:
            raise
        if len(sig.parameters) == 0:
            return factory()
        raise e


def _flatten(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[Any] = []
        for x in value:
            out.extend(list(_flatten(x)))
        return out
    return [value]


def _resolve_tool_specs(specs: Sequence[Any], viewer: Any) -> Iterable[Any]:
    for spec in specs:
        for item in _flatten(spec):
            if item is None:
                continue
            if _is_tool_instance(item):
                yield item
                continue
            if callable(item):
                produced = _call_maybe_with_viewer(item, viewer)
                for t in _flatten(produced):
                    if not _is_tool_instance(t):
                        raise TypeError(f"Tool factory produced a non-tool: {t!r}")
                    yield t
                continue
            raise TypeError(f"Unsupported tool spec: {item!r}")


def _load_tool_entry_points(group: str) -> List[Any]:
    try:
        from importlib import metadata as importlib_metadata
    except Exception:
        import importlib_metadata

    eps = importlib_metadata.entry_points()
    if hasattr(eps, "select"):  # py>=3.10
        selected = list(eps.select(group=group))
    else:
        selected = list(eps.get(group, []))

    out: List[Any] = []
    for ep in selected:
        try:
            out.append(ep.load())
        except Exception as e:
            warnings.warn(
                f"Failed to import tool entry point {ep!r}: {e}", RuntimeWarning
            )
    return out
