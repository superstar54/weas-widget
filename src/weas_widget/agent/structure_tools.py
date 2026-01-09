from typing import Any, Dict, List, Literal, Optional

import numpy as np

from .tool_helpers import (
    WeasToolResult,
    _cell_center_xy,
    _get_current_atoms_data,
    _get_current_ase_atoms,
    _normalize_indices,
    _set_current_ase_atoms,
)


def build_structure_tools(viewer: Any):
    from langchain_core.tools import tool

    def _sanitize_group(label: str) -> str:
        import re

        name = re.sub(r"[^A-Za-z0-9_]+", "_", str(label).strip().lower())
        return name.strip("_") or "group"

    def _with_group_name(prefix: str, label: str) -> str:
        return f"{prefix}_{_sanitize_group(label)}"

    def _merge_groups(
        existing: Optional[List[Any]], n_old: int, n_new: int, group_name: str
    ) -> List[List[str]]:
        groups: List[List[str]] = []
        existing = existing if isinstance(existing, list) else []
        for i in range(n_old):
            entry = existing[i] if i < len(existing) else []
            if isinstance(entry, list):
                groups.append([str(x) for x in entry])
            else:
                groups.append([])
        for _ in range(n_new):
            groups.append([group_name])
        return groups

    def _apply_groups(groups: List[List[str]]) -> None:
        viewer.avr.set_attribute("groups", groups, domain="atom")

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
    def read_atoms_from_file(file_path: str) -> Dict[str, Any]:
        """Read atoms from a file and load into the viewer.
        Supported formats depend on ASE installation, e.g., XYZ, CIF, POSCAR, etc.
        """
        from ase.io import read

        atoms = read(file_path)
        viewer.from_ase(atoms)
        viewer.avr.selected_atoms_indices = []
        return WeasToolResult(
            f"Loaded structure from '{file_path}' with {len(atoms)} atoms."
        ).to_dict()

    @tool
    def load_molecule(name: str) -> Dict[str, Any]:
        """Load an ASE molecule by name/formula (e.g., 'H2O', 'CO2', 'benzene') into the viewer."""
        from ase.build import molecule

        atoms = molecule(name)
        viewer.from_ase(atoms)
        _apply_groups([[_with_group_name("mol", name)] for _ in range(len(atoms))])
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
                raise ValueError("repeat must be a length-3 list like [nx, ny, nz].")
            atoms = atoms * tuple(int(x) for x in repeat)
        viewer.from_ase(atoms)
        _apply_groups([[_with_group_name("bulk", symbol)] for _ in range(len(atoms))])
        viewer.avr.selected_atoms_indices = []
        return WeasToolResult(
            f"Loaded bulk {symbol} ({crystalstructure}) with {len(atoms)} atoms."
        ).to_dict()

    @tool
    def load_fcc_surface(
        symbol: str = "Pt",
        miller: List[int] = [1, 1, 1],
        size: List[int] = [4, 4, 4],
        vacuum: float = 10.0,
        a: Optional[float] = None,
        orthogonal: bool = True,
        mode: Optional[Literal["override", "append"]] = None,
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
        mode
            "override" replaces the current structure, "append" adds the slab to it.
        """
        from ase.build import bulk, fcc100, fcc110, fcc111, surface

        atoms, is_traj, frame, traj = _get_current_ase_atoms(viewer)
        atoms_data, *_ = _get_current_atoms_data(viewer)
        existing_groups = (
            atoms_data.get("attributes", {}).get("atom", {}).get("groups")
            if isinstance(atoms_data, dict)
            else None
        )
        has_atoms = len(atoms) > 0
        if mode is None and has_atoms:
            return WeasToolResult(
                "Confirmation required to load surface.",
                summary={"existing_atoms": int(len(atoms))},
                confirmation={
                    "prompt": "Viewer already has atoms. Choose how to proceed.",
                    "options": ["override", "append"],
                    "note": "override replaces the current structure; append adds the slab.",
                },
            ).to_dict()
        if mode is not None and mode not in {"override", "append"}:
            raise ValueError("mode must be 'override' or 'append'.")

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

        if mode == "append" and has_atoms:
            orig_cell = np.asarray(atoms.get_cell().array, dtype=float)
            slab_cell = np.asarray(slab.get_cell().array, dtype=float)
            orig_has_cell = orig_cell.shape == (3, 3) and not np.allclose(
                orig_cell, 0.0
            )
            if orig_has_cell:
                slab.set_cell(orig_cell, scale_atoms=False)
                slab.set_pbc(atoms.get_pbc())
                center_orig = _cell_center_xy(atoms)
                center_slab = _cell_center_xy(slab)
                dxy = center_orig[:2] - center_slab[:2]
                if not np.allclose(dxy, 0.0):
                    slab.translate([float(dxy[0]), float(dxy[1]), 0.0])
            else:
                atoms.set_cell(slab_cell, scale_atoms=False)
                atoms.set_pbc(slab.get_pbc())
            atoms += slab
            _set_current_ase_atoms(
                viewer, atoms, is_trajectory=is_traj, frame=frame, trajectory=traj
            )
            group_name = _with_group_name("slab", symbol)
            groups = _merge_groups(
                existing_groups, len(atoms) - len(slab), len(slab), group_name
            )
            _apply_groups(groups)
        else:
            viewer.from_ase(slab)
            group_name = _with_group_name("slab", symbol)
            _apply_groups([[group_name] for _ in range(len(slab))])
        viewer.avr.selected_atoms_indices = []
        return WeasToolResult(
            f"Loaded {symbol}({h}{k}{l}) slab with {len(slab)} atoms.",
            summary={"symbol": symbol, "miller": [h, k, l], "size": [nx, ny, nlayers]},
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
        mode: Optional[Literal["override", "append"]] = None,
    ) -> Dict[str, Any]:
        """
        Build and load a general surface slab using ASE.

        This is the general counterpart to `load_fcc_surface` and supports any
        crystal structure accepted by `ase.build.bulk` and any Miller index.
        """
        from ase.build import bulk, surface

        atoms, is_traj, frame, traj = _get_current_ase_atoms(viewer)
        atoms_data, *_ = _get_current_atoms_data(viewer)
        existing_groups = (
            atoms_data.get("attributes", {}).get("atom", {}).get("groups")
            if isinstance(atoms_data, dict)
            else None
        )
        has_atoms = len(atoms) > 0
        if mode is None and has_atoms:
            return WeasToolResult(
                "Confirmation required to load surface.",
                summary={"existing_atoms": int(len(atoms))},
                confirmation={
                    "prompt": "Viewer already has atoms. Choose how to proceed.",
                    "options": ["override", "append"],
                    "note": "override replaces the current structure; append adds the slab.",
                },
            ).to_dict()
        if mode is not None and mode not in {"override", "append"}:
            raise ValueError("mode must be 'override' or 'append'.")

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

        if mode == "append" and has_atoms:
            atoms += slab
            _set_current_ase_atoms(
                viewer, atoms, is_trajectory=is_traj, frame=frame, trajectory=traj
            )
            group_name = _with_group_name("slab", symbol)
            groups = _merge_groups(
                existing_groups, len(atoms) - len(slab), len(slab), group_name
            )
            _apply_groups(groups)
        else:
            viewer.from_ase(slab)
            group_name = _with_group_name("slab", symbol)
            _apply_groups([[group_name] for _ in range(len(slab))])
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
        atoms_data, *_ = _get_current_atoms_data(viewer)
        existing_groups = (
            atoms_data.get("attributes", {}).get("atom", {}).get("groups")
            if isinstance(atoms_data, dict)
            else None
        )
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
        group_name = _with_group_name("mol", name)
        groups = _merge_groups(existing_groups, before_n, len(mol), group_name)
        _apply_groups(groups)

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
            raise ValueError("No selected atoms. Select the molecule/fragment first.")

        if surface_indices is None:
            sel_set = set(sel)
            surf = [i for i in range(len(atoms)) if i not in sel_set]
        else:
            surf = _normalize_indices(surface_indices, selected=[], n_atoms=len(atoms))
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
            raise ValueError("align_xy must be one of: cell_center, surface_com, none.")

        slab = atoms[surf].copy()
        slab.set_cell(atoms.get_cell())
        slab.set_pbc(atoms.get_pbc())
        ads = atoms[sel].copy()
        old_ads_pos = ads.get_positions().copy()
        mol_index = int(np.argmin(old_ads_pos[:, 2]))

        if align_xy == "none":
            position_xy = tuple(float(x) for x in old_ads_pos[mol_index, :2])
        else:
            position_xy = tuple(float(x) for x in (old_ads_pos[mol_index, :2] + dxy))

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

    return [
        get_structure_summary,
        read_atoms_from_file,
        load_molecule,
        load_bulk,
        load_fcc_surface,
        load_surface,
        append_molecule,
        place_selected_on_top,
    ]
