from typing import Any, Dict, List, Literal, Optional

from .tool_helpers import (
    WeasToolResult,
    _get_current_ase_atoms,
    _normalize_indices,
    _to_jsonable,
)


def build_visualization_tools(viewer: Any):
    from langchain_core.tools import tool

    @tool
    def set_instanced_mesh_primitives(settings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Set instanced mesh primitives (viewer.imp.settings)."""
        imp = getattr(viewer, "imp", None)
        if imp is None:
            raise AttributeError("Viewer does not have .imp; expected a WeasWidget.")
        imp.settings = _to_jsonable(settings)
        return WeasToolResult(
            f"Set {len(settings)} instanced mesh primitive groups."
        ).to_dict()

    @tool
    def clear_instanced_mesh_primitives() -> Dict[str, Any]:
        """Clear instanced mesh primitives."""
        imp = getattr(viewer, "imp", None)
        if imp is None:
            raise AttributeError("Viewer does not have .imp; expected a WeasWidget.")
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
            raise AttributeError("Viewer does not have .camera; expected a WeasWidget.")
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
            raise AttributeError("Viewer does not have .camera; expected a WeasWidget.")
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
        viewer._widget.send_js_task({"name": "avr.Measurement.measure", "args": [use]})
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
        return WeasToolResult(f"Bond pair {key} not found; nothing deleted.").to_dict()

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
