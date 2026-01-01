from typing import Any, Dict, List, Literal, Optional

from .tool_helpers import WeasToolResult, _to_jsonable


def build_operations_tools(viewer: Any):
    from langchain_core.tools import tool

    @tool
    def undo() -> Dict[str, Any]:
        """Undo the last operation (equivalent to clicking the GUI undo)."""
        ops = getattr(viewer, "ops", None)
        if ops is None:
            raise AttributeError("Viewer does not have .ops; expected a WeasWidget.")
        ops.undo()
        return WeasToolResult("Undid last operation.").to_dict()

    @tool
    def redo() -> Dict[str, Any]:
        """Redo the last undone operation (equivalent to clicking the GUI redo)."""
        ops = getattr(viewer, "ops", None)
        if ops is None:
            raise AttributeError("Viewer does not have .ops; expected a WeasWidget.")
        ops.redo()
        return WeasToolResult("Redid last operation.").to_dict()

    @tool
    def select_all_atoms() -> Dict[str, Any]:
        """Select all atoms using the operation system (undoable)."""
        ops = getattr(viewer, "ops", None)
        if ops is None:
            raise AttributeError("Viewer does not have .ops; expected a WeasWidget.")
        ops.selection.select_all()
        return WeasToolResult("Selected all atoms.").to_dict()

    @tool
    def invert_selection() -> Dict[str, Any]:
        """Invert the atom selection (undoable)."""
        ops = getattr(viewer, "ops", None)
        if ops is None:
            raise AttributeError("Viewer does not have .ops; expected a WeasWidget.")
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
            raise AttributeError("Viewer does not have .ops; expected a WeasWidget.")
        ops.selection.inside_selection()
        return WeasToolResult("Selected atoms inside selected objects.").to_dict()

    @tool
    def op_scale(scale: List[float]) -> Dict[str, Any]:
        """Scale selected objects via the operation system (undoable)."""
        if len(scale) != 3:
            raise ValueError("scale must be a length-3 list like [sx, sy, sz].")
        ops = getattr(viewer, "ops", None)
        if ops is None:
            raise AttributeError("Viewer does not have .ops; expected a WeasWidget.")
        ops.transform.scale(scale=list(scale))
        return WeasToolResult("Scaled selection (undoable).").to_dict()

    @tool
    def delete_selected_objects() -> Dict[str, Any]:
        """Delete selected objects/meshes via the operation system (undoable)."""
        ops = getattr(viewer, "ops", None)
        if ops is None:
            raise AttributeError("Viewer does not have .ops; expected a WeasWidget.")
        ops.object.delete()
        return WeasToolResult("Deleted selected objects.").to_dict()

    @tool
    def duplicate_selected_objects() -> Dict[str, Any]:
        """Duplicate selected objects/meshes via the operation system (undoable)."""
        ops = getattr(viewer, "ops", None)
        if ops is None:
            raise AttributeError("Viewer does not have .ops; expected a WeasWidget.")
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
            raise AttributeError("Viewer does not have .ops; expected a WeasWidget.")
        params = {} if params is None else dict(params)
        meth = getattr(ops.mesh, f"add_{kind}", None)
        if meth is None:
            raise ValueError(f"Unsupported mesh primitive kind: {kind!r}.")
        meth(**_to_jsonable(params))
        return WeasToolResult(f"Added mesh primitive '{kind}'.").to_dict()

    return [
        undo,
        redo,
        select_all_atoms,
        invert_selection,
        select_atoms_inside_selected_objects,
        op_scale,
        delete_selected_objects,
        duplicate_selected_objects,
        add_mesh_primitive,
    ]
