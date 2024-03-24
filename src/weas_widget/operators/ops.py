from .mesh import MeshOperation
from .object import ObjectOperation
from .atoms import AtomsOperation
from .selection import SelectionOperation
from .transform import TransformOperation


class Operators:
    def __init__(self, weas_viewer):
        self.weas_viewer = weas_viewer
        self.mesh = MeshOperation(weas_viewer)
        self.object = ObjectOperation(weas_viewer)
        self.atoms = AtomsOperation(weas_viewer)
        self.selection = SelectionOperation(weas_viewer)
        self.transform = TransformOperation(weas_viewer)

    def undo(self):
        self.weas_viewer.send_js_task({"name": "ops.undo"})

    def redo(self):
        self.weas_viewer.send_js_task({"name": "ops.redo"})
