"""
AddCubeOperation
AddPlaneOperation
AddCylinderOperation
AddIcosahedronOperation
AddConeOperation
AddSphereOperation
AddTorusOperation
AddArrowOperation
"""


class MeshOperation:
    def __init__(self, base_widget):
        self.base_widget = base_widget

    def add_cube(self, **kwargs):
        self.base_widget.send_js_task(
            {"name": "ops.mesh.AddCubeOperation", "kwargs": kwargs}
        )

    def add_plane(self, **kwargs):
        self.base_widget.send_js_task(
            {"name": "ops.mesh.AddPlaneOperation", "kwargs": kwargs}
        )

    def add_cylinder(self, **kwargs):
        self.base_widget.send_js_task(
            {"name": "ops.mesh.AddCylinderOperation", "kwargs": kwargs}
        )

    def add_icosahedron(self, **kwargs):
        self.base_widget.send_js_task(
            {"name": "ops.mesh.AddIcosahedronOperation", "kwargs": kwargs}
        )

    def add_cone(self, **kwargs):
        self.base_widget.send_js_task(
            {"name": "ops.mesh.AddConeOperation", "kwargs": kwargs}
        )

    def add_sphere(self, **kwargs):
        self.base_widget.send_js_task(
            {"name": "ops.mesh.AddSphereOperation", "kwargs": kwargs}
        )

    def add_torus(self, **kwargs):
        self.base_widget.send_js_task(
            {"name": "ops.mesh.AddTorusOperation", "kwargs": kwargs}
        )

    def add_arrow(self, **kwargs):
        self.base_widget.send_js_task(
            {"name": "ops.mesh.AddArrowOperation", "kwargs": kwargs}
        )
