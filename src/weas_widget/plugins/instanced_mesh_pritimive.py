from ..base_class import WidgetWrapper


class InstancedMeshPrimitive(WidgetWrapper):

    catalog = "instanced_mesh_primitive"

    _attribute_map = {
        "settings": "instancedMeshPrimitive",
    }

    _extra_allowed_attrs = []

    def __init__(self, _widget):
        super().__init__(_widget)
