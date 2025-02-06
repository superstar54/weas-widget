from ..base_class import WidgetWrapper


class AnyMesh(WidgetWrapper):

    catalog = "any_mesh"

    _attribute_map = {
        "settings": "anyMesh",
    }

    _extra_allowed_attrs = []

    def __init__(self, _widget):
        super().__init__(_widget)
