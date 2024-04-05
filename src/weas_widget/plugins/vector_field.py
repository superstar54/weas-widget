from ..base_class import WidgetWrapper


class VectorField(WidgetWrapper):

    catalog = "vector_field"

    _attribute_map = {
        "settings": "vectorField",
        "show": "showVectorField",
    }

    _extra_allowed_attrs = []

    def __init__(self, _widget):
        super().__init__(_widget)
