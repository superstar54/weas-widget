from ..base_class import WidgetWrapper


class Isosurface(WidgetWrapper):

    catalog = "isosurface"

    _attribute_map = {
        "volumetric_data": "volumetricData",
        "settings": "isoSettings",
    }

    _extra_allowed_attrs = []

    def __init__(self, _widget):
        super().__init__(_widget)
