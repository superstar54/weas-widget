from ..base_class import WidgetWrapper


class FermiSurface(WidgetWrapper):

    catalog = "fermiSurface"

    _attribute_map = {
        "fermi_data": "fermiData",
        "settings": "fermiSettings",
    }

    _extra_allowed_attrs = []

    def __init__(self, _widget):
        super().__init__(_widget)
