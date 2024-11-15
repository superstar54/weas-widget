from ..base_class import WidgetWrapper


class VolumeSlice(WidgetWrapper):

    catalog = "VolumeSlice"

    _attribute_map = {
        "volumetric_data": "volumetricData",
        "settings": "sliceSettings",
    }

    _extra_allowed_attrs = []

    def __init__(self, _widget):
        super().__init__(_widget)
