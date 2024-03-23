from .base_class import WidgetWrapper


class Camera(WidgetWrapper):

    catalog = "camera"

    _attribute_map = {
        "zoom": "cameraZoom",
        "position": "cameraPosition",
        "look_at": "cameraLookAt",
        "setting": "cameraSetting",
    }
    _extra_allowed_attrs = []

    def __init__(self, _widget):
        super().__init__(_widget)
