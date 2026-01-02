from ..base_class import WidgetWrapper, ChangeTrackingDict

DEFAULT_CELL_SETTINGS = {
    "showCell": True,  # Show unit cell
    "showAxes": True,  # Show cell axes
    "cellColor": 0x000000,  # Default cell line color (black)
    "cellLineWidth": 2,  # Default line width
    "axisColors": {"a": 0xFF0000, "b": 0x00FF00, "c": 0x0000FF},  # RGB colors for axes
    "axisRadius": 0.15,  # Default axis cylinder radius
    "axisConeHeight": 0.8,  # Cone height for axis arrows
    "axisConeRadius": 0.3,  # Cone radius for axis arrows
    "axisSphereRadius": 0.3,  # Sphere radius at the cell origin
}


class CellManager(WidgetWrapper):

    catalog = "cell"

    _attribute_map = {}

    _extra_allowed_attrs = ["_settings", "settings"]

    def __init__(self, _widget):
        super().__init__(_widget)
        self._settings = ChangeTrackingDict(
            widget=self._widget, key="cellSettings", **DEFAULT_CELL_SETTINGS
        )

    @property
    def settings(self):
        return self._settings

    @settings.setter
    def settings(self, value):
        self._settings = ChangeTrackingDict(value, widget=self._widget, key="cell")
