from ..base_class import WidgetWrapper, ChangeTrackingDict


class HighlightManager(WidgetWrapper):

    catalog = "highlight"

    _attribute_map = {}

    _extra_allowed_attrs = ["_settings", "settings"]

    def __init__(self, _widget):
        super().__init__(_widget)
        self._settings = ChangeTrackingDict(
            widget=self._widget, key="highlightSettings"
        )

    @property
    def settings(self):
        return self._settings

    @settings.setter
    def settings(self, value):
        self._settings = ChangeTrackingDict(
            value, widget=self._widget, key="highlightSettings"
        )

    def update_atoms(self):
        self.settings = self.get_default_settings()

    def get_default_settings(self):
        settings = {
            "selection": {
                "indices": [],
                "scale": 1.1,
                "color": "yellow",
                "type": "sphere",
            },
            "fixed": {"indices": [], "scale": 1.1, "color": "black", "type": "cross"},
        }
        return settings
