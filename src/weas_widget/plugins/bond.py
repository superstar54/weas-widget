from ..base_class import WidgetWrapper, ChangeTrackingDict
from weas_widget.data import default_bond_pairs


class BondManager(WidgetWrapper):

    catalog = "bond"

    _attribute_map = {}

    _extra_allowed_attrs = ["_settings", "settings"]

    def __init__(self, _widget):
        super().__init__(_widget)
        self._settings = ChangeTrackingDict(widget=self._widget, key="bondSettings")

    @property
    def settings(self):
        return self._settings

    @settings.setter
    def settings(self, value):
        self._settings = ChangeTrackingDict(
            value, widget=self._widget, key="bondSettings"
        )

    def update_atoms(self):
        self.settings = self.get_default_settings()

    def get_default_settings(self):
        settings = {}
        species_dict = self._widget.speciesSettings
        for species1, data1 in species_dict.items():
            for species2, data2 in species_dict.items():
                if (data1["element"], data2["element"]) not in default_bond_pairs:
                    continue
                settings[f"[{species1}, {species2}]"] = {
                    "species1": species1,
                    "species2": species2,
                    "color1": data1["color"],
                    "color2": data2["color"],
                    "min": 0,
                    "max": (data1["radius"] + data2["radius"]) * 1.1,
                }
        return settings
