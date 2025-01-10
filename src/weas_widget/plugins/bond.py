from __future__ import annotations
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
        for specie1, data1 in species_dict.items():
            for specie2, data2 in species_dict.items():
                if (data1["element"], data2["element"]) not in default_bond_pairs:
                    continue
                bond_type = default_bond_pairs[(data1["element"], data2["element"])][2]
                min = 0
                max = (data1["radius"] + data2["radius"]) * 1.1
                color1 = data1["color"]
                color2 = data2["color"]
                if bond_type == 1:
                    min = max + 0.4
                    max = min + 1
                    color1 = "#808080"
                    color2 = "#808080"
                    bond_line_type = 1
                else:
                    bond_line_type = 0

                settings[f"{specie1}-{specie2}"] = {
                    "specie1": specie1,
                    "specie2": specie2,
                    "color1": color1,
                    "color2": color2,
                    "min": min,
                    "max": max,
                    "type": bond_line_type,
                }
        return settings

    def add_bond_pair(
        self,
        species1: str,
        species2: str | None = None,
        min: float = 0.0,
        max: float = None,
        color1: list | None = None,
        color2: list | None = None,
        type: int = 0,
    ) -> None:
        """Add a bond between two species."""

        species2 = species1 if species2 is None else species2
        if max is None:
            max = (
                self._widget.speciesSettings[species1]["radius"]
                + self._widget.speciesSettings[species2]["radius"]
            ) * 1.1
        color1 = (
            self._widget.speciesSettings[species1]["color"]
            if color1 is None
            else color1
        )
        color2 = (
            self._widget.speciesSettings[species2]["color"]
            if color2 is None
            else color2
        )
        self.settings[f"{species1}-{species2}"] = {
            "specie1": species1,
            "specie2": species2,
            "color1": color1,
            "color2": color2,
            "min": min,
            "max": max,
            "type": type,
        }
