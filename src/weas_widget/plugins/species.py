from ..base_class import WidgetWrapper
from ase.data import atomic_numbers
from weas_widget.data import color_data, radii_data


class SpeciesManager(WidgetWrapper):

    catalog = "species"

    _attribute_map = {
        "settings": "speciesSettings",
    }

    _extra_allowed_attrs = []

    def __init__(self, _widget):
        super().__init__(_widget)

    def update_atoms(self):
        self.settings = self.get_default_settings()

    def get_default_settings(self):
        settings = {}
        species_dict = self._widget.atoms.get("species", {})
        for species in species_dict:
            element = species_dict[species]
            number = atomic_numbers[element]
            color = color_data[self._widget.colorType][number]
            settings[species] = {
                "element": element,
                "symbol": species,
                "color": color,
                "radius": radii_data[self._widget.radiusType.upper()][number],
            }
        return settings
