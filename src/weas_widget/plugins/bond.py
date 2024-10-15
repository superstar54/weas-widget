from ..base_class import WidgetWrapper
from ase.data.colors import cpk_colors
from ase.data import covalent_radii, atomic_numbers


class BondManager(WidgetWrapper):

    catalog = "bond"

    _attribute_map = {
        "settings": "bondSettings",
    }

    _extra_allowed_attrs = []

    def __init__(self, _widget):
        super().__init__(_widget)

    def update_atoms(self):
        self.settings = self.get_default_settings()

    def get_default_settings(self):
        settings = {}
        species_dict = self._widget.atoms.get("species", {})
        for species1 in species_dict:
            for species2 in species_dict:
                element1 = atomic_numbers[species_dict[species1]]
                element2 = atomic_numbers[species_dict[species2]]
                color1 = cpk_colors[element1]
                color2 = cpk_colors[element2]
                settings[f"[{species1}, {species2}]"] = {
                    "species1": species1,
                    "species2": species2,
                    "color1": color1,
                    "color2": color2,
                    "min": 0,
                    "max": (covalent_radii[element1] + covalent_radii[element2]) * 1.1,
                }
        return settings
