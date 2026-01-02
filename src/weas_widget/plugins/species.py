from ..base_class import WidgetWrapper, ChangeTrackingDict
from ase.data import atomic_numbers
from weas_widget.data import color_data, radii_data


class SpeciesManager(WidgetWrapper):

    catalog = "species"

    _attribute_map = {}

    _extra_allowed_attrs = ["_settings", "settings"]

    def __init__(self, _widget):
        super().__init__(_widget)
        self._settings = ChangeTrackingDict(widget=self._widget, key="speciesSettings")

    @property
    def settings(self):
        return self._settings

    @settings.setter
    def settings(self, value):
        self._settings = ChangeTrackingDict(
            value, widget=self._widget, key="speciesSettings"
        )

    def update_atoms(self):
        self.settings = self.get_default_settings()

    def get_default_settings(self):
        settings = {}
        atoms = self._widget.atoms
        if isinstance(atoms, list):
            atoms = atoms[0]
        species_dict = atoms.get("species", {})
        for species in species_dict:
            element = species_dict[species]
            number = atomic_numbers[element]
            color = list(color_data[self._widget.colorType][number])
            settings[species] = {
                "element": element,
                "symbol": species,
                "color": color,
                "radius": radii_data[self._widget.radiusType.upper()][number],
            }
        return settings
