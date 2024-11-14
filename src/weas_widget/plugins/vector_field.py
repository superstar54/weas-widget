from ..base_class import WidgetWrapper, ChangeTrackingDict


class VectorField(WidgetWrapper):

    catalog = "vector_field"

    _attribute_map = {
        "show": "showVectorField",
    }

    _extra_allowed_attrs = ["_settings", "settings"]

    def __init__(self, _widget):
        super().__init__(_widget)

    @property
    def settings(self):
        return self._settings

    @settings.setter
    def settings(self, value):
        self._settings = ChangeTrackingDict(
            value, widget=self._widget, key="vectorField"
        )

    def update_atoms(self):
        self.settings = self.set_moment()

    def set_moment(self):
        """Set magnetic moment vector field."""
        atoms = self._widget.atoms
        if isinstance(atoms, list):
            atoms = atoms[0]
        settings = {}
        # separate spin up and down, add two vector fields
        if "moment" in atoms["attributes"]["atom"]:
            moment = atoms["attributes"]["atom"]["moment"]
            spin_up = [i for i, m in enumerate(moment) if m > 0]
            spin_down = [i for i, m in enumerate(moment) if m < 0]
            settings = {
                "up": {
                    "origins": [atoms["positions"][i] for i in spin_up],
                    "vectors": [[0, 0, moment[i]] for i in spin_up],
                    "color": "blue",
                },
                "down": {
                    "origins": [atoms["positions"][i] for i in spin_down],
                    "vectors": [[0, 0, moment[i]] for i in spin_down],
                    "color": "red",
                },
            }
        return settings
