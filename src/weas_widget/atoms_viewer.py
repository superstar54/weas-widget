from .base_class import WidgetWrapper
from .plugins.vector_field import VectorField
from .plugins.isosurface import Isosurface
from .plugins.lattice_plane import LatticePlane


class AtomsViewer(WidgetWrapper):

    catalog = "viewer"

    _attribute_map = {
        "model_style": "modelStyle",
        "selected_atoms_indices": "selectedAtomsIndices",
        "boundary": "boundary",
        "color_type": "colorType",
        "color_by": "colorBy",
        "color_ramp": "colorRamp",
        "show_cell": "showCell",
        "show_bonded_atoms": "showBondedAtoms",
        "atom_label_type": "atomLabelType",
        "material_type": "materialType",
        "model_sticks": "modelSticks",
        "atom_scales": "atomScales",
        "model_polyhedras": "modelPolyhedras",
    }

    _extra_allowed_attrs = ["vf", "iso", "lp", "atoms"]

    def __init__(self, _widget):

        super().__init__(_widget)
        # Initialize plugins
        setattr(self, "vf", VectorField(_widget))
        setattr(self, "iso", Isosurface(_widget))
        setattr(self, "lp", LatticePlane(_widget))

    @property
    def atoms(self):
        return self._widget.atoms

    @atoms.setter
    def atoms(self, atoms):
        self._widget.atoms = atoms
        # initialize atomScales
        if isinstance(atoms, list):
            atoms = atoms[0]
        natom = len(atoms["symbols"])
        self._widget.atomScales = [1] * natom
        self._widget.modelSticks = [0] * natom
        self._widget.modelPolyhedras = [0] * natom
        # magnetic moment vector field
        # separate spin up and down, add two vector fields
        if "moment" in atoms["attributes"]["atom"]:
            moment = atoms["attributes"]["atom"]["moment"]
            spin_up = [i for i, m in enumerate(moment) if m > 0]
            spin_down = [i for i, m in enumerate(moment) if m < 0]
            self.vf.settings = [
                {
                    "origins": [atoms["positions"][i] for i in spin_up],
                    "vectors": [[0, 0, moment[i]] for i in spin_up],
                    "color": "blue",
                },
                {
                    "origins": [atoms["positions"][i] for i in spin_down],
                    "vectors": [[0, 0, moment[i]] for i in spin_down],
                    "color": "red",
                },
            ]

    def draw(self):
        """Redraw the widget."""
        self._widget.send_js_task({"name": "avr.drawModels", "kwargs": {}})
