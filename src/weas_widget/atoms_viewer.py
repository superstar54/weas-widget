from .base_class import WidgetWrapper
from .plugins.vector_field import VectorField
from .plugins.isosurface import Isosurface
from .plugins.lattice_plane import LatticePlane
from .plugins.bond import BondManager
from .plugins.species import SpeciesManager
from .plugins.highlight import HighlightManager
from copy import deepcopy


class AtomsViewer(WidgetWrapper):

    catalog = "viewer"

    _attribute_map = {
        "model_style": "modelStyle",
        "selected_atoms_indices": "selectedAtomsIndices",
        "boundary": "boundary",
        "color_by": "colorBy",
        "color_ramp": "colorRamp",
        "show_cell": "showCell",
        "show_bonded_atoms": "showBondedAtoms",
        "hide_long_bonds": "hideLongBonds",
        "show_hydrogen_bonds": "showHydrogenBonds",
        "atom_label_type": "atomLabelType",
        "material_type": "materialType",
        "model_sticks": "modelSticks",
        "atom_scales": "atomScales",
        "model_polyhedras": "modelPolyhedras",
        "current_frame": "currentFrame",
        "phonon_setting": "phonon",
    }

    _extra_allowed_attrs = [
        "species",
        "vf",
        "iso",
        "lp",
        "atoms",
        "bond",
        "highlight",
        "color_type",
    ]

    def __init__(self, _widget):

        super().__init__(_widget)
        # Initialize plugins
        setattr(self, "vf", VectorField(_widget))
        setattr(self, "iso", Isosurface(_widget))
        setattr(self, "lp", LatticePlane(_widget))
        setattr(self, "bond", BondManager(_widget))
        setattr(self, "species", SpeciesManager(_widget))
        setattr(self, "highlight", HighlightManager(_widget))

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
        # species
        self.species.update_atoms()
        # bond
        self.bond.update_atoms()
        # vector field
        self.vf.update_atoms()
        # highlight
        self.highlight.update_atoms()

    def draw(self):
        """Redraw the widget."""
        self._widget.send_js_task({"name": "avr.drawModels", "kwargs": {}})

    def set_attribute(self, name, value, domain="atoms"):
        """Set an attribute of the widget."""
        atoms = deepcopy(self._widget.atoms)
        atoms["attributes"][domain][name] = value
        self._widget.atoms = atoms
        # update the widget
        self._widget.send_js_task(
            {"name": "avr.setAttribute", "args": [name, value, domain]}
        )

    def get_attribute(self, name):
        """Get an attribute of the widget."""
        raise NotImplementedError("This method is not implemented yet.")

    @property
    def color_type(self):
        return self._widget.colorType

    @color_type.setter
    def color_type(self, value):
        self._widget.colorType = value
        self.species.update_atoms()
        self.bond.update_atoms()
