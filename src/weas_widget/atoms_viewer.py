from .base_class import WidgetWrapper
from .plugins.vector_field import VectorField
from .plugins.isosurface import Isosurface
from .plugins.volume_slice import VolumeSlice
from .plugins.lattice_plane import LatticePlane
from .plugins.cell import CellManager
from .plugins.bond import BondManager
from .plugins.species import SpeciesManager
from .plugins.highlight import HighlightManager


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
        "show_atom_legend": "showAtomLegend",
        "show_hydrogen_bonds": "showHydrogenBonds",
        "show_out_boundary_bonds": "showOutBoundaryBonds",
        "atom_label_type": "atomLabelType",
        "material_type": "materialType",
        "model_sticks": "modelSticks",
        "atom_scales": "atomScales",
        "model_polyhedras": "modelPolyhedras",
        "current_frame": "currentFrame",
        "phonon_setting": "phonon",
        "continuous_update": "continuousUpdate",
    }

    _extra_allowed_attrs = [
        "species",
        "vf",
        "iso",
        "volume_slice",
        "lp",
        "atoms",
        "cell",
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
        setattr(self, "cell", CellManager(_widget))
        setattr(self, "bond", BondManager(_widget))
        setattr(self, "species", SpeciesManager(_widget))
        setattr(self, "highlight", HighlightManager(_widget))
        setattr(self, "volume_slice", VolumeSlice(_widget))

    @property
    def atoms(self):
        return self._widget.atoms

    @atoms.setter
    def atoms(self, atoms):
        self._widget.atoms = atoms
        # initialize atomScales
        if isinstance(atoms, list):
            atoms = atoms[0]
        # species
        self.species.update_atoms()
        # bond
        self.bond.update_atoms()
        # vector field
        self.vf.update_atoms()
        # highlight
        self.highlight.update_atoms()
        self._update_fixed_highlight(atoms)
        self._widget.send_js_task({"name": "tjs.onWindowResize"})

    def _update_fixed_highlight(self, atoms):
        if not isinstance(atoms, dict):
            return
        attributes = atoms.get("attributes", {})
        atom_attrs = attributes.get("atom", {})
        fixed_xyz = atom_attrs.get("fixed_xyz")
        if not isinstance(fixed_xyz, list):
            return
        indices = [
            idx
            for idx, mask in enumerate(fixed_xyz)
            if isinstance(mask, list) and any(mask)
        ]
        if not indices:
            return
        self.highlight.settings["fixed"] = {
            "type": "crossView",
            "indices": indices,
            "scale": 1.0,
            "color": "black",
        }

    def draw(self):
        """Redraw the widget."""
        self._widget.send_js_task({"name": "avr.drawModels", "kwargs": {}})

    def set_attribute(self, name, value, domain="atoms"):
        """Set an attribute of the widget."""
        atoms = self._widget.atoms
        if isinstance(atoms, list):
            frame = int(getattr(self, "current_frame", 0))
            frame = max(0, min(frame, len(atoms) - 1))
            target = atoms[frame]
        else:
            target = atoms
        if "attributes" not in target:
            target["attributes"] = {"atom": {}, "species": {}}
        if domain not in target["attributes"]:
            target["attributes"][domain] = {}
        target["attributes"][domain][name] = value
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
