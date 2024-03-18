from .plugins.vector_field import VectorField
from .plugins.isosurface import Isosurface


class AtomsViewer:
    def __init__(self, base_widget):
        self.base_widget = base_widget
        self.vf = VectorField(base_widget)
        self.iso = Isosurface(base_widget)

    @property
    def model_style(self):
        return self.base_widget.modelStyle

    @model_style.setter
    def model_style(self, value):
        self.base_widget.modelStyle = value

    @property
    def selected_atoms_indices(self):
        return self.base_widget.selectedAtomsIndices

    @selected_atoms_indices.setter
    def selected_atoms_indices(self, value):
        self.base_widget.selectedAtomsIndices = value

    @property
    def boundary(self):
        return self.base_widget.boundary

    @boundary.setter
    def boundary(self, value):
        self.base_widget.boundary = value

    @property
    def color_type(self):
        return self.base_widget.colorType

    @color_type.setter
    def color_type(self, value):
        self.base_widget.colorType = value

    @property
    def color_by(self):
        return self.base_widget.colorBy

    @color_by.setter
    def color_by(self, value):
        self.base_widget.colorBy = value

    @property
    def color_ramp(self):
        return self.base_widget.colorRamp

    @color_ramp.setter
    def color_ramp(self, value):
        self.base_widget.colorRamp = value

    @property
    def show_cell(self):
        return self.base_widget.showCell

    @show_cell.setter
    def show_cell(self, value):
        self.base_widget.showCell = value

    @property
    def show_bonded_atoms(self):
        return self.base_widget.showBondedAtoms

    @show_bonded_atoms.setter
    def show_bonded_atoms(self, value):
        self.base_widget.showBondedAtoms = value

    @property
    def atom_label_type(self):
        return self.base_widget.atomLabelType

    @atom_label_type.setter
    def atom_label_type(self, value):
        self.base_widget.atomLabelType = value

    @property
    def material_type(self):
        return self.base_widget.materialType

    @material_type.setter
    def material_type(self, value):
        self.base_widget.materialType = value

    @property
    def model_sticks(self):
        return self.base_widget.modelSticks

    @model_sticks.setter
    def model_sticks(self, value):
        self.base_widget.modelSticks = value

    @property
    def atom_scales(self):
        return self.base_widget.atomScales

    @atom_scales.setter
    def atom_scales(self, value):
        self.base_widget.atomScales = value

    @property
    def model_polyhedras(self):
        return self.base_widget.modelPolyhedras

    @model_polyhedras.setter
    def model_polyhedras(self, value):
        self.base_widget.modelPolyhedras = value

    @property
    def selectedAtomsIndices(self):
        return self.base_widget.selectedAtomsIndices

    @selectedAtomsIndices.setter
    def selectedAtomsIndices(self, value):
        self.base_widget.selectedAtomsIndices = value

    @property
    def atoms(self):
        return self.base_widget.atoms

    @atoms.setter
    def atoms(self, atoms):
        self.base_widget.atoms = atoms
        # initialize atomScales
        if isinstance(atoms, list):
            atoms = atoms[0]
        natom = len(atoms["symbols"])
        self.base_widget.atomScales = [1] * natom
        self.base_widget.modelSticks = [0] * natom
        self.base_widget.modelPolyhedras = [0] * natom
        # magnetic moment vector field
        # separate spin up and down, add two vector fields
        if "moment" in atoms["attributes"]["atom"]:
            moment = atoms["attributes"]["atom"]["moment"]
            spin_up = [i for i, m in enumerate(moment) if m > 0]
            spin_down = [i for i, m in enumerate(moment) if m < 0]
            self.vf.settings = [
                {
                    "origins": atoms["positions"][spin_up],
                    "vectors": [[0, 0, m] for m in moment[spin_up]],
                    "color": "blue",
                },
                {
                    "origins": atoms["positions"][spin_down],
                    "vectors": [[0, 0, m] for m in moment[spin_down]],
                    "color": "red",
                },
            ]

    def draw(self):
        """Redraw the widget."""
        self.base_widget.send_js_task({"name": "avr.drawModels", "kwargs": {}})
