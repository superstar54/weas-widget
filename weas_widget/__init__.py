import anywidget
import traitlets
import os
from .utils import ASE_Adapter, Pymatgen_Adapter

esm_path = os.path.join(os.path.dirname(__file__), """index.js""")
css_path = os.path.join(os.path.dirname(__file__), """style.css""")


class WeasWidget(anywidget.AnyWidget):
    _esm = esm_path
    _css = css_path
    atoms = traitlets.Dict().tag(sync=True)
    selectedAtomsIndices = traitlets.List([]).tag(sync=True)
    boundary = traitlets.List([[0, 1], [0, 1], [0, 1]]).tag(sync=True)
    modelStyle = traitlets.Int(0).tag(sync=True)
    colorType = traitlets.Unicode("CPK").tag(sync=True)
    materialType = traitlets.Unicode("Standard").tag(sync=True)
    atomLabelType = traitlets.Unicode("None").tag(sync=True)
    showCell = traitlets.Bool(True).tag(sync=True)
    showBondedAtoms = traitlets.Bool(False).tag(sync=True)
    _drawModels = traitlets.Bool(False).tag(sync=True)
    atomScales = traitlets.List([]).tag(sync=True)
    modelSticks = traitlets.List([]).tag(sync=True)
    modelPolyhedras = traitlets.List([]).tag(sync=True)

    def drawModels(self):
        """Redraw the widget."""
        self._drawModels = not self._drawModels

    def set_atoms(self, atoms):
        self.atoms = atoms
        # initialize atomScales
        self.atomScales = [1] * len(atoms["speciesArray"])
        self.modelSticks = [0] * len(atoms["speciesArray"])
        self.modelPolyhedras = [0] * len(atoms["speciesArray"])

    def from_ase(self, atoms):
        self.set_atoms(ASE_Adapter.to_weas(atoms))

    def to_ase(self):
        return ASE_Adapter.to_ase(self.atoms)

    def from_pymatgen(self, structure):
        self.set_atoms(Pymatgen_Adapter.to_weas(structure))

    def to_pymatgen(self):
        return Pymatgen_Adapter.to_pymatgen(self.atoms)

    def load_example(self, name="tio2.cif"):
        from ase.io import read

        atoms = read(os.path.join(os.path.dirname(__file__), f"datas/{name}"))
        self.set_atoms(ASE_Adapter.to_weas(atoms))
