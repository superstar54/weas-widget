import anywidget
import traitlets
import os

esm_path = os.path.join(os.path.dirname(__file__), """index.js""")
css_path = os.path.join(os.path.dirname(__file__), """style.css""")


class WeasWidget(anywidget.AnyWidget):
    _esm = esm_path
    _css = css_path
    atoms = traitlets.Dict().tag(sync=True)
    selectedAtoms = traitlets.List([]).tag(sync=True)
    boundary = traitlets.List([[0, 1], [0, 1], [0, 1]]).tag(sync=True)
    modelStyle = traitlets.Int(1).tag(sync=True)
    colorType = traitlets.Unicode("CPK").tag(sync=True)
    materialType = traitlets.Unicode("Standard").tag(sync=True)
    atomLabelType = traitlets.Unicode("None").tag(sync=True)

    def from_ase(self, atoms):
        # Convert an ASE Atoms object to the widget's format
        species = {}
        cell = atoms.get_cell().array.flatten().tolist()
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        numbers = atoms.get_atomic_numbers()
        speciesArray = symbols
        for i in range(len(symbols)):
            species[symbols[i]] = [symbols[i], numbers[i]]
        atoms = {
            "species": species,
            "cell": cell,
            "positions": positions,
            "speciesArray": speciesArray,
        }
        self.atoms = atoms

    def to_ase(self):
        # Convert the widget's format to an ASE Atoms object
        from ase import Atoms
        import numpy as np

        symbols = [self.atoms["species"][s][0] for s in self.atoms["speciesArray"]]
        positions = self.atoms["positions"]
        cell = np.array(self.atoms["cell"]).reshape(3, 3)
        atoms = Atoms(symbols=symbols, positions=positions, cell=cell)
        return atoms

    def from_pymatgen(self, structure):
        # Convert a Pymatgen Structure object to the widget's format
        atoms_data = self._convert_to_dict(structure)
        self.atoms = atoms_data
