import anywidget
import traitlets
import os

esm_path = os.path.join(os.path.dirname(__file__), """index.js""")


class WeasWidget(anywidget.AnyWidget):
    _esm = esm_path
    atoms = traitlets.Dict().tag(sync=True)

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

    def from_pymatgen(self, structure):
        # Convert a Pymatgen Structure object to the widget's format
        atoms_data = self._convert_to_dict(structure)
        self.atoms = atoms_data
