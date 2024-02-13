class ASE_Adapter:
    def __init__(self):
        pass

    @classmethod
    def to_weas(cls, ase_atoms):
        # Convert an ASE Atoms object to the widget's format
        species = {}
        cell = ase_atoms.get_cell().array.flatten().tolist()
        positions = ase_atoms.get_positions()
        symbols = ase_atoms.get_chemical_symbols()
        numbers = ase_atoms.get_atomic_numbers()
        speciesArray = symbols
        for i in range(len(symbols)):
            species[symbols[i]] = [symbols[i], numbers[i]]
        weas_atoms = {
            "species": species,
            "cell": cell,
            "positions": positions,
            "speciesArray": speciesArray,
        }
        return weas_atoms

    @classmethod
    def to_ase(cls, weas_atoms):
        # Convert the widget's format to an ASE Atoms object
        from ase import Atoms
        import numpy as np

        symbols = [weas_atoms["species"][s][0] for s in weas_atoms["speciesArray"]]
        positions = weas_atoms["positions"]
        cell = np.array(weas_atoms["cell"]).reshape(3, 3)
        ase_atoms = Atoms(symbols=symbols, positions=positions, cell=cell)
        return ase_atoms


class Pymatgen_Adapter:
    def __init__(self):
        pass

    @classmethod
    def to_weas(cls, pymatgen_structure):
        # Convert a Pymatgen Structure object to the widget's format
        species = {}
        cell = pymatgen_structure.lattice.matrix.flatten().tolist()
        positions = [site.frac_coords for site in pymatgen_structure.sites]
        symbols = [site.species_string for site in pymatgen_structure.sites]
        numbers = [site.species.number for site in pymatgen_structure.sites]
        speciesArray = symbols
        for i in range(len(symbols)):
            species[symbols[i]] = [symbols[i], numbers[i]]
        weas_atoms = {
            "species": species,
            "cell": cell,
            "positions": positions,
            "speciesArray": speciesArray,
        }
        return weas_atoms

    @classmethod
    def to_pymatgen(cls, weas_atoms):
        # Convert the widget's format to a Pymatgen Structure object
        from pymatgen import Structure, Lattice
        from pymatgen.core.sites import PeriodicSite

        lattice = Lattice(weas_atoms["cell"])
        species = weas_atoms["species"]
        sites = [
            PeriodicSite(
                species[weas_atoms["speciesArray"][i]][0],
                weas_atoms["positions"][i],
                lattice,
            )
            for i in range(len(weas_atoms["speciesArray"]))
        ]
        structure = Structure(lattice, species, sites)
        return structure
