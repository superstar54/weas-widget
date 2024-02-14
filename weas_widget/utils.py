class ASE_Adapter:
    def __init__(self):
        pass

    @classmethod
    def to_weas(cls, ase_atoms):
        """Convert an ASE Atoms object to the widget's format."""
        # if atoms is a list of atoms, check if they are the same species and number of atoms
        # then convert all atoms to weas format as a list of atoms
        if isinstance(ase_atoms, list):
            if len(ase_atoms) > 0:
                for atoms in ase_atoms:
                    if (
                        atoms.get_chemical_symbols()
                        != ase_atoms[0].get_chemical_symbols()
                    ):
                        raise ValueError("All atoms must have the same species")
                weas_atoms = [cls.to_weas(atom) for atom in ase_atoms]
                return weas_atoms
            else:
                raise ValueError("The list of atoms is empty")
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

        # if atoms is a list of atoms, convert all atoms to a list of ase atoms
        if isinstance(weas_atoms, list):
            return [cls.to_ase(atom) for atom in weas_atoms]
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
        positions = [site.coords for site in pymatgen_structure.sites]
        symbols = [site.species_string for site in pymatgen_structure.sites]
        speciesArray = symbols
        for i in range(len(symbols)):
            species[symbols[i]] = [symbols[i]]
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
        from pymatgen.core import Structure, Lattice

        if isinstance(weas_atoms, list):
            return [cls.to_pymatgen(atom) for atom in weas_atoms]

        lattice = Lattice(weas_atoms["cell"])
        species = weas_atoms["speciesArray"]
        sites = weas_atoms["positions"]
        structure = Structure(lattice, species, sites)
        return structure
