import numpy as np
from ase.io.trajectory import TrajectoryReader


class ASEAdapter:
    def __init__(self):
        pass

    @classmethod
    def to_weas(cls, ase_atoms):
        """Convert an ASE Atoms object to the widget's format."""
        # if atoms is a list of atoms, check if they are the same species and number of atoms
        # then convert all atoms to weas format as a list of atoms
        if isinstance(ase_atoms, (list, TrajectoryReader)):
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
        for i in range(len(symbols)):
            species[symbols[i]] = symbols[i]
        # save other arrays to attributes
        attributes = {"atom": {}, "species": {}}
        for key in ase_atoms.arrays.keys():
            if key not in ["positions", "numbers"]:
                attributes["atom"][key] = ase_atoms.arrays[key].tolist()

        weas_atoms = {
            "species": species,
            "cell": cell,
            "positions": positions.tolist(),
            "symbols": symbols,
            "attributes": attributes,
            "pbc": ase_atoms.get_pbc().tolist(),
        }
        return weas_atoms

    @classmethod
    def to_ase(cls, weas_atoms):
        # Convert the widget's format to an ASE Atoms object
        from ase import Atoms
        import numpy as np

        # if atoms is a list of atoms, convert all atoms to a list of ase atoms
        if isinstance(weas_atoms, list):
            trajectory = [cls.to_ase(atom) for atom in weas_atoms]
            return trajectory[0] if len(trajectory) == 1 else trajectory
        symbols = [weas_atoms["species"][s] for s in weas_atoms["symbols"]]
        positions = weas_atoms["positions"]
        cell = np.array(weas_atoms["cell"]).reshape(3, 3)
        ase_atoms = Atoms(
            symbols=symbols, positions=positions, cell=cell, pbc=weas_atoms["pbc"]
        )
        return ase_atoms


class PymatgenAdapter:
    def __init__(self):
        pass

    @classmethod
    def to_weas(cls, pymatgen_structure):
        from pymatgen.core import Molecule

        species = {}
        # structure is a Molecule, convert it to Structure
        if isinstance(pymatgen_structure, Molecule):
            cell = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            pbc = [False, False, False]
        else:
            cell = pymatgen_structure.lattice.matrix.flatten().tolist()
            pbc = pymatgen_structure.lattice.pbc
        positions = [site.coords.tolist() for site in pymatgen_structure.sites]
        symbols = [site.species_string for site in pymatgen_structure.sites]
        for i in range(len(symbols)):
            species[symbols[i]] = symbols[i]
        # save other arrays to attributes
        attributes = {"atom": {}, "species": {}}
        # read pymatgen site properties
        for key in pymatgen_structure.site_properties.keys():
            attributes["atom"][key] = [
                site.properties[key] for site in pymatgen_structure.sites
            ]
        weas_atoms = {
            "species": species,
            "cell": cell,
            "positions": positions,
            "symbols": symbols,
            "attributes": attributes,
            "pbc": pbc,
        }
        return weas_atoms

    @classmethod
    def to_pymatgen(cls, weas_atoms):
        # Convert the widget's format to a Pymatgen Structure object
        from pymatgen.core import Molecule, Structure, Lattice

        if isinstance(weas_atoms, list):
            return [cls.to_pymatgen(atom) for atom in weas_atoms]
        species = weas_atoms["symbols"]
        sites = weas_atoms["positions"]
        cell = np.array(weas_atoms["cell"]).reshape(3, 3)
        # if all cell are close to zeros, it is a molecule
        if np.allclose(cell, np.zeros((3, 3))):
            structure = Molecule(species, sites)
        else:
            lattice = Lattice(weas_atoms["cell"], pbc=weas_atoms["pbc"])
            structure = Structure(lattice, species, sites, coords_are_cartesian=True)
        return structure


def load_online_example(name="tio2.cif"):
    """Load an example from the online data."""
    from ase.io import read
    import requests
    from io import StringIO

    url = "https://raw.githubusercontent.com/superstar54/weas/main/demo/datas/" + name
    # Download the file content
    response = requests.get(url)
    if response.status_code == 200:
        file_content = response.text
        # Use StringIO to simulate a file-like object for ASE to read from
        file_like_object = StringIO(file_content)
        atoms = read(file_like_object, format="cif")
        return atoms
    else:
        raise ValueError(f"Failed to download the file {name}")


def create_volume_data(data, cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
    """
    Convert a 3D nested list (data.values) into a structured volume data format.

    Parameters:
        data (dict): A dictionary containing a 3D list `values`.
        cell (list, optional): The transformation matrix for the cell. Default is identity matrix.

    Returns:
        dict: A dictionary with dimensions, flattened values, cell, and origin.
    """
    # Get the dimensions of the 3D data
    dims = [len(data), len(data[0]), len(data[0][0])]

    # Flatten the 3D data into a 1D list
    values = np.array(data).flatten().tolist()

    return {"dims": dims, "values": values, "cell": cell, "origin": [0, 0, 0]}
