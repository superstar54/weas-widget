import numpy as np


class ASEAdapter:
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
        for i in range(len(symbols)):
            species[symbols[i]] = symbols[i]
        # save other arrays to attributes
        attributes = {"atom": {}, "species": {}}
        for key in ase_atoms.arrays.keys():
            if key not in ["positions", "numbers"]:
                attributes["atom"][key] = ase_atoms.arrays[key]

        weas_atoms = {
            "species": species,
            "cell": cell,
            "positions": positions.tolist(),
            "symbols": symbols,
            "attributes": attributes,
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
        ase_atoms = Atoms(symbols=symbols, positions=positions, cell=cell)
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
        else:
            cell = pymatgen_structure.lattice.matrix.flatten().tolist()
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
            lattice = Lattice(weas_atoms["cell"])
            structure = Structure(lattice, species, sites, coords_are_cartesian=True)
        return structure


def generate_phonon_trajectory(
    atoms,
    eigenvectors,
    amplitude=1,
    nframes=20,
    scale=5,
    attribute="movement",
    repeat=[1, 1, 1],
):
    """Generate a trajectory of atoms vibrating along the given eigenvectors.
    Args:
        atoms: ASE Atoms object
        eigenvectors: Eigenvectors of the dynamical matrix
        amplitude: Amplitude of the vibration
    Returns:
        A list of ASE Atoms objects representing the trajectory
    """
    import numpy as np

    trajectory = []
    times = np.linspace(0, 2 * np.pi, nframes)
    for t in times:
        # arrow vector
        vectors = amplitude * eigenvectors * np.sin(t)
        # new atoms
        new_atoms = atoms.copy()
        new_atoms.positions = atoms.positions + vectors
        new_atoms.set_array(attribute, vectors * scale)
        new_atoms = new_atoms.repeat(repeat)
        trajectory.append(new_atoms)
    return trajectory


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
