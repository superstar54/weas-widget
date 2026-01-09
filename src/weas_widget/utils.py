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
        fixed_xyz = cls._extract_fixed_xyz(ase_atoms)
        if fixed_xyz is not None:
            attributes["atom"]["fixed_xyz"] = fixed_xyz

        weas_atoms = {
            "species": species,
            "cell": cell,
            "positions": positions.tolist(),
            "symbols": symbols,
            "attributes": attributes,
            "pbc": ase_atoms.get_pbc().tolist(),
        }
        return weas_atoms

    @staticmethod
    def _extract_fixed_xyz(ase_atoms):
        constraints = getattr(ase_atoms, "constraints", None)
        if not constraints:
            return None
        if not isinstance(constraints, (list, tuple)):
            constraints = [constraints]
        n_atoms = len(ase_atoms)
        fixed = np.zeros((n_atoms, 3), dtype=bool)

        for constraint in constraints:
            indices = None
            if hasattr(constraint, "get_indices"):
                try:
                    indices = list(constraint.get_indices())
                except Exception:
                    indices = None
            if indices is None:
                if hasattr(constraint, "indices"):
                    indices = list(constraint.indices)
                elif hasattr(constraint, "index"):
                    indices = [constraint.index]
            if not indices:
                continue

            if hasattr(constraint, "mask"):
                mask = np.asarray(constraint.mask, dtype=bool)
                if mask.shape == (3,):
                    fixed[indices, :] |= mask
                    continue
                if (
                    mask.ndim == 2
                    and mask.shape[1] == 3
                    and mask.shape[0] == len(indices)
                ):
                    for idx, axis_mask in zip(indices, mask):
                        fixed[idx] |= axis_mask
                    continue
            if hasattr(constraint, "masks"):
                masks = np.asarray(constraint.masks, dtype=bool)
                if (
                    masks.ndim == 2
                    and masks.shape[1] == 3
                    and masks.shape[0] == len(indices)
                ):
                    for idx, axis_mask in zip(indices, masks):
                        fixed[idx] |= axis_mask
                    continue

            fixed[indices, :] = True

        if not fixed.any():
            return None
        return fixed.tolist()

    @classmethod
    def to_ase(cls, weas_atoms):
        # Convert the widget's format to an ASE Atoms object
        from ase import Atoms
        from ase.constraints import FixAtoms, FixCartesian
        import numpy as np

        # if atoms is a list of atoms, convert all atoms to a list of ase atoms
        if isinstance(weas_atoms, list):
            trajectory = [cls.to_ase(atom) for atom in weas_atoms]
            return trajectory[0] if len(trajectory) == 1 else trajectory
        if not isinstance(weas_atoms, dict) or "symbols" not in weas_atoms:
            return Atoms(
                symbols=[],
                positions=[],
                cell=np.zeros((3, 3), dtype=float),
                pbc=[False, False, False],
            )
        symbols = [weas_atoms["species"][s] for s in weas_atoms["symbols"]]
        positions = weas_atoms["positions"]
        cell = np.array(weas_atoms["cell"]).reshape(3, 3)
        ase_atoms = Atoms(
            symbols=symbols, positions=positions, cell=cell, pbc=weas_atoms["pbc"]
        )
        attributes = weas_atoms.get("attributes", {})
        atom_attrs = attributes.get("atom", {})
        fixed_xyz = atom_attrs.get("fixed_xyz")
        if isinstance(fixed_xyz, list) and fixed_xyz:
            masks = []
            for entry in fixed_xyz:
                if isinstance(entry, (list, tuple)) and len(entry) == 3:
                    masks.append(tuple(bool(x) for x in entry))
                else:
                    masks.append((False, False, False))
            if any(any(mask) for mask in masks):
                constraints = []
                fixed_all = [
                    i for i, mask in enumerate(masks) if mask == (True, True, True)
                ]
                if fixed_all:
                    constraints.append(FixAtoms(indices=fixed_all))
                partial_groups = {}
                for i, mask in enumerate(masks):
                    if mask == (True, True, True) or mask == (False, False, False):
                        continue
                    partial_groups.setdefault(mask, []).append(i)
                for mask, indices in partial_groups.items():
                    constraints.append(FixCartesian(indices=indices, mask=list(mask)))
                if constraints:
                    ase_atoms.set_constraint(
                        constraints if len(constraints) > 1 else constraints[0]
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
        selective = attributes["atom"].get("selective_dynamics")
        if isinstance(selective, list) and selective:
            fixed_xyz = []
            for entry in selective:
                if isinstance(entry, (list, tuple)) and len(entry) == 3:
                    fixed_xyz.append([not bool(x) for x in entry])
                else:
                    fixed_xyz.append([False, False, False])
            if any(any(mask) for mask in fixed_xyz):
                attributes["atom"]["fixed_xyz"] = fixed_xyz
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
        attributes = weas_atoms.get("attributes", {})
        atom_attrs = attributes.get("atom", {})
        fixed_xyz = atom_attrs.get("fixed_xyz")
        if isinstance(fixed_xyz, list) and fixed_xyz:
            selective = []
            for entry in fixed_xyz:
                if isinstance(entry, (list, tuple)) and len(entry) == 3:
                    selective.append([not bool(x) for x in entry])
                else:
                    selective.append([True, True, True])
            if any(not all(mask) for mask in selective):
                try:
                    structure.add_site_property("selective_dynamics", selective)
                except Exception:
                    structure.site_properties["selective_dynamics"] = selective
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


def group_layers_by_coordinate(values, tolerance: float, *, descending: bool = False):
    if len(values) == 0:
        return []
    order = np.argsort(values)
    if descending:
        order = order[::-1]
    groups = []
    current = [int(order[0])]
    ref_value = float(values[order[0]])
    for idx in order[1:]:
        idx = int(idx)
        if abs(float(values[idx]) - ref_value) <= float(tolerance):
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
            ref_value = float(values[idx])
    groups.append(current)
    return groups
