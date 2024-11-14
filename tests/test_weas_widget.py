from weas_widget import WeasWidget
import numpy as np


def test_widget_initialization():
    widget = WeasWidget()
    assert widget is not None


def test_set_and_get_atoms_ase_molecule(h2o):
    from ase.build import molecule

    atoms = molecule("H2O")
    retrieved_atoms = h2o.to_ase()

    # Assert
    assert retrieved_atoms == atoms


def test_set_and_get_atoms_ase_bulk():
    from ase.build import bulk

    atoms = bulk("Si", "diamond", a=5.43)
    atoms.pbc = [True, True, False]
    retrieved_atoms = WeasWidget(from_ase=atoms).to_ase()

    assert retrieved_atoms == atoms
    assert np.allclose(retrieved_atoms.cell, atoms.cell)
    assert np.all(retrieved_atoms.pbc == atoms.pbc)


def test_set_and_get_pymatgen_structure():
    from pymatgen.core import Structure, Lattice

    # Original structure with PBC in all directions
    lattice = Lattice.from_parameters(
        a=5, b=5, c=10, alpha=90, beta=90, gamma=90, pbc=[True, True, False]
    )
    species = ["Si", "Si"]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    structure = Structure(lattice, species, coords)

    viewer = WeasWidget(from_pymatgen=structure)
    retrieved_structure = viewer.to_pymatgen()

    # Assert
    assert retrieved_structure == structure
    assert retrieved_structure.lattice.pbc == structure.lattice.pbc
    assert np.allclose(retrieved_structure.lattice.matrix, structure.lattice.matrix)


def test_set_and_get_pymatgen_molecule():
    from pymatgen.core import Molecule

    structure = Molecule(["O", "H", "H"], [[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    viewer = WeasWidget(from_pymatgen=structure)
    retrieved_structure = viewer.to_pymatgen()

    # Assert
    assert len(retrieved_structure) == len(structure)
    assert isinstance(retrieved_structure, Molecule)
