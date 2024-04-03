from weas_widget import WeasWidget


def test_widget_initialization():
    widget = WeasWidget()
    assert widget is not None


def test_set_and_get_atoms_ase(h2o):
    from ase.build import molecule

    atoms = molecule("H2O")
    retrieved_atoms = h2o.to_ase()

    # Assert
    assert retrieved_atoms == atoms


def test_set_and_get_pymatgen_structure():
    from pymatgen.core import Structure, Lattice

    structure = Structure.from_spacegroup(
        "Pm-3m", Lattice.cubic(4.1437), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    viewer = WeasWidget(from_pymatgen=structure)
    retrieved_structure = viewer.to_pymatgen()

    # Assert
    assert retrieved_structure == structure


def test_set_and_get_pymatgen_molecule():
    from pymatgen.core import Molecule

    structure = Molecule(["O", "H", "H"], [[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    viewer = WeasWidget(from_pymatgen=structure)
    retrieved_structure = viewer.to_pymatgen()

    # Assert
    assert len(retrieved_structure) == len(structure)
