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
