def test_add_bond():
    from weas_widget import WeasWidget
    from ase.build import bulk

    atoms = bulk("Al", cubic=True)
    atoms[0].symbol = "Cu"
    viewer = WeasWidget(from_ase=atoms)
    viewer.avr.show_bonded_atoms = True
    viewer.avr.model_style = 1
    viewer.avr.boundary = [[-0.1, 1.1], [-0.1, 1.1], [-0.1, 1.1]]
    # add bonds
    viewer.avr.bond.settings["Al-Cu"] = {
        "specie1": "Al",
        "specie2": "Cu",
        "color1": viewer.avr.species.settings["Al"]["color"],
        "color2": viewer.avr.species.settings["Cu"]["color"],
        "min": 0,
        "max": 3,
    }
    len(viewer.avr.bond.settings) == 1
