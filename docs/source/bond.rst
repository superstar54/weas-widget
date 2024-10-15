Bond
===============
Use can control the bond using `avr.bond.settings`. For example, we delete the bond between Ca and Ti, and Ti and Ca.

.. code-block:: python

    from weas_widget import WeasWidget
    from ase.io import read
    from ase import Atoms
    from pprint import pprint
    from copy import deepcopy
    import time

    positions = [
        (0.50000000, 0.50000000, 0.50000000),  # Ca
        (0.00000000, 0.00000000, 0.00000000),  # Ti
        (0.00000000, 0.50000000, 0.00000000),  # O
        (0.00000000, 0.00000000, 0.50000000),  # O
        (0.50000000, 0.00000000, 0.00000000)   # O
    ]
    atoms = Atoms(
        symbols=['Ca', 'Ti', 'O', 'O', 'O'],  # List of element symbols
        scaled_positions=positions,
        cell=[3.889, 3.889, 3.889],
        pbc=[True, True, True]  # Periodic boundary conditions
    )
    viewer1 = WeasWidget()
    viewer1.from_ase(atoms)
    # show boundary atoms
    viewer1.avr.boundary = [[-0.1, 1.1], [-0.1, 1.1], [-0.1, 1.1]]
    # show bonded atoms outside the cell
    viewer1.avr.show_bonded_atoms = True
    # Change color tyoe to "VESTA"
    viewer1.avr.color_type = "VESTA"
    settings = viewer1.avr.bond.settings.copy()
    del settings['[Ti, Ca]']
    del settings['[Ca, Ti]']
    viewer1.avr.bond.settings = settings
    viewer1.avr.model_style = 2
    viewer1

.. image:: _static/images/example_bond.png
   :width: 6cm
