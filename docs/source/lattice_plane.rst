Lattice plane
=================

The lattice plane is a plane that intersects the lattice. It is useful to visualize the lattice plane in the crystal structure.

Plane form miller indices
--------------------------
The lattice plane can be defined by the miller indices and distance from the origin or by selecting the atoms.

Here is an example of how to visualize lattice planes (111):

.. code-block:: python

    from ase.build import bulk
    from weas_widget import WeasWidget
    import numpy as np
    atoms = bulk("Au", cubic=True)
    viewer = WeasWidget()
    viewer.from_ase(atoms)
    viewer.avr.model_style = 1
    viewer.camera.setting = {"direction": [0, -0.2, 1], "zoom": 0.8}
    viewer

In another cell:

.. code-block:: python

    # color is defined by RGBA, where R is red, G is green, B is blue, and A is the transparency
    viewer.avr.lp.add_plane_from_indices(name = "111",
                                         indices = [1, 1, 1],
                                         distance = 4,
                                         scale = 1.0,
                                         color = [0, 1, 1, 0.5])
    viewer.avr.lp.build_plane()

.. figure:: _static/images/lattice_plane.png
   :align: center


Plane from selected atoms
--------------------------
One can also draw a plane from the selected atoms. Here is an example:


.. code-block:: python

    viewer.avr.lp.add_plane_from_selected_atoms(name = "plane1",
                                                color = [1, 0, 0, 0.5])
    viewer.avr.lp.build_plane()
