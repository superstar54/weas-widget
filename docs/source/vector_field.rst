Vector Field
=================

The vector field is defined by the origins and vectors, and it is visualized by the arrows. The vector field is useful for visualizing the magnetic moment, phonon visualization, etc.


Magentic moment visualization
-----------------------------
Show the magnetic moment as a vector field.

.. code-block:: python

    from ase.build import bulk
    from weas_widget import WeasWidget
    import numpy as np
    atoms = bulk("Fe", cubic=True)
    atoms*=[2, 2, 1]
    atoms.set_array("moment", np.ones(len(atoms)))
    viewer = WeasWidget()
    viewer.from_ase(atoms)
    viewer.avr.model_style = 1
    viewer

.. figure:: _static/images/example-magnetic-moment.png
   :align: center



Interactive phonon visualization
--------------------------------
One can visualize the phonon dispersion via lattice vibrations. One only need to use the eigenstates (calculated with an external software) to generate the trajectory. Each eigenvector should has the real and imaginary part. One can also specify the kpoint, amplitude, nframes, repeat, color, and radius.

.. code-block:: python

    import numpy as np
    from ase.build import bulk
    from weas_widget import WeasWidget
    atoms = bulk("Fe", cubic=True)
    phonon_setting = {"eigenvectors": np.array([[[0, 0], [0, 0],[0.5, 0]],
                                        [[0, 0], [0, 0], [-0.5, 0]]]
                                       ),
            "kpoint": [0, 0, 0], # optional
            "amplitude": 5, # scale the motion of the atoms
            "factor": 1.5, # scale the length of the arrows
            "nframes": 20,
            "repeat": [4, 4, 1],
            "color": "blue",
            "radius": 0.1,
            }
    viewer = WeasWidget()
    viewer.from_ase(atoms)
    viewer.avr.phonon_setting = phonon_setting
    viewer


.. figure:: _static/images/example-phonon.gif
   :align: center
