Color
===============

One can color the atoms using the following scheme:

- Element
- Random
- Uniform
- Index
- Attribute


Color by element
----------------

Supported style are:

#. **JMOL**: http://jmol.sourceforge.net/jscolors/#color_U
#. **VESTA**: https://jp-minerals.org/vesta/en/
#. **CPK**: https://en.wikipedia.org/wiki/CPK_coloring


Color by attribute
----------------------
Coloring based on the attribute of the atoms. The attribute can be: charge, magmom, or any other attribute in the structure.

Here we show how to color the atoms by their forces.


.. code-block:: python

    from ase.build import bulk
    from ase.calculators.emt import EMT
    import numpy as np
    from weas_widget import WeasWidget

    atoms = bulk('Au', cubic = True)
    atoms *= [3, 3, 3]
    atoms.positions += np.random.random((len(atoms), 3))
    atoms.calc = EMT()
    atoms.get_potential_energy()
    # set the forces as an attribute
    atoms.set_array("Force", atoms.calc.results["forces"])

    viewer = WeasWidget()
    viewer.from_ase(atoms)
    viewer.avr.color_by = "Force"
    viewer.avr.color_ramp = ["red", "yellow", "blue"]
    viewer.avr.model_style = 1
    viewer



.. image:: _static/images/example_color_by_force.png
   :width: 10cm
