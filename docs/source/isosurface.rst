Isosurface
=================

The isosurface is a 3D surface that represents points of a constant value within a volume of data. It is a powerful tool to visualize the orbital of a molecule, the charge density of a crystal and other volumetric data.


Usage Example
-------------
Here is an example of drawing isosurfaces for HOMO of H2O molecule.

.. code-block:: python

   from ase.build import molecule
   from weas_widget import WeasWidget
   from ase.io.cube import read_cube_data
   volume, atoms = read_cube_data("h2o-homo.cube")
   viewer = WeasWidget()
   viewer.from_ase(atoms)
   viewer.avr.iso.volumetric_data = {"values": volume}
   viewer.avr.iso.settings = {"positive": {"isovalue": 0.001},
                              "negative": {"isovalue": -0.001, "color": "yellow"}
                             }
   viewer

.. figure:: _static/images/example-isosurface.png
   :width: 40%
   :align: center


For the ``isoSetting``:

- **isovalue**: The value used to generate the isosurface. If null, it will be computed as the average of the data range.
- **color**: The color of the isosurface.
- **mode**: The mode of isosurface generation.
   - mode=0: Positive and negative isosurfaces are drawn. In this case, the color of the positive is the given color, and the color of the negative is the complementary color of the given color
   - mode=other: Only the given isosurface is drawn.


.. tip::

   Support for multiple isosurfaces with individual properties (isovalue and color).
