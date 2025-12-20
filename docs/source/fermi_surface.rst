Fermi Surface
=================
Generate and visualize Fermi surfaces from BXSF files.

Here's an example of how to use the `add_fermi_surface_from_bxsf` method:


.. code-block:: python

    from weas_widget import WeasWidget

    viewer = WeasWidget()
    viewer.add_fermi_surface_from_bxsf(file_path="copper.bxsf", clip_bz=True)
    viewer



.. image:: _static/images/fermi-surface.png
   :width: 15cm


Parameters summary for `add_fermi_surface_from_bxsf`:

- `file_path`: Path to the `.bxsf` file.
- `fermi_energy`: Override the Fermi energy (default: use value in file).
- `drop_periodic`: Drop the duplicated periodic end points (default: `True`).
- `clip_bz`: Clip to the first Brillouin zone (default: `False`, requires `seekpath`).
- `show_bz`: Add the Brillouin zone mesh (default: `True`).
- `show_reciprocal_axes`: Add reciprocal axes vectors (default: `True`).
- `band_index`: Render a single band by index (default: `None`).
- `combine_bands`: Merge all Fermi-crossing bands into one mesh (default: `True`).
- `name`: Mesh name override.
- `color`: RGB or RGBA list for the mesh color (default: orange with `opacity`).
- `opacity`: Alpha channel used when `color` is RGB (default: `0.6`).
- `material_type`: Material type for the mesh (default: `"Standard"`).
- `supercell_size`: Tuple for band replication in reciprocal space (default: `(2, 2, 2)`).

Notes:

- If `band_index` is `None`, all Fermi-crossing bands are used.
- If `combine_bands=False`, each band is added as a separate mesh.
- A `ValueError` is raised when no Fermi-crossing bands are found.
