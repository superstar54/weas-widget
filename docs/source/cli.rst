=======================
Command Line Interface
=======================

WeasWidget provides a command-line interface (CLI) for visualizing atomic structures, molecules, trajectories, and volumetric data using WEAS.

Usage
-----
Run `weas <filename>` to visualize structures:

.. code-block:: bash

    weas <filename> [OPTIONS]

Examples
--------

1. **Visualizing a crystal structure (CIF file)**:

   .. code-block:: bash

       weas Li2Mn3NiO8.cif

2. **Visualizing a molecular dynamics trajectory (XYZ file)**:

   .. code-block:: bash

       weas deca_ala_md.xyz

3. **Phonon mode visualization**:

   .. code-block:: bash

       weas graphene.cif --phonon --eigenvectors '[ [[-0.31, 0.47], [-0.16, -0.38], [0, 0]], [[0.54, -0.15], [-0.31, -0.27], [0, 0]] ]'

Options
-------

- `--style <int>` : Model style (0 = ball, 1 = ball+stick, 2 = polyhedra).
- `--color-type <str>` : Atom color scheme (`CPK`, `VESTA`).
- `--boundary <str>` : Periodic boundary conditions (JSON format).
- `--phonon` : Enable phonon visualization.
- `--eigenvectors <json>` : Phonon eigenvectors in JSON format.
- `--kpoint <json>` : K-point for phonon mode.
- `--amplitude <float>` : Phonon amplitude (default: 2).
- `--nframes <int>` : Number of animation frames (default: 50).
