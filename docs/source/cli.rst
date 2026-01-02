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

4. **Loading a saved WEAS state**:

   .. code-block:: bash

       weas weas-state.json

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

.. note::

    When the input file is a WEAS state snapshot (`.json`), the viewer loads
    the saved state directly and ignores `--style`, `--color-type`, and
    `--boundary` unless explicitly overridden in the snapshot itself.


Running on a Remote Computer
----------------------------

If you are running `weas` on a remote server, you can access the visualization locally using SSH port forwarding.

1. **Start the visualization on the remote machine**:

   .. code-block:: bash

       weas structure.cif --use-server

   The command will print a message like:

   .. code-block::

       Serving at http://localhost:8000
       Open this URL in your browser to access the visualization.

2. **Forward the port to your local machine**:

   On your local machine, run:

   .. code-block:: bash

       ssh -L 8000:localhost:8000 your_remote_user@your_remote_host

   Then, open `http://localhost:8000` in your browser.


.. note::

    If you are using **Visual Studio Code** with Remote SSH, port forwarding is handled automatically. When you start the server with `--use-server`, VS Code will detect the port and provide a clickable link in the terminal.
