Save and Restore State
======================

You can export the full widget state (atoms, viewer settings, plugins, camera,
measurement, and animation) as JSON and load it later.

Example (Python)
---------------

.. code-block:: python

   # export
   state = viewer.export_state()

   # save to file
   viewer.save_state("snapshot.json")

   # load later
   viewer.load_state("snapshot.json")

Class method
------------

.. code-block:: python

   viewer = WeasWidget.from_state_file("snapshot.json")
