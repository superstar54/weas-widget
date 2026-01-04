Apps
====

Apps are small, task-specific widgets that demonstrate how to combine
``weas-widget`` with other Python packages to build structures
interactively. ``SurfaceBuilder`` is the first app in this series, and
more apps will be added over time.

SurfaceBuilder
--------------

``SurfaceBuilder`` uses ASE to cut a slab from a bulk structure and
visualize the result in a live widget.

.. code-block:: python

   import ase
   from ase.build import bulk
   from weas_widget.apps.surface import SurfaceBuilder

   w = SurfaceBuilder()
   w.bulk = bulk("Cu", "fcc", a=3.6)
   w
