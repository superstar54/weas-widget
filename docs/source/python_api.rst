==================
Python API
==================

This API is not stable yet and is still being extended and improved.


Features:
==================
The API allows users to:

- Inspect and edit any data the user interface can (Objects, Materials etc.).
- Modify user preferences, style and themes.


Data Access
==================

You can access WEAS's data with the Python API in the same way as the Javascript API.

.. code-block:: python

    from ase.build import molecule
    from weas_widget import WeasWidget
    atoms = molecule("C2H6SO")
    viewer = WeasWidget()
    viewer.from_ase(atoms)
    viewer

In another cell, you can access the data:

.. code-block:: python

    viewer.data.objects

Accessing Attributes
--------------------

.. code-block:: python

    viewer.data.objects[0].position

Data Removal
======================
You can create and remove data in the same way as the Javascript API.

.. code-block:: python

    viewer.data.objects.remove(viewer.data.objects[0])

One can not create a new object with the Python API yet. So this will not work:
.. code-block:: python

    viewer.data.objects.new()

Instead, you need to use operations to create new objects. For example, to create a new sphere:

.. code-block:: python

    viewer.ops.mesh.add_sphere()

Context
======================
The Python API does not have a context manager yet. So this will not work:

.. code-block:: python

    viewer.context.selected_object


Operations
======================
Operators are tools generally accessed by the user from buttons, menu items or key shortcuts. From the user perspective they are a tool but Python can run these with its own settings through the bpy.ops module.

Examples:

.. code-block:: python

    viewer.ops.mesh.add_sphere()
