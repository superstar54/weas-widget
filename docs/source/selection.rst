

=======================
Selection
=======================

- `Click` the atom to select it. Click again to deselect.
- `Shift + drag` to select multiple atoms with a box. Support accumulation of selection.
- `Shift + Alt + drag` to select multiple atoms with a lasso. Support accumulation of selection.

Group selection (agent tools)
-----------------------------
You can tag atoms into named groups and select them by group name when using the
agent toolkit.

Python operations also expose group selection:

.. code-block:: python

   viewer.ops.selection.select_by_group(group = "molecule")
