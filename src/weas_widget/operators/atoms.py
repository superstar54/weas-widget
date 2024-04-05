"""
# weas/src/operation/atoms.js
This module contains the Transform class which is used to perform
ReplaceOperation
AddAtomOperation
ColorByAttribute
"""


class AtomsOperation:
    def __init__(self, base_widget):
        self.base_widget = base_widget

    def replace(self, **kwargs):
        self.base_widget.send_js_task(
            {"name": "ops.atoms.ReplaceOperation", "kwargs": kwargs}
        )

    def add_atom(self, **kwargs):
        self.base_widget.send_js_task(
            {"name": "ops.atoms.AddAtomOperation", "kwargs": kwargs}
        )

    def color_by_attribute(self, **kwargs):
        self.base_widget.send_js_task(
            {"name": "ops.atoms.ColorByAttribute", "kwargs": kwargs}
        )

    def add_molecule(self, *args, **kwargs):
        from ase.build import molecule
        from weas_widget.utils import ASE_Adapter

        mol = molecule(*args, **kwargs)
        atoms = ASE_Adapter.to_weas(mol)
        self.base_widget.atoms = atoms
