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

    def replace(self, symbol: str):
        self.base_widget.send_js_task(
            {"name": "ops.atoms.ReplaceOperation", "kwargs": {"symbol": symbol}}
        )

    def add_atom(self, symbol: str, position: list[float] | dict = [0, 0, 0]):
        if isinstance(position, list):
            position = {"x": position[0], "y": position[1], "z": position[2]}
        self.base_widget.send_js_task(
            {
                "name": "ops.atoms.AddAtomOperation",
                "kwargs": {"symbol": symbol, "position": position},
            }
        )

    def color_by_attribute(
        self,
        attribute: str = "Element",
        color1: str = "#ff0000",
        color2: str = "#0000ff",
    ):
        self.base_widget.send_js_task(
            {
                "name": "ops.atoms.ColorByAttribute",
                "kwargs": {"attribute": attribute, "color1": color1, "color2": color2},
            }
        )
