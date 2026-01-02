from __future__ import annotations

"""
# weas/src/operation/transform.js
This module contains the Transform class which is used to perform
TranslateOperation
RotateOperation
ScaleOperation
"""


class TransformOperation:
    def __init__(self, base_widget):
        self.base_widget = base_widget

    def translate(self, vector: list[float] | dict = [0, 0, 0]):
        self.base_widget.send_js_task(
            {"name": "ops.transform.TranslateOperation", "kwargs": {"vector": vector}}
        )

    def rotate(self, axis: list[float] | dict = [0, 0, 1], angle: float = 90.0):
        self.base_widget.send_js_task(
            {
                "name": "ops.transform.RotateOperation",
                "kwargs": {"axis": axis, "angle": angle},
            }
        )

    def scale(self, scale: list[float] | dict = [1.0, 1.0, 1.0]):
        self.base_widget.send_js_task(
            {"name": "ops.transform.ScaleOperation", "kwargs": {"scale": scale}}
        )
