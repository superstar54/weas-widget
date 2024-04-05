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

    def translate(self, **kwargs):
        self.base_widget.send_js_task(
            {"name": "ops.transform.TranslateOperation", "kwargs": kwargs}
        )

    def rotate(self, **kwargs):
        self.base_widget.send_js_task(
            {"name": "ops.transform.RotateOperation", "kwargs": kwargs}
        )

    def scale(self, **kwargs):
        self.base_widget.send_js_task(
            {"name": "ops.transform.ScaleOperation", "kwargs": kwargs}
        )
