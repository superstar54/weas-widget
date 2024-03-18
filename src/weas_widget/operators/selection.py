"""
# weas/src/operation/selection.js
This module contains the Transform class which is used to perform
SelectAll
InvertSelection
InsideSelection
"""


class SelectionOperation:
    def __init__(self, base_widget):
        self.base_widget = base_widget

    def select_all(self, **kwargs):
        self.base_widget.send_js_task(
            {"name": "ops.selection.SelectAll", "kwargs": kwargs}
        )

    def invert_selection(self, **kwargs):
        self.base_widget.send_js_task(
            {"name": "ops.selection.InvertSelection", "kwargs": kwargs}
        )

    def inside_selection(self, **kwargs):
        self.base_widget.send_js_task(
            {"name": "ops.selection.InsideSelection", "kwargs": kwargs}
        )
