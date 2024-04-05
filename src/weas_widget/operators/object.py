"""
DeleteOperation
CopyOperation
"""


class ObjectOperation:
    def __init__(self, base_widget):
        self.base_widget = base_widget

    def delete(self, **kwargs):
        self.base_widget.send_js_task(
            {"name": "ops.object.DeleteOperation", "kwargs": kwargs}
        )

    def copy(self, **kwargs):
        self.base_widget.send_js_task(
            {"name": "ops.object.CopyOperation", "kwargs": kwargs}
        )
