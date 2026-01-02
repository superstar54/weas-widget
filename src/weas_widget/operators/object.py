"""
DeleteOperation
CopyOperation
"""


class ObjectOperation:
    def __init__(self, base_widget):
        self.base_widget = base_widget

    def delete(self, indices: list[int] | None = None):
        self.base_widget.send_js_task(
            {"name": "ops.object.DeleteOperation", "kwargs": {"indices": indices}}
        )

    def copy(self, indices: list[int] | None = None):
        self.base_widget.send_js_task(
            {"name": "ops.object.CopyOperation", "kwargs": {"indices": indices}}
        )
