class InstancedMeshPrimitive:
    def __init__(self, _widget):
        self._widget = _widget

    @property
    def settings(self):
        return self._widget.instancedMeshPrimitive

    @settings.setter
    def settings(self, value):
        self._widget.instancedMeshPrimitive = value
