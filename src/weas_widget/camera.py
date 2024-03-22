class Camera:
    def __init__(self, _widget):
        self._widget = _widget

    @property
    def setting(self):
        return self._widget.cameraSetting

    @setting.setter
    def setting(self, value):
        self._widget.cameraSetting = value
