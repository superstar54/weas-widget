class Isosurface:
    def __init__(self, base_widget):
        self.base_widget = base_widget

    @property
    def volumetric_data(self):
        return self.base_widget.volumetricData

    @volumetric_data.setter
    def volumetric_data(self, value):
        self.base_widget.volumetricData = value

    @property
    def settings(self):
        return self.base_widget.isoSettings

    @settings.setter
    def settings(self, value):
        self.base_widget.isoSettings = value
