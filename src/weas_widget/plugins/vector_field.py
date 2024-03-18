class VectorField:
    def __init__(self, base_widget):
        self.base_widget = base_widget

    @property
    def settings(self):
        return self.base_widget.vectorField

    @settings.setter
    def settings(self, value):
        self.base_widget.vectorField = value

    @property
    def show(self):
        return self.base_widget.showVectorField

    @show.setter
    def show(self, value):
        self.base_widget.showVectorField = value
