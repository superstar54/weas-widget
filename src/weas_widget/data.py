class Data:
    def __init__(self, weas_viewer):
        self.weas_viewer = weas_viewer
        self.objects = [Object("InitialObject")]
        self.geometries = [Geometry("InitialGeometry")]


class Object:
    def __init__(self, name):
        self.name = name


class Geometry:
    def __init__(self, name):
        self.name = name
