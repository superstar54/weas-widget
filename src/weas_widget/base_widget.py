import importlib.metadata
import pathlib
import anywidget
import traitlets as tl

try:
    __version__ = importlib.metadata.version("weas_widget")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"


class BaseWidget(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "static" / "widget.js"
    _css = pathlib.Path(__file__).parent / "static" / "widget.css"

    # indicate if the widget is displayed and available for interaction.
    ready = tl.Bool(False).tag(sync=True)
    # atoms can be a dictionary or a list of dictionaries
    atoms = tl.Union([tl.Dict({}), tl.List(tl.Dict({}))]).tag(sync=True)
    selectedAtomsIndices = tl.List([]).tag(sync=True)
    boundary = tl.List([[0, 1], [0, 1], [0, 1]]).tag(sync=True)
    modelStyle = tl.Int(0).tag(sync=True)
    # color
    colorBy = tl.Unicode("Element").tag(sync=True)
    colorType = tl.Unicode("CPK").tag(sync=True)
    colorRamp = tl.List(["red", "blue"]).tag(sync=True)
    # material
    materialType = tl.Unicode("Standard").tag(sync=True)
    atomLabelType = tl.Unicode("None").tag(sync=True)
    showCell = tl.Bool(True).tag(sync=True)
    showBondedAtoms = tl.Bool(False).tag(sync=True)
    atomScales = tl.List([]).tag(sync=True)
    modelSticks = tl.List([]).tag(sync=True)
    modelPolyhedras = tl.List([]).tag(sync=True)
    volumetricData = tl.Dict({"values": [[[]]]}).tag(sync=True)
    isoSettings = tl.List([]).tag(sync=True)
    imageData = tl.Unicode("").tag(sync=True)
    vectorField = tl.List().tag(sync=True)
    showVectorField = tl.Bool(True).tag(sync=True)
    guiConfig = tl.Dict({}).tag(sync=True)
    # instanced mesh primitives
    instancedMeshPrimitive = tl.List(tl.Dict({})).tag(sync=True)
    # any mesh
    anyMesh = tl.List(tl.Dict({})).tag(sync=True)
    # viewer
    viewerStyle = tl.Dict({}).tag(sync=True)
    # camera
    cameraSetting = tl.Dict({}).tag(sync=True)
    cameraZoom = tl.Float().tag(sync=True)
    cameraPosition = tl.List().tag(sync=True)
    cameraRotation = tl.List().tag(sync=True)
    cameraLookAt = tl.List().tag(sync=True)
    # task
    js_task = tl.Dict({}).tag(sync=True)
    python_task = tl.Dict({}).tag(sync=True)
    debug = tl.Bool(False).tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def send_js_task(self, task):
        """Send a task to the javascript side.
        task is a dictionary with the following keys
        - name: the name of the task
        - kwargs: a dictionary of arguments
        """
        self.js_task = {}
        self.js_task = task

    def drawModels(self):
        """Redraw the widget."""
        self.send_js_task({"name": "drawModels"})
