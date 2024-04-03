import importlib.metadata
import pathlib
import anywidget
import traitlets as tl
from .utils import ASE_Adapter, Pymatgen_Adapter, load_online_example
import time
import threading

try:
    __version__ = importlib.metadata.version("weas_widget")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"


class WeasWidget(anywidget.AnyWidget):
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
    # mesh primitives
    meshPrimitives = tl.List(tl.Dict({})).tag(sync=True)
    # viewer
    viewerStyle = tl.Dict({}).tag(sync=True)
    # camera
    cameraSetting = tl.Dict({}).tag(sync=True)
    # task
    js_task = tl.Dict({}).tag(sync=True)
    debug = tl.Bool(False).tag(sync=True)

    def __init__(self, from_ase=None, from_pymatgen=None, **kwargs):
        super().__init__(**kwargs)
        if from_ase is not None:
            self.from_ase(from_ase)
        if from_pymatgen is not None:
            self.from_pymatgen(from_pymatgen)

    def send_js_task(self, task):
        """Send a task to the javascript side.
        task is a dictionary with the following keys
        - name: the name of the task
        - kwargs: a dictionary of arguments
        """
        self.js_task = task
        self.js_task = {}

    def drawModels(self):
        """Redraw the widget."""
        self.send_js_task({"name": "drawModels"})

    def set_atoms(self, atoms):
        self.atoms = atoms
        # initialize atomScales
        if isinstance(atoms, list):
            atoms = atoms[0]
        natom = len(atoms["speciesArray"])
        self.atomScales = [1] * natom
        self.modelSticks = [0] * natom
        self.modelPolyhedras = [0] * natom
        # magnetic moment vector field
        # separate spin up and down, add two vector fields
        if "moment" in atoms["attributes"]["atom"]:
            moment = atoms["attributes"]["atom"]["moment"]
            spin_up = [i for i, m in enumerate(moment) if m > 0]
            spin_down = [i for i, m in enumerate(moment) if m < 0]
        if "moment" in atoms["attributes"]["atom"]:
            moment = atoms["attributes"]["atom"]["moment"]
            self.vectorField = [
                {
                    "origins": [atoms["positions"][i] for i in spin_up],
                    "vectors": [[0, 0, moment[i]] for i in spin_up],
                    "color": "blue",
                },
                {
                    "origins": [atoms["positions"][i] for i in spin_down],
                    "vectors": [[0, 0, moment[i]] for i in spin_down],
                    "color": "red",
                },
            ]

    def from_ase(self, atoms):
        self.set_atoms(ASE_Adapter.to_weas(atoms))

    def to_ase(self):
        return ASE_Adapter.to_ase(self.atoms)

    def from_pymatgen(self, structure):
        self.set_atoms(Pymatgen_Adapter.to_weas(structure))

    def to_pymatgen(self):
        return Pymatgen_Adapter.to_pymatgen(self.atoms)

    def load_example(self, name="tio2.cif"):
        atoms = load_online_example(name)
        self.set_atoms(ASE_Adapter.to_weas(atoms))

    def export_image(self, resolutionScale=5):
        self.send_js_task(
            {
                "name": "exportImage",
                "kwargs": {"resolutionScale": resolutionScale},
            }
        )

    def display_image(self):
        from IPython.display import display, Image
        import base64

        if self.imageData == "":
            print(
                "No image data available, please export the image first: running export_image() in another cell."
            )
            return None
        base64_data = self.imageData.split(",")[1]
        # Decode the base64 string
        image_data = base64.b64decode(base64_data)

        # Display the image
        return display(Image(data=image_data))

    def download_image(self, filename="weas-model.png"):
        self.send_js_task(
            {
                "name": "downloadImage",
                "kwargs": {"filename": filename},
            }
        )

    def save_image(self, filename="weas-model.png", resolutionScale=5):
        import base64

        def _save_image():
            while not self.ready:
                time.sleep(0.1)
            self.export_image(resolutionScale)
            # polling mechanism to check if the image data is available
            while not self.imageData:
                time.sleep(0.1)
            base64_data = self.imageData.split(",")[1]
            # Decode the base64 string
            image_data = base64.b64decode(base64_data)
            with open(filename, "wb") as f:
                f.write(image_data)

        thread = threading.Thread(target=_save_image, args=(), daemon=False)
        thread.start()
