import anywidget
import traitlets as tl
import os
from .utils import ASE_Adapter, Pymatgen_Adapter

esm_path = os.path.join(os.path.dirname(__file__), """index.js""")
# css_path = os.path.join(os.path.dirname(__file__), """style.css""")
css_path = "https://unpkg.com/weas/dist/style.css"


class WeasWidget(anywidget.AnyWidget):
    _esm = esm_path
    _css = css_path
    # atoms can be a dictionary or a list of dictionaries
    atoms = tl.Union([tl.Dict({}), tl.List(tl.Dict({}))]).tag(sync=True)
    selectedAtomsIndices = tl.List([]).tag(sync=True)
    boundary = tl.List([[0, 1], [0, 1], [0, 1]]).tag(sync=True)
    modelStyle = tl.Int(0).tag(sync=True)
    colorType = tl.Unicode("CPK").tag(sync=True)
    materialType = tl.Unicode("Standard").tag(sync=True)
    atomLabelType = tl.Unicode("None").tag(sync=True)
    showCell = tl.Bool(True).tag(sync=True)
    showBondedAtoms = tl.Bool(False).tag(sync=True)
    _drawModels = tl.Bool(False).tag(sync=True)
    atomScales = tl.List([]).tag(sync=True)
    modelSticks = tl.List([]).tag(sync=True)
    modelPolyhedras = tl.List([]).tag(sync=True)
    volumetricData = tl.Dict({"values": [[[]]]}).tag(sync=True)
    isoSettings = tl.List([]).tag(sync=True)
    imageData = tl.Unicode("").tag(sync=True)
    _exportImage = tl.Bool(False).tag(sync=True)
    _downloadImage = tl.Bool(False).tag(sync=True)
    _imageFileName = tl.Unicode("atomistic-model.png").tag(sync=True)

    def drawModels(self):
        """Redraw the widget."""
        self._drawModels = not self._drawModels

    def set_atoms(self, atoms):
        self.atoms = atoms
        # initialize atomScales
        natom = (
            len(atoms["speciesArray"])
            if isinstance(atoms, dict)
            else len(atoms[0]["speciesArray"])
        )
        self.atomScales = [1] * natom
        self.modelSticks = [0] * natom
        self.modelPolyhedras = [0] * natom

    def from_ase(self, atoms):
        self.set_atoms(ASE_Adapter.to_weas(atoms))

    def to_ase(self):
        return ASE_Adapter.to_ase(self.atoms)

    def from_pymatgen(self, structure):
        self.set_atoms(Pymatgen_Adapter.to_weas(structure))

    def to_pymatgen(self):
        return Pymatgen_Adapter.to_pymatgen(self.atoms)

    def load_example(self, name="tio2.cif"):
        from ase.io import read

        atoms = read(os.path.join(os.path.dirname(__file__), f"datas/{name}"))
        self.set_atoms(ASE_Adapter.to_weas(atoms))

    def export_image(self):
        self._exportImage = not self._exportImage

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

    def download_image(self, imageFileName="atomistic-model.png"):
        self._imageFileName = imageFileName
        self._downloadImage = not self._downloadImage
