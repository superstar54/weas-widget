from .base_widget import BaseWidget
from .utils import ASE_Adapter, Pymatgen_Adapter, load_online_example
from .data import Data
from .atoms_viewer import AtomsViewer
from .camera import Camera
from .plugins.instanced_mesh_pritimive import InstancedMeshPrimitive
import time
import threading


class WeasWidget:
    def __init__(self, **kwargs):
        self._widget = BaseWidget(**kwargs)
        self.avr = AtomsViewer(self._widget)
        self.data = Data(self._widget)
        self.camera = Camera(self._widget)
        self.imp = InstancedMeshPrimitive(self._widget)

    def _repr_mimebundle_(self, *args, **kwargs):
        return self._widget._repr_mimebundle_(*args, **kwargs)

    def from_ase(self, atoms):
        self.avr.atoms = ASE_Adapter.to_weas(atoms)

    def to_ase(self):
        return ASE_Adapter.to_ase(self.avr.atoms)

    def from_pymatgen(self, structure):
        self.avr.atoms = Pymatgen_Adapter.to_weas(structure)

    def to_pymatgen(self):
        return Pymatgen_Adapter.to_pymatgen(self.avr.atoms)

    def load_example(self, name="tio2.cif"):
        atoms = load_online_example(name)
        self.avr.atoms = ASE_Adapter.to_weas(atoms)

    def export_image(self, resolutionScale=5):
        self._widget.send_js_task(
            {
                "name": "exportImage",
                "kwargs": {"resolutionScale": resolutionScale},
            }
        )

    def display_image(self):
        from IPython.display import display, Image
        import base64

        if self._widget.imageData == "":
            print(
                "No image data available, please export the image first: running export_image() in another cell."
            )
            return None
        base64_data = self._widget.imageData.split(",")[1]
        # Decode the base64 string
        image_data = base64.b64decode(base64_data)

        # Display the image
        return display(Image(data=image_data))

    def download_image(self, filename="weas-model.png"):
        self._widget.send_js_task(
            {
                "name": "downloadImage",
                "kwargs": {"filename": filename},
            }
        )

    def save_image(self, filename="weas-model.png", resolutionScale=5):
        import base64

        def _save_image():
            while not self._widget.ready:
                time.sleep(0.1)
            self.export_image(resolutionScale)
            # polling mechanism to check if the image data is available
            while not self._widget.imageData:
                time.sleep(0.1)
            base64_data = self._widget.imageData.split(",")[1]
            # Decode the base64 string
            image_data = base64.b64decode(base64_data)
            with open(filename, "wb") as f:
                f.write(image_data)

        thread = threading.Thread(target=_save_image, args=(), daemon=False)
        thread.start()
