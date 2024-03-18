import ipywidgets as ipw
from .base_widget import BaseWidget
from .utils import ASE_Adapter, Pymatgen_Adapter, load_online_example
from .data import Data
from .operators.ops import Ops
from .atoms_viewer import AtomsViewer
import time
import threading


class WeasWidget(ipw.VBox):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_widget = BaseWidget()
        self.avr = AtomsViewer(self.base_widget)
        self.data = Data(self.base_widget)
        self.ops = Ops(self.base_widget)
        self.children = [self.base_widget]

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
        self.base_widget.send_js_task(
            {
                "name": "exportImage",
                "kwargs": {"resolutionScale": resolutionScale},
            }
        )

    def display_image(self):
        from IPython.display import display, Image
        import base64

        if self.base_widget.imageData == "":
            print(
                "No image data available, please export the image first: running export_image() in another cell."
            )
            return None
        base64_data = self.base_widget.imageData.split(",")[1]
        # Decode the base64 string
        image_data = base64.b64decode(base64_data)

        # Display the image
        return display(Image(data=image_data))

    def download_image(self, filename="weas-model.png"):
        self.base_widget.send_js_task(
            {
                "name": "downloadImage",
                "kwargs": {"filename": filename},
            }
        )

    def save_image(self, filename="weas-model.png", resolutionScale=5):
        import base64

        def _save_image():
            while not self.base_widget.ready:
                time.sleep(0.1)
            self.export_image(resolutionScale)
            # polling mechanism to check if the image data is available
            while not self.base_widget.imageData:
                time.sleep(0.1)
            base64_data = self.base_widget.imageData.split(",")[1]
            # Decode the base64 string
            image_data = base64.b64decode(base64_data)
            with open(filename, "wb") as f:
                f.write(image_data)

        thread = threading.Thread(target=_save_image, args=(), daemon=False)
        thread.start()
