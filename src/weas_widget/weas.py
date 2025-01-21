from .base_widget import BaseWidget
from .utils import ASEAdapter, PymatgenAdapter, load_online_example
from .atoms_viewer import AtomsViewer
from .camera import Camera
from .plugins.instanced_mesh_pritimive import InstancedMeshPrimitive
from .operators.ops import Operators
import time
import threading
import ipywidgets as ipw


class WeasWidget(ipw.HBox):
    def __init__(self, from_ase=None, from_pymatgen=None, from_aiida=None, **kwargs):
        self._widget = BaseWidget(**kwargs)
        super().__init__([self._widget])
        self.avr = AtomsViewer(self._widget)
        self.camera = Camera(self._widget)
        self.imp = InstancedMeshPrimitive(self._widget)
        self.ops = Operators(self._widget)
        if from_ase is not None:
            self.from_ase(from_ase)
        if from_pymatgen is not None:
            self.from_pymatgen(from_pymatgen)
        if from_aiida is not None:
            self.from_aiida(from_aiida)

    def from_ase(self, atoms):
        self.avr.atoms = ASEAdapter.to_weas(atoms)

    def to_ase(self):
        return ASEAdapter.to_ase(self.avr.atoms)

    def from_pymatgen(self, structure):
        self.avr.atoms = PymatgenAdapter.to_weas(structure)

    def to_pymatgen(self):
        return PymatgenAdapter.to_pymatgen(self.avr.atoms)

    def from_aiida(self, structure, cell=None):
        from aiida.orm import StructureData, TrajectoryData

        if isinstance(structure, TrajectoryData):
            images = []
            for i in range(structure.numsteps):
                # it is not efficient to get the structure for each step
                # but this way we can use the built-in get_ase() method
                atoms = structure.get_step_structure(i).get_ase()
                if cell is not None:
                    atoms.set_cell(cell)
                images.append(atoms)
            atoms = images
        elif isinstance(structure, StructureData):
            atoms = structure.get_ase()
        else:
            raise ValueError("Input should be either StructureData or TrajectoryData")
        self.from_ase(atoms)

    def to_aiida(self):
        from aiida.orm import StructureData, TrajectoryData
        import numpy as np

        if isinstance(self.avr.atoms, list):
            traj = TrajectoryData()
            cells = np.array(
                [np.array(atoms["cell"]).reshape(3, 3) for atoms in self.avr.atoms]
            )
            traj.set_trajectory(
                stepids=np.array([i + 1 for i in range(len(self.avr.atoms))]),
                symbols=self.avr.atoms[0]["symbols"],
                cells=cells,
                positions=np.array([atoms["positions"] for atoms in self.avr.atoms]),
            )
            return traj
        else:
            return StructureData(ase=self.to_ase())

    def load_example(self, name="tio2.cif"):
        atoms = load_online_example(name)
        self.avr.atoms = ASEAdapter.to_weas(atoms)

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
                "name": "tjs.downloadImage",
                "kwargs": {"filename": filename},
            }
        )

    def save_image(self, filename="weas-model.png", resolutionScale=5, callback=None):
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
            if callback:
                callback(filename)

        thread = threading.Thread(target=_save_image, args=(), daemon=False)
        thread.start()
