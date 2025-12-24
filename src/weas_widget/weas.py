from .base_widget import BaseWidget
from .utils import ASEAdapter, PymatgenAdapter, load_online_example
from .atoms_viewer import AtomsViewer
from .camera import Camera
from .plugins.instanced_mesh_pritimive import InstancedMeshPrimitive
from .plugins.any_mesh import AnyMesh
from .operators.ops import Operators
from .fermi_surface import add_fermi_surface_from_bxsf
import time
import threading
import ipywidgets as ipw
import json
from copy import deepcopy


class WeasWidget(ipw.HBox):
    def __init__(self, from_ase=None, from_pymatgen=None, from_aiida=None, **kwargs):
        self._widget = BaseWidget(**kwargs)
        super().__init__([self._widget])
        self.avr = AtomsViewer(self._widget)
        self.camera = Camera(self._widget)
        self.imp = InstancedMeshPrimitive(self._widget)
        self.any_mesh = AnyMesh(self._widget)
        self.ops = Operators(self._widget)
        if from_ase is not None:
            self.from_ase(from_ase)
        if from_pymatgen is not None:
            self.from_pymatgen(from_pymatgen)
        if from_aiida is not None:
            self.from_aiida(from_aiida)

    @classmethod
    def from_state_file(cls, filename: str, **kwargs):
        widget = cls(**kwargs)
        widget.load_state(filename)
        return widget

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

    def export_state(self) -> dict:
        snapshot = self._request_state_snapshot()
        snapshot["widget"] = self._collect_widget_extras()
        return snapshot

    def import_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            raise ValueError("State must be a dict.")
        self._apply_widget_extras(state)
        self._widget.send_js_task({"name": "importState", "args": [state]})

    def _request_state_snapshot(
        self, timeout: float = 5.0, poll_interval: float = 0.05
    ) -> dict:
        start = time.time()
        while not self._widget.ready:
            if time.time() - start > timeout:
                raise TimeoutError("Timed out waiting for the widget to be ready.")
            time.sleep(poll_interval)
        self._widget.stateSnapshot = {}
        self._widget.send_js_task({"name": "exportState"})
        while True:
            snapshot = self._widget.stateSnapshot
            if snapshot:
                return deepcopy(snapshot)
            if time.time() - start > timeout:
                raise TimeoutError("Timed out waiting for a state snapshot.")
            time.sleep(poll_interval)

    def _collect_widget_extras(self) -> dict:
        return {
            "ui": {
                "viewerStyle": deepcopy(self._widget.viewerStyle),
                "guiConfig": deepcopy(self._widget.guiConfig),
                "logLevel": self._widget.logLevel,
            },
            "viewer": {"showAtomLegend": self._widget.showAtomLegend},
            "plugins": {
                "volumetricData": deepcopy(self._widget.volumetricData),
                "phonon": deepcopy(self._widget.phonon),
            },
        }

    def _apply_widget_extras(self, state: dict) -> None:
        widget = state.get("widget", {})
        ui = widget.get("ui") or state.get("ui") or {}
        viewer = widget.get("viewer") or state.get("viewer") or {}
        plugins = widget.get("plugins") or state.get("plugins") or {}
        if "viewerStyle" in ui:
            self._widget.viewerStyle = deepcopy(ui["viewerStyle"])
        if "guiConfig" in ui:
            self._widget.guiConfig = deepcopy(ui["guiConfig"])
        if "logLevel" in ui:
            self._widget.logLevel = deepcopy(ui["logLevel"])
        if "showAtomLegend" in viewer:
            self._widget.showAtomLegend = deepcopy(viewer["showAtomLegend"])
        if "volumetricData" in plugins:
            self._widget.volumetricData = deepcopy(plugins["volumetricData"])
        if "phonon" in plugins:
            self._widget.phonon = deepcopy(plugins["phonon"])

    def save_state(self, filename: str, callback=None) -> None:
        def _save_state():
            payload = self.export_state()
            with open(filename, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            if callback:
                callback(filename)

        thread = threading.Thread(target=_save_state, args=(), daemon=False)
        thread.start()

    def load_state(self, filename: str) -> None:
        with open(filename, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.import_state(payload)

    def add_fermi_surface_from_bxsf(
        self,
        file_path: str,
        band_index: int = None,
        fermi_energy: float = None,
        supercell_size: tuple = (2, 2, 2),
        drop_periodic: bool = True,
        clip_bz: bool = False,
        show_bz: bool = True,
        show_reciprocal_axes: bool = True,
        combine_bands: bool = True,
        name: str = None,
        color: list = None,
        opacity: float = 0.6,
        material_type: str = "Standard",
    ):
        """Compute Fermi surface meshes from a BXSF file and render via AnyMesh."""
        add_fermi_surface_from_bxsf(
            viewer=self,
            file_path=file_path,
            band_index=band_index,
            fermi_energy=fermi_energy,
            supercell_size=supercell_size,
            drop_periodic=drop_periodic,
            clip_bz=clip_bz,
            show_bz=show_bz,
            show_reciprocal_axes=show_reciprocal_axes,
            combine_bands=combine_bands,
            name=name,
            color=color,
            opacity=opacity,
            material_type=material_type,
        )
