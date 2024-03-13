"""Module to provide functionality to edit structures."""


import ase
import ipywidgets as ipw
import traitlets as tl
from weas_widget import WeasWidget


class SurfaceBuilder(ipw.VBox):
    """Widget that allows for cut surface slab."""

    structure = tl.Instance(ase.Atoms, allow_none=True)
    bulk = tl.Instance(ase.Atoms, allow_none=True)

    def __init__(self, title="Surface editor"):
        self.title = title

        self.index_h = ipw.IntText(value=1, layout={"width": "60px"})
        self.index_k = ipw.IntText(value=1, layout={"width": "60px"})
        self.index_l = ipw.IntText(value=1, layout={"width": "60px"})
        surface_indices = ipw.HBox([self.index_h, self.index_k, self.index_l])
        self.nlayer = ipw.IntText(
            description="Layers",
            value=3,
        )
        self.vacuum = ipw.FloatSlider(
            description="Vacuum (Å)",
            value=5,
            max=10,
            min=0,
        )
        self.periodic = ipw.Checkbox(
            description="Periodic",
            value=False,
            indent=False,
        )
        self.editor = ipw.VBox(
            children=[
                ipw.HBox([ipw.HTML("Surface indices: "), surface_indices]),
                self.nlayer,
                self.vacuum,
                self.periodic,
            ]
        )

        self.viewer = WeasWidget()
        self.viewer.cameraSetting = {"direction": [0, 1, 0]}

        # Observe changes in the nlayer widget
        self.index_h.observe(self.update_surface, names="value")
        self.index_k.observe(self.update_surface, names="value")
        self.index_l.observe(self.update_surface, names="value")
        self.nlayer.observe(self.update_surface, names="value")
        self.vacuum.observe(self.update_surface, names="value")
        self.periodic.observe(self.update_surface, names="value")

        super().__init__(
            children=[
                ipw.HBox(
                    [
                        self.viewer,
                        self.editor,
                    ]
                )
            ],
        )

    @tl.observe("bulk")
    def update_surface(self, change=None):
        """Apply the transformation matrix to the structure."""
        from ase.build import surface

        # only update structure when atoms is not None.
        if self.bulk is not None:
            indices = [self.index_h.value, self.index_k.value, self.index_l.value]
            try:
                atoms = surface(
                    self.bulk,
                    indices,
                    layers=self.nlayer.value,
                    vacuum=self.vacuum.value,
                    periodic=self.periodic.value,
                )
            except Exception as e:
                self._status_message.message = """
            <div class="alert alert-info">
            <strong>The transformation matrix is wrong! {}</strong>
            </div>
            """.format(
                    e
                )
                return
            # translate
            self.structure = atoms

    @tl.observe("structure")
    def _observe_structure(self, change):
        if self.structure is not None:
            self.viewer.from_ase(self.structure)
