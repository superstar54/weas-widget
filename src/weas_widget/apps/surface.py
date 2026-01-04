"""Module to provide functionality to edit structures."""


import ase
import ipywidgets as ipw
import numpy as np
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
        self.preset_100 = ipw.Button(description="(100)", layout={"width": "64px"})
        self.preset_110 = ipw.Button(description="(110)", layout={"width": "64px"})
        self.preset_111 = ipw.Button(description="(111)", layout={"width": "64px"})
        self.preset_row = ipw.HBox([self.preset_100, self.preset_110, self.preset_111])
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
        self.repeat_x = ipw.IntText(description="Repeat x", value=2)
        self.repeat_y = ipw.IntText(description="Repeat y", value=2)
        self.align_xy = ipw.Dropdown(
            description="Align XY",
            options=["none", "cell_center", "surface_com"],
            value="none",
        )
        self.center_z = ipw.Checkbox(description="Center Z", value=False, indent=False)
        self.z_shift = ipw.FloatText(description="Z shift", value=0.0)
        self.trim_bottom = ipw.IntText(description="Trim bottom", value=0)
        self.trim_top = ipw.IntText(description="Trim top", value=0)
        self.remove_layer = ipw.IntText(description="Remove layer", value=-1)
        self.layer_tol = ipw.FloatText(description="Layer tol (Å)", value=0.1)
        self.clip_enable = ipw.Checkbox(
            description="Clip Z",
            value=False,
            indent=False,
        )
        self.clip_z_min = ipw.FloatText(description="Clip z min", value=0.0)
        self.clip_z_max = ipw.FloatText(description="Clip z max", value=1000.0)
        self.freeze_layers = ipw.IntText(description="Freeze bottom", value=0)
        self.adsorbate_species = ipw.Text(description="Adsorbate", value="CO")
        self.adsorbate_height = ipw.FloatText(description="Height (Å)", value=2.0)
        self.adsorbate_side = ipw.Dropdown(
            description="Side",
            options=["top", "bottom"],
            value="top",
        )
        self.adsorbate_place_mode = ipw.Dropdown(
            description="Placement",
            options=["pymatgen_site", "custom_xy"],
            value="pymatgen_site",
        )
        self.adsorbate_site_type = ipw.Dropdown(
            description="Site type",
            options=["ontop", "bridge", "hollow", "subsurface", "all"],
            value="ontop",
        )
        self.adsorbate_site_index = ipw.IntText(description="Site idx", value=0)
        self._adsorbate_site_count = ipw.HTML("")
        self.adsorbate_x = ipw.FloatText(description="X", value=0.0)
        self.adsorbate_y = ipw.FloatText(description="Y", value=0.0)
        self.adsorbate_add = ipw.Button(description="Add adsorbate")
        self.adsorbate_clear = ipw.Button(description="Clear adsorbates")
        self._adsorbate_list = ipw.HTML("")
        self.auto_update = ipw.Checkbox(
            description="Auto update",
            value=True,
            indent=False,
        )
        self.apply_btn = ipw.Button(description="Apply", button_style="primary")
        self._status_message = ipw.HTML("")
        self._summary = ipw.HTML("")
        surface_box = ipw.VBox(
            children=[
                ipw.HBox([ipw.HTML("Surface indices: "), surface_indices]),
                self.preset_row,
                self.nlayer,
                self.vacuum,
                self.periodic,
            ]
        )
        cell_box = ipw.VBox(
            children=[
                self.repeat_x,
                self.repeat_y,
                self.align_xy,
                self.center_z,
                self.z_shift,
            ]
        )
        termination_box = ipw.VBox(
            children=[
                self.trim_bottom,
                self.trim_top,
                self.remove_layer,
                self.layer_tol,
                self.clip_enable,
                self.clip_z_min,
                self.clip_z_max,
            ]
        )
        constraints_box = ipw.VBox(children=[self.freeze_layers])
        adsorbate_box = ipw.VBox(
            children=[
                self.adsorbate_species,
                self.adsorbate_height,
                self.adsorbate_side,
                self.adsorbate_place_mode,
                self.adsorbate_site_type,
                self.adsorbate_site_index,
                self._adsorbate_site_count,
                self.adsorbate_x,
                self.adsorbate_y,
                ipw.HBox([self.adsorbate_add, self.adsorbate_clear]),
                self._adsorbate_list,
            ]
        )
        accordion = ipw.Accordion(
            children=[
                surface_box,
                cell_box,
                termination_box,
                constraints_box,
                adsorbate_box,
            ]
        )
        accordion.set_title(0, "Surface")
        accordion.set_title(1, "Cell")
        accordion.set_title(2, "Termination")
        accordion.set_title(3, "Constraints")
        accordion.set_title(4, "Adsorbates")
        self.editor = ipw.VBox(
            children=[
                accordion,
                ipw.HBox([self.auto_update, self.apply_btn]),
                self._summary,
                self._status_message,
            ]
        )
        self._adsorbates = []
        self._fixed_indices = []
        self._updating_sites = False

        self.viewer = WeasWidget()
        self.viewer.cameraSetting = {"direction": [0, 1, 0]}

        # Observe changes in the nlayer widget
        self.index_h.observe(self.update_surface, names="value")
        self.index_k.observe(self.update_surface, names="value")
        self.index_l.observe(self.update_surface, names="value")
        self.nlayer.observe(self.update_surface, names="value")
        self.vacuum.observe(self.update_surface, names="value")
        self.periodic.observe(self.update_surface, names="value")
        self.repeat_x.observe(self.update_surface, names="value")
        self.repeat_y.observe(self.update_surface, names="value")
        self.align_xy.observe(self.update_surface, names="value")
        self.center_z.observe(self.update_surface, names="value")
        self.z_shift.observe(self.update_surface, names="value")
        self.trim_bottom.observe(self.update_surface, names="value")
        self.trim_top.observe(self.update_surface, names="value")
        self.remove_layer.observe(self.update_surface, names="value")
        self.layer_tol.observe(self.update_surface, names="value")
        self.clip_enable.observe(self.update_surface, names="value")
        self.clip_z_min.observe(self.update_surface, names="value")
        self.clip_z_max.observe(self.update_surface, names="value")
        self.freeze_layers.observe(self.update_surface, names="value")
        self.adsorbate_place_mode.observe(self.update_surface, names="value")
        self.adsorbate_site_type.observe(self.update_surface, names="value")
        self.adsorbate_site_index.observe(self.update_surface, names="value")
        self.adsorbate_side.observe(self.update_surface, names="value")
        self.adsorbate_height.observe(self.update_surface, names="value")
        self.apply_btn.on_click(self.update_surface)
        self.adsorbate_add.on_click(self._add_adsorbate)
        self.adsorbate_clear.on_click(self._clear_adsorbates)
        self.preset_100.on_click(lambda _btn: self._set_indices(1, 0, 0))
        self.preset_110.on_click(lambda _btn: self._set_indices(1, 1, 0))
        self.preset_111.on_click(lambda _btn: self._set_indices(1, 1, 1))

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

    def _set_indices(self, h, k, l_index):
        self.index_h.value = h
        self.index_k.value = k
        self.index_l.value = l_index

    def _set_status(self, message, is_error=False):
        if not message:
            self._status_message.value = ""
            return
        status = "alert-danger" if is_error else "alert-info"
        self._status_message.value = f"""
        <div class="alert {status}">
        <strong>{message}</strong>
        </div>
        """

    def _update_summary(self, atoms, layer_groups=None):
        if atoms is None:
            self._summary.value = ""
            return
        cell = atoms.get_cell()
        lengths = [f"{x:.3f}" for x in cell.lengths()]
        normal = self._surface_normal(atoms)
        normal_txt = ", ".join(f"{x:.3f}" for x in normal.tolist())
        top_count = bottom_count = 0
        if layer_groups:
            bottom_count = len(layer_groups[0])
            top_count = len(layer_groups[-1])
        ads_count = len(self._adsorbates)
        self._summary.value = (
            f"<div><strong>Atoms:</strong> {len(atoms)}"
            f" <strong>Cell:</strong> [{', '.join(lengths)}] Å</div>"
            f"<div><strong>Normal:</strong> [{normal_txt}]"
            f" <strong>Surface atoms:</strong> top {top_count}, bottom {bottom_count}"
            f" <strong>Adsorbates:</strong> {ads_count}</div>"
        )

    def _surface_normal(self, atoms):
        cell = atoms.get_cell()
        normal = np.cross(cell[0], cell[1])
        norm = np.linalg.norm(normal)
        if norm <= 0:
            return np.array([0.0, 0.0, 1.0])
        return normal / norm

    def _layer_groups(self, atoms, tol):
        if len(atoms) == 0:
            return []
        z = atoms.positions[:, 2]
        order = np.argsort(z)
        groups = []
        current = [int(order[0])]
        z_ref = z[order[0]]
        for idx in order[1:]:
            idx = int(idx)
            if abs(z[idx] - z_ref) <= tol:
                current.append(idx)
            else:
                groups.append(current)
                current = [idx]
                z_ref = z[idx]
        groups.append(current)
        return groups

    def _cell_center_xy(self, atoms):
        cell = atoms.get_cell()
        center = 0.5 * (cell[0] + cell[1])
        return np.array([center[0], center[1]], dtype=float)

    def _surface_com_xy(self, atoms, layer_groups):
        if not layer_groups:
            return np.zeros(2, dtype=float)
        top = layer_groups[-1]
        positions = atoms.positions[top]
        return np.array([positions[:, 0].mean(), positions[:, 1].mean()], dtype=float)

    def _apply_alignment(self, atoms, layer_groups):
        if self.align_xy.value == "cell_center":
            target = self._cell_center_xy(atoms)
            current = atoms.get_center_of_mass()[:2]
            shift = target - current
            atoms.positions[:, 0] += shift[0]
            atoms.positions[:, 1] += shift[1]
        elif self.align_xy.value == "surface_com":
            target = self._cell_center_xy(atoms)
            current = self._surface_com_xy(atoms, layer_groups)
            shift = target - current
            atoms.positions[:, 0] += shift[0]
            atoms.positions[:, 1] += shift[1]
        if self.center_z.value:
            atoms.center(axis=2)
        if self.z_shift.value != 0.0:
            atoms.positions[:, 2] += self.z_shift.value

    def _apply_constraints(self, atoms, layer_groups):
        from ase.constraints import FixAtoms

        self._fixed_indices = []
        if self.freeze_layers.value > 0 and layer_groups:
            if self.freeze_layers.value > len(layer_groups):
                self._set_status("Freeze layers exceeds total layers.", is_error=True)
                return False
            frozen = []
            for group in layer_groups[: self.freeze_layers.value]:
                frozen.extend(group)
            self._fixed_indices = sorted(set(int(i) for i in frozen))
            atoms.set_constraint(FixAtoms(indices=self._fixed_indices))
        else:
            atoms.set_constraint(None)
        return True

    def _update_adsorbate_list(self):
        if not self._adsorbates:
            self._adsorbate_list.value = ""
            return
        rows = []
        for idx, ads in enumerate(self._adsorbates, start=1):
            if ads["place_mode"] == "pymatgen_site":
                rows.append(
                    f"{idx}. {ads['species']} ({ads['side']}, {ads['site_type']}, "
                    f"idx={ads['site_index']}, h={ads['height']:.2f})"
                )
            else:
                rows.append(
                    f"{idx}. {ads['species']} ({ads['side']}, custom, "
                    f"x={ads['x']:.2f}, y={ads['y']:.2f}, h={ads['height']:.2f})"
                )
        self._adsorbate_list.value = "<div>" + "<br/>".join(rows) + "</div>"

    def _add_adsorbate(self, _btn):
        species = self.adsorbate_species.value.strip()
        if not species:
            self._set_status("Adsorbate species is required.", is_error=True)
            return
        entry = {
            "species": species,
            "height": float(self.adsorbate_height.value),
            "side": self.adsorbate_side.value,
            "place_mode": self.adsorbate_place_mode.value,
            "site_type": self.adsorbate_site_type.value,
            "site_index": int(self.adsorbate_site_index.value),
            "x": float(self.adsorbate_x.value),
            "y": float(self.adsorbate_y.value),
        }
        self._adsorbates.append(entry)
        self._update_adsorbate_list()
        self.update_surface()

    def _clear_adsorbates(self, _btn):
        self._adsorbates.clear()
        self._update_adsorbate_list()
        self.update_surface()

    def _build_adsorbate(self, species):
        from ase.build import molecule

        try:
            return molecule(species)
        except Exception:
            return ase.Atoms([ase.Atom(species)])

    def _adsorbate_xy(self, atoms, layer_groups, entry):
        if entry["place_mode"] == "custom_xy":
            return np.array([entry["x"], entry["y"]], dtype=float)
        return self._cell_center_xy(atoms)

    def _find_adsorbate_sites(self, atoms, side, distance, site_type):
        from pymatgen.analysis.adsorption import AdsorbateSiteFinder
        from pymatgen.io.ase import AseAtomsAdaptor

        structure = AseAtomsAdaptor.get_structure(atoms)
        normal = self._surface_normal(atoms)
        if side == "bottom":
            normal = -normal
        positions = (site_type,)
        if site_type == "all":
            positions = ("ontop", "bridge", "hollow", "subsurface")
        finder = AdsorbateSiteFinder(structure, mi_vec=normal.tolist())
        finder.mvec = np.array(normal, dtype=float)
        sites = finder.find_adsorption_sites(distance=distance, positions=positions)
        if site_type == "all":
            return sites.get("all", [])
        return sites.get(site_type, [])

    def _update_adsorbate_sites_preview(self, atoms):
        if atoms is None or len(atoms) == 0:
            self._adsorbate_site_count.value = ""
            return
        if self.adsorbate_place_mode.value != "pymatgen_site":
            self._adsorbate_site_count.value = ""
            return
        if self._updating_sites:
            return
        self._updating_sites = True
        try:
            counts = {}
            for site_type in ["ontop", "bridge", "hollow", "subsurface"]:
                sites = self._find_adsorbate_sites(
                    atoms,
                    self.adsorbate_side.value,
                    float(self.adsorbate_height.value),
                    site_type,
                )
                counts[site_type] = len(sites)
        except Exception as exc:
            self._adsorbate_site_count.value = (
                f"<div><strong>Sites:</strong> error ({exc})</div>"
            )
            self._updating_sites = False
            return
        available = [k for k, v in counts.items() if v > 0]
        options = available + (["all"] if available else [])
        if not options:
            self._adsorbate_site_type.options = ["all"]
            self._adsorbate_site_type.value = "all"
            self._adsorbate_site_count.value = "<div><strong>Sites:</strong> 0</div>"
            self._updating_sites = False
            return
        self.adsorbate_site_type.options = options
        if self.adsorbate_site_type.value not in options:
            self.adsorbate_site_type.value = options[0]
        chosen = self.adsorbate_site_type.value
        total = sum(counts.values())
        if chosen == "all":
            count_txt = f"{total} (all)"
        else:
            count_txt = f"{counts[chosen]} ({chosen})"
        detail = ", ".join(f"{k}:{v}" for k, v in counts.items() if v > 0)
        self._adsorbate_site_count.value = (
            f"<div><strong>Sites:</strong> {count_txt}</div>"
            f"<div><strong>Available:</strong> {detail}</div>"
        )
        self._updating_sites = False

    def _apply_adsorbates(self, atoms, layer_groups):
        if not self._adsorbates:
            return atoms
        if len(atoms) == 0:
            return atoms
        base_atoms = atoms.copy()
        z_min = base_atoms.positions[:, 2].min()
        z_max = base_atoms.positions[:, 2].max()
        site_cache = {}
        for entry in self._adsorbates:
            ads = self._build_adsorbate(entry["species"])
            if entry["place_mode"] == "pymatgen_site":
                cache_key = (
                    entry["side"],
                    entry["height"],
                    entry["site_type"],
                )
                if cache_key not in site_cache:
                    site_cache[cache_key] = self._find_adsorbate_sites(
                        base_atoms,
                        entry["side"],
                        float(entry["height"]),
                        entry["site_type"],
                    )
                sites = site_cache[cache_key]
                if not sites:
                    self._set_status("No adsorption sites found.", is_error=True)
                    continue
                idx = entry["site_index"]
                if idx < 0 or idx >= len(sites):
                    self._set_status(
                        "Adsorbate site index out of range.", is_error=True
                    )
                    continue
                from pymatgen.analysis.adsorption import AdsorbateSiteFinder
                from pymatgen.io.ase import AseAtomsAdaptor

                structure = AseAtomsAdaptor.get_structure(atoms)
                normal = self._surface_normal(atoms)
                if entry["side"] == "bottom":
                    normal = -normal
                finder = AdsorbateSiteFinder(structure, mi_vec=normal.tolist())
                finder.mvec = np.array(normal, dtype=float)
                mol = AseAtomsAdaptor.get_molecule(ads)
                try:
                    struct_with_ads = finder.add_adsorbate(
                        mol,
                        np.array(sites[idx], dtype=float),
                    )
                except Exception as exc:
                    self._set_status(f"Add adsorbate failed: {exc}", is_error=True)
                    continue
                if "tags" in struct_with_ads.site_properties:
                    struct_with_ads.site_properties.pop("tags", None)
                    for site in struct_with_ads:
                        if "tags" in site.properties:
                            del site.properties["tags"]
                atoms = AseAtomsAdaptor.get_atoms(struct_with_ads)
            else:
                target_xy = self._adsorbate_xy(atoms, layer_groups, entry)
                ads_com = ads.get_center_of_mass()[:2]
                ads.positions[:, 0] += target_xy[0] - ads_com[0]
                ads.positions[:, 1] += target_xy[1] - ads_com[1]
                ads_min_z = ads.positions[:, 2].min()
                ads_max_z = ads.positions[:, 2].max()
                if entry["side"] == "bottom":
                    target_z = z_min - entry["height"]
                    ads.positions[:, 2] += target_z - ads_max_z
                else:
                    target_z = z_max + entry["height"]
                    ads.positions[:, 2] += target_z - ads_min_z
                atoms += ads
        return atoms

    def _update_fixed_highlight(self):
        self.viewer.avr.highlight.settings["fixed"] = {
            "type": "crossView",
            "indices": list(self._fixed_indices),
            "scale": 1.0,
        }
        self.viewer.avr.draw()

    @tl.observe("bulk")
    def update_surface(self, change=None):
        """Apply the transformation matrix to the structure."""
        from ase.build import surface

        if change is not None and getattr(change, "new", None) is None:
            pass
        if (
            change is not None
            and getattr(change, "name", None) == "value"
            and not self.auto_update.value
        ):
            return

        # only update structure when atoms is not None.
        if self.bulk is not None:
            indices = [self.index_h.value, self.index_k.value, self.index_l.value]
            if indices == [0, 0, 0]:
                self._set_status("Surface indices cannot be all zero.", is_error=True)
                return
            if self.nlayer.value < 1:
                self._set_status("Layers must be at least 1.", is_error=True)
                return
            if self.repeat_x.value < 1 or self.repeat_y.value < 1:
                self._set_status("Repeat values must be at least 1.", is_error=True)
                return
            if self.trim_bottom.value < 0 or self.trim_top.value < 0:
                self._set_status("Trim values must be >= 0.", is_error=True)
                return
            if self.layer_tol.value <= 0:
                self._set_status("Layer tolerance must be > 0.", is_error=True)
                return
            if self.freeze_layers.value < 0:
                self._set_status("Freeze layers must be >= 0.", is_error=True)
                return
            if self.remove_layer.value < -1:
                self._set_status("Remove layer must be -1 or >= 0.", is_error=True)
                return
            try:
                atoms = surface(
                    self.bulk,
                    indices,
                    layers=self.nlayer.value,
                    vacuum=self.vacuum.value,
                    periodic=self.periodic.value,
                )
                if self.repeat_x.value > 1 or self.repeat_y.value > 1:
                    atoms = atoms.repeat((self.repeat_x.value, self.repeat_y.value, 1))
            except Exception as e:
                self._set_status(
                    f"Surface build failed: {e}",
                    is_error=True,
                )
                return
            groups = self._layer_groups(atoms, self.layer_tol.value)
            if not groups:
                self._set_status("No layers found for surface.", is_error=True)
                return
            total_layers = len(groups)
            trim_bottom = self.trim_bottom.value
            trim_top = self.trim_top.value
            if trim_bottom + trim_top >= total_layers:
                self._set_status("Trim removes all layers.", is_error=True)
                return
            start = trim_bottom
            end = total_layers - trim_top
            groups = groups[start:end]
            if self.remove_layer.value >= 0:
                if self.remove_layer.value >= len(groups):
                    self._set_status("Remove layer index out of range.", is_error=True)
                    return
                del groups[self.remove_layer.value]
            if not groups:
                self._set_status("All layers removed.", is_error=True)
                return
            keep = sorted([idx for group in groups for idx in group])
            atoms = atoms[keep]
            if self.clip_enable.value:
                z_min = self.clip_z_min.value
                z_max = self.clip_z_max.value
                if z_min >= z_max:
                    self._set_status("Clip z min must be < z max.", is_error=True)
                    return
                z = atoms.positions[:, 2]
                mask = (z >= z_min) & (z <= z_max)
                if not np.any(mask):
                    self._set_status("Clip removed all atoms.", is_error=True)
                    return
                atoms = atoms[mask]
            groups = self._layer_groups(atoms, self.layer_tol.value)
            self._apply_alignment(atoms, groups)
            self._update_adsorbate_sites_preview(atoms)
            atoms = self._apply_adsorbates(atoms, groups)
            if not self._apply_constraints(atoms, groups):
                return
            self.structure = atoms
            self._set_status("")
            self._update_summary(atoms, groups)
            self._update_fixed_highlight()
        else:
            self._set_status("No bulk structure loaded.", is_error=True)

    @tl.observe("structure")
    def _observe_structure(self, change):
        if self.structure is not None:
            self.viewer.from_ase(self.structure)
            self._update_fixed_highlight()
