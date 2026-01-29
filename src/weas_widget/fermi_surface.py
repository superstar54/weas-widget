from typing import Dict, Iterable, List, Tuple, Optional, Union
import colorsys
import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BXSFBand:
    index: int
    grid_shape: Tuple[int, int, int]  # (nx, ny, nz)
    origin: np.ndarray  # shape (3,)
    energies: np.ndarray  # shape (nx, ny, nz)


@dataclass
class BXSFData:
    fermi_energy: float
    rec_lattice: np.ndarray  # shape (3, 3), rows are vectors
    bands: List[BXSFBand]


class BXSFParseError(RuntimeError):
    pass


def parse_bxsf(path: Union[str, Path]) -> BXSFData:
    """
    Parse an XCrySDen .bxsf file (band energy on a uniform k-grid).

    Returns
    -------
    BXSFData with:
      - fermi_energy (float)
      - rec_lattice (3x3 numpy array)
      - bands: list of BXSFBand, each with:
          - grid_shape (nx, ny, nz)
          - origin (3,)
          - energies (nx, ny, nz) float array

    Notes
    -----
    - Energies are read in the order stored in the file and reshaped to (nx, ny, nz).
    - The .bxsf format commonly stores energies with the x-index varying fastest.
      This parser reshapes with C-order to match that convention.
    - Some .bxsf files include duplicated endpoints (periodic wrap). This parser does
      not remove duplicates; handle that in downstream processing if needed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    text = path.read_text(errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() != ""]

    # Find Fermi energy: typically a line like "Fermi Energy:  <value>"
    fermi_energy = _find_fermi_energy(lines)

    # Find reciprocal lattice block (3 lines after "BEGIN_BLOCK_BANDGRID_3D" / "BEGIN_BANDGRID_3D")
    rec_lattice = _find_reciprocal_lattice(lines)

    # Parse bands
    bands = _parse_bands(lines)

    return BXSFData(
        fermi_energy=fermi_energy,
        rec_lattice=rec_lattice,
        bands=bands,
    )


def _find_fermi_energy(lines: List[str]) -> float:
    for ln in lines:
        low = ln.lower()
        if "fermi" in low and "energy" in low:
            # Try to grab the last float on the line
            tokens = ln.replace(":", " ").split()
            floats = []
            for t in tokens:
                try:
                    floats.append(float(t))
                except ValueError:
                    pass
            if floats:
                return floats[-1]
    raise BXSFParseError(
        "Could not find Fermi energy line (looked for 'Fermi' and 'Energy')."
    )


def _find_bandgrid_header(lines: List[str]) -> Optional[int]:
    for i, ln in enumerate(lines):
        up = ln.upper()
        if "BANDGRID_3D" not in up:
            continue
        if up.startswith("BEGIN_BLOCK_BANDGRID_3D"):
            continue
        if up.startswith("END_BANDGRID_3D") or up.startswith("END_BLOCK_BANDGRID_3D"):
            continue
        return i
    return None


def _find_reciprocal_lattice(lines: List[str]) -> np.ndarray:
    # Typical structure:
    # BEGIN_BLOCK_BANDGRID_3D
    #   ...
    # BEGIN_BANDGRID_3D
    #   <nbands>
    #   <nx> <ny> <nz>
    #   <origin_kx> <origin_ky> <origin_kz>
    #   <b1x> <b1y> <b1z>
    #   <b2x> <b2y> <b2z>
    #   <b3x> <b3y> <b3z>
    #   BAND: 1
    #   ...
    begin_idx = _find_bandgrid_header(lines)
    if begin_idx is None:
        raise BXSFParseError("Could not find BANDGRID_3D header.")

    # After BEGIN_BANDGRID_3D:
    # i+1: nbands
    # i+2: grid dims
    # i+3: origin
    # i+4..i+6: reciprocal lattice vectors
    try:
        b1 = np.array([float(x) for x in lines[begin_idx + 4].split()], dtype=float)
        b2 = np.array([float(x) for x in lines[begin_idx + 5].split()], dtype=float)
        b3 = np.array([float(x) for x in lines[begin_idx + 6].split()], dtype=float)
    except Exception as e:
        raise BXSFParseError(
            f"Failed to parse reciprocal lattice vectors near BEGIN_BANDGRID_3D: {e}"
        ) from e

    if b1.size != 3 or b2.size != 3 or b3.size != 3:
        raise BXSFParseError(
            "Reciprocal lattice vectors do not have 3 components each."
        )

    return np.vstack([b1, b2, b3])


def _parse_bands(lines: List[str]) -> List[BXSFBand]:
    # Locate BEGIN_BANDGRID_3D to get nbands, grid, origin, and start position.
    begin_idx = _find_bandgrid_header(lines)
    if begin_idx is None:
        raise BXSFParseError("Could not find BANDGRID_3D header.")

    try:
        nbands = int(float(lines[begin_idx + 1].split()[0]))
        nx, ny, nz = (int(float(x)) for x in lines[begin_idx + 2].split()[:3])
        origin = np.array(
            [float(x) for x in lines[begin_idx + 3].split()[:3]], dtype=float
        )
    except Exception as e:
        raise BXSFParseError(
            f"Failed to parse header after BEGIN_BANDGRID_3D: {e}"
        ) from e

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise BXSFParseError(f"Invalid grid dimensions: {(nx, ny, nz)}")

    total_points = nx * ny * nz

    # Find each "BAND:" marker and read total_points floats after it (until next BAND/END)
    band_markers: List[Tuple[int, int]] = []  # (band_index, line_number_of_marker)
    for i, ln in enumerate(lines):
        up = ln.upper()
        if up.startswith("BAND:"):
            # BAND: <int>
            try:
                band_idx = int(float(ln.split(":")[1].strip().split()[0]))
            except Exception as e:
                raise BXSFParseError(
                    f"Could not parse band index from line: '{ln}' ({e})"
                ) from e
            band_markers.append((band_idx, i))

    if not band_markers:
        raise BXSFParseError("No 'BAND:' sections found.")

    # Sort by position in file
    band_markers.sort(key=lambda t: t[1])

    bands: List[BXSFBand] = []
    for b_i, (band_idx, marker_line) in enumerate(band_markers):
        # Data starts on next line
        start = marker_line + 1
        end = band_markers[b_i + 1][1] if b_i + 1 < len(band_markers) else len(lines)

        floats: List[float] = []
        for j in range(start, end):
            up = lines[j].upper()
            if up.startswith("END_BANDGRID_3D") or up.startswith(
                "END_BLOCK_BANDGRID_3D"
            ):
                break
            # Accumulate floats from this line
            for tok in lines[j].split():
                try:
                    floats.append(float(tok))
                except ValueError:
                    # Ignore non-numeric tokens just in case
                    pass
            if len(floats) >= total_points:
                break

        if len(floats) < total_points:
            raise BXSFParseError(
                f"Band {band_idx}: expected {total_points} energy points, got {len(floats)}."
            )

        arr = np.array(floats[:total_points], dtype=float)
        energies = arr.reshape((nx, ny, nz), order="C")

        bands.append(
            BXSFBand(
                index=band_idx,
                grid_shape=(nx, ny, nz),
                origin=origin.copy(),
                energies=energies,
            )
        )

    # Optional consistency check vs nbands in header
    # Some files lie or include spin channels etc., so we don't hard-fail.
    if nbands != len(bands):
        # Keep going; user can decide how to handle.
        pass

    return bands


def fermi_crossing_bands(
    data: BXSFData, fermi_energy: float = None, tol: float = 1e-6
) -> List[int]:
    """
    Return band indices whose energy range brackets the Fermi level (likely to contribute to the FS).
    """
    ef = data.fermi_energy if fermi_energy is None else fermi_energy
    hits: List[int] = []
    for b in data.bands:
        e_min = float(np.min(b.energies))
        e_max = float(np.max(b.energies))
        if (e_min - tol) <= ef <= (e_max + tol):
            hits.append(b.index)
    return hits


def compute_brillouin_zone_mesh(
    b_vectors: Iterable[Iterable[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from seekpath.brillouinzone.brillouinzone import get_BZ
    except ImportError as exc:
        raise ImportError(
            "seekpath is required to compute the Brillouin zone mesh."
        ) from exc

    b1, b2, b3 = map(np.array, b_vectors)
    bz = get_BZ(b1, b2, b3)
    faces = bz.get("faces")
    if not faces:
        triangles = np.array(bz["triangles_vertices"], dtype=float)
        vertices = triangles.reshape(-1, 3)
        face_indices = np.arange(vertices.shape[0], dtype=int).reshape(-1, 3)
        return vertices, face_indices

    vertices: List[List[float]] = []
    vertex_map: Dict[Tuple[float, float, float], int] = {}
    face_indices: List[List[int]] = []

    def vertex_key(point: np.ndarray, tol: float = 1e-8) -> Tuple[float, float, float]:
        return (
            round(float(point[0]) / tol) * tol,
            round(float(point[1]) / tol) * tol,
            round(float(point[2]) / tol) * tol,
        )

    for face in faces:
        poly = np.array(face, dtype=float)
        if poly.shape[0] < 3:
            continue
        idx = []
        for p in poly:
            key = vertex_key(p)
            if key not in vertex_map:
                vertex_map[key] = len(vertices)
                vertices.append([float(p[0]), float(p[1]), float(p[2])])
            idx.append(vertex_map[key])
        for i in range(1, len(idx) - 1):
            face_indices.append([idx[0], idx[i], idx[i + 1]])

    return np.array(vertices, dtype=float), np.array(face_indices, dtype=int)


def compute_brillouin_zone_planes(
    b_vectors: Iterable[Iterable[float]],
) -> List[Dict[str, List[float]]]:
    try:
        from seekpath.brillouinzone.brillouinzone import get_BZ
    except ImportError as exc:
        raise ImportError(
            "seekpath is required to compute the Brillouin zone planes."
        ) from exc

    b1, b2, b3 = map(np.array, b_vectors)
    bz = get_BZ(b1, b2, b3)
    faces = bz.get("faces") or []
    triangles = bz.get("triangles_vertices") or []
    if not faces and triangles:
        faces = triangles

    if len(faces) == 0:
        return []

    center = np.array(triangles, dtype=float).reshape(-1, 3).mean(axis=0)
    planes: List[Dict[str, List[float]]] = []
    for face in faces:
        face = np.array(face, dtype=float)
        if face.shape[0] < 3:
            continue
        p0, p1, p2 = face[0], face[1], face[2]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-30:
            continue
        n = n / norm
        if np.dot(n, center - p0) > 0:
            n = -n
        planes.append({"normal": n.tolist(), "point": p0.tolist()})

    return planes


def add_brillouin_zone(
    viewer: "WeasWidget",
    b_vectors: Iterable[Iterable[float]],
    name: str = "Brillouin-zone",
    color: Optional[Union[List[float], str]] = None,
    opacity: Optional[float] = 0.1,
    show_edges: bool = True,
    edge_color: Optional[Union[List[float], str]] = None,
    material_type: str = "Standard",
    depth_write: bool = False,
    depth_test: bool = False,
    side: str = "DoubleSide",
    clear_depth: bool = True,
    render_order: int = 10,
):
    vertices, faces = compute_brillouin_zone_mesh(b_vectors)
    if isinstance(color, (list, tuple)) and len(color) != 3:
        raise ValueError("RGB color must be like [r, g, b].")
    mesh_color = color or [0.0, 0.0, 0.5]
    mesh_opacity = 0.1 if opacity is None else float(opacity)

    edge_color = edge_color or [0.0, 0.0, 0.0, 1.0]
    if len(edge_color) == 3:
        edge_color = [edge_color[0], edge_color[1], edge_color[2], 1.0]
    setting = {
        "name": name,
        "vertices": vertices.reshape(-1).tolist(),
        "faces": faces.reshape(-1).tolist(),
        "color": mesh_color,
        "opacity": float(mesh_opacity),
        "position": [0.0, 0.0, 0.0],
        "materialType": material_type,
        "showEdges": bool(show_edges),
        "edgeColor": [float(c) for c in edge_color],
        "depthWrite": bool(depth_write),
        "depthTest": bool(depth_test),
        "side": side,
        "clearDepth": bool(clear_depth),
        "renderOrder": render_order,
    }
    settings = []
    existing_settings = getattr(viewer.any_mesh, "settings", None)
    if isinstance(existing_settings, list):
        settings = list(existing_settings)
    settings.append(setting)
    viewer.any_mesh.settings = settings
    return setting


def add_reciprocal_axes(
    viewer: "WeasWidget",
    b_vectors: Iterable[Iterable[float]],
    name: str = "reciprocal-axes",
    color: Union[List[float], str] = "#000000",
    radius: float = 0.02,
    factor: float = 0.8,
):
    b1, b2, b3 = [list(vec) for vec in b_vectors]
    setting = {
        "origins": [[0.0, 0.0, 0.0]] * 3,
        "vectors": [b1, b2, b3],
        "color": color,
        "radius": radius,
        "factor": factor,
    }
    settings = {}
    existing_settings = getattr(viewer.avr.vf, "_settings", None)
    if isinstance(existing_settings, dict):
        settings = dict(existing_settings)
    settings[name] = setting
    viewer.avr.vf.settings = settings
    return setting


def add_fermi_surface_from_bxsf(
    viewer: "WeasWidget",
    file_path: str,
    band_index: int = None,
    fermi_energy: Optional[float] = None,
    drop_periodic: bool = True,
    clip_bz: bool = True,
    combine_bands: bool = False,
    name: str = None,
    color: list = None,
    opacity: float = 0.6,
    material_type: str = "Standard",
    show_bz: bool = True,
    show_reciprocal_axes: bool = True,
    merge_vertices_tolerance: Optional[float] = None,
    smooth_normals: Optional[bool] = None,
    brillouin_zone_options: Optional[Dict] = None,
    reciprocal_axes_options: Optional[Dict] = None,
):
    """Compute Fermi surface meshes from a BXSF file and render via AnyMesh.

    Parameters
    ----------
    brillouin_zone_options : dict, optional
        Extra keyword arguments forwarded to :func:`add_brillouin_zone`.
    reciprocal_axes_options : dict, optional
        Extra keyword arguments forwarded to :func:`add_reciprocal_axes`.
    """
    from ase import Atoms

    bxsf_data = parse_bxsf(file_path)
    fermi, b_vectors = bxsf_data.fermi_energy, bxsf_data.rec_lattice
    fermi_energy = fermi if fermi_energy is None else fermi_energy
    if band_index is not None:
        fs_bands = [band_index]
    else:
        fs_bands = fermi_crossing_bands(bxsf_data, fermi_energy=fermi_energy)
    if not fs_bands:
        raise ValueError(
            "No Fermi-crossing bands found. Specify band_index to render a single band."
        )
    bands_by_index = {band.index: band for band in bxsf_data.bands}
    missing = [idx for idx in fs_bands if idx not in bands_by_index]
    if missing:
        raise ValueError(f"Band indices not found in BXSF data: {missing}.")

    def _is_rgb_list(value) -> bool:
        return (
            isinstance(value, (list, tuple))
            and len(value) == 3
            and all(isinstance(x, (int, float)) for x in value)
        )

    def _auto_band_colors(count: int) -> List[List[float]]:
        colors = []
        for i in range(count):
            hue = (i * 0.61803398875) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.9)
            colors.append([float(r), float(g), float(b)])
        return colors

    if color is not None and not _is_rgb_list(color):
        if not (
            isinstance(color, (list, tuple))
            and color
            and all(_is_rgb_list(c) for c in color)
        ):
            raise ValueError(
                "Color must be an RGB list like [r, g, b] or a list of RGB lists."
            )

    mesh_color = color if _is_rgb_list(color) else [0.0, 1.0, 0.0]
    mesh_opacity = float(opacity)

    b1, b2, b3 = map(np.array, b_vectors)
    origin = [0.0, 0.0, 0.0]
    planes = compute_brillouin_zone_planes(b_vectors) if clip_bz else []

    datasets = []
    for idx in fs_bands:
        band = bands_by_index[idx]
        energy = band.energies
        if drop_periodic:
            if energy.shape[0] > 1 and energy.shape[1] > 1 and energy.shape[2] > 1:
                energy = energy[:-1, :-1, :-1]
        dims = list(map(int, energy.shape))
        values = energy.reshape(-1).tolist()
        datasets.append(
            {
                "name": f"Band-{idx}",
                "dims": dims,
                "values": values,
                "cell": [b1.tolist(), b2.tolist(), b3.tolist()],
                "origin": origin,
            }
        )

    fermi_data = {
        "datasets": datasets,
        "cell": [b1.tolist(), b2.tolist(), b3.tolist()],
        "origin": origin,
    }
    if planes:
        fermi_data["bzPlanes"] = planes
    if show_bz:
        bz_vertices, bz_faces = compute_brillouin_zone_mesh(b_vectors)
        fermi_data["bzMesh"] = {
            "name": "Brillouin-zone",
            "vertices": bz_vertices.reshape(-1).tolist(),
            "faces": bz_faces.reshape(-1).tolist(),
            "color": [0.0, 0.0, 0.5],
            "opacity": 0.1,
            "showEdges": True,
            "edgeColor": [0.0, 0.0, 0.0, 1.0],
            "materialType": material_type,
            "depthWrite": False,
            "depthTest": False,
            "side": "DoubleSide",
            "clearDepth": True,
            "renderOrder": 10,
        }

    fermi_settings = {}
    if combine_bands or len(fs_bands) == 1:
        default_name = f"Band-{fs_bands[0]}" if len(fs_bands) == 1 else "Bands"
        fermi_settings[name or default_name] = {
            "isovalue": float(fermi_energy),
            "color": mesh_color,
            "step_size": 1,
            "opacity": mesh_opacity,
            "periodic": False,
            "clipToBZ": bool(clip_bz),
            "tile": 1,
            "bzCropMargin": 1,
            "datasets": [f"Band-{idx}" for idx in fs_bands],
            "materialType": material_type,
            "mergeVerticesTolerance": merge_vertices_tolerance,
            "smoothNormals": smooth_normals,
        }
    else:
        if color is None:
            band_colors = _auto_band_colors(len(fs_bands))
        elif _is_rgb_list(color):
            band_colors = [list(map(float, color)) for _ in fs_bands]
        else:
            base_colors = [list(map(float, c)) for c in color]
            band_colors = [
                base_colors[i % len(base_colors)] for i in range(len(fs_bands))
            ]
        for idx in fs_bands:
            band_color = band_colors[len(fermi_settings)]
            band_name = name or f"Band-{idx}"
            if name is not None and len(fs_bands) > 1:
                band_name = f"{name}-{idx}"
            fermi_settings[band_name] = {
                "isovalue": float(fermi_energy),
                "color": band_color,
                "step_size": 1,
                "opacity": mesh_opacity,
                "periodic": False,
                "clipToBZ": bool(clip_bz),
                "tile": 1,
                "bzCropMargin": 1,
                "dataset": f"Band-{idx}",
                "materialType": material_type,
                "mergeVerticesTolerance": merge_vertices_tolerance,
                "smoothNormals": smooth_normals,
            }

    atoms = Atoms(symbols=[], positions=[], cell=b_vectors, pbc=True)
    viewer.from_ase(atoms)
    viewer.avr.cell.settings["showCell"] = False
    viewer.avr.fermi.fermi_data = fermi_data
    viewer.avr.fermi.settings = fermi_settings
    if show_reciprocal_axes:
        axes_kwargs = dict(reciprocal_axes_options or {})
        add_reciprocal_axes(viewer, b_vectors, **axes_kwargs)
    if len(fermi_settings) == 1:
        return list(fermi_settings.values())[0]
    return list(fermi_settings.values())
