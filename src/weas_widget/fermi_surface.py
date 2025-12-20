from typing import Dict, Iterable, List, Tuple, Optional, Union
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
    begin_idx = None
    for i, ln in enumerate(lines):
        if ln.upper().startswith("BEGIN_BANDGRID_3D"):
            begin_idx = i
            break
    if begin_idx is None:
        raise BXSFParseError("Could not find 'BEGIN_BANDGRID_3D'.")

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
    begin_idx = None
    for i, ln in enumerate(lines):
        if ln.upper().startswith("BEGIN_BANDGRID_3D"):
            begin_idx = i
            break
    if begin_idx is None:
        raise BXSFParseError("Could not find 'BEGIN_BANDGRID_3D'.")

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


def fermi_crossing_bands(data: BXSFData, tol: float = 1e-6) -> List[int]:
    """
    Return band indices whose energy range brackets the Fermi level (likely to contribute to the FS).
    """
    ef = data.fermi_energy
    hits: List[int] = []
    for b in data.bands:
        e_min = float(np.min(b.energies))
        e_max = float(np.max(b.energies))
        if (e_min - tol) <= ef <= (e_max + tol):
            hits.append(b.index)
    return hits


def kgrid_fractional(data: BXSFData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build fractional k-grid coordinates (u,v,w) in [0,1] along the three reciprocal vectors.

    Returns (u, v, w) as 1D arrays of lengths nx, ny, nz.
    """
    if not data.bands:
        raise ValueError("No bands present.")
    nx, ny, nz = data.bands[0].grid_shape
    u = np.linspace(0.0, 1.0, nx, endpoint=False)
    v = np.linspace(0.0, 1.0, ny, endpoint=False)
    w = np.linspace(0.0, 1.0, nz, endpoint=False)
    return u, v, w


def kgrid_cartesian(
    data: BXSFData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build full k-grid in Cartesian coordinates.

    Returns
    -------
    kx, ky, kz, kpts where:
      - kx, ky, kz are 3D arrays with shape (nx, ny, nz)
      - kpts is an array with shape (nx, ny, nz, 3)
    """
    u, v, w = kgrid_fractional(data)
    U, V, W = np.meshgrid(u, v, w, indexing="ij")  # (nx, ny, nz)

    # Reciprocal vectors as rows
    b1, b2, b3 = data.rec_lattice
    kpts = U[..., None] * b1 + V[..., None] * b2 + W[..., None] * b3  # (nx, ny, nz, 3)
    return kpts[..., 0], kpts[..., 1], kpts[..., 2], kpts


def _clip_triangles_against_plane(
    vertices: np.ndarray,
    faces: np.ndarray,
    normal: np.ndarray,
    point: np.ndarray,
    eps: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    v_list = vertices.tolist()
    f_list: List[List[int]] = []

    def signed_distance(x: np.ndarray) -> float:
        return float(np.dot(normal, x - point))

    for face in faces:
        tri = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
        distances = [
            signed_distance(tri[0]),
            signed_distance(tri[1]),
            signed_distance(tri[2]),
        ]
        inside = [d <= eps for d in distances]

        if all(inside):
            f_list.append(face.tolist())
            continue
        if not any(inside):
            continue

        poly = tri
        poly_d = distances
        new_poly: List[np.ndarray] = []
        for a, da, b, db in zip(
            poly, poly_d, poly[1:] + poly[:1], poly_d[1:] + poly_d[:1]
        ):
            a_in = da <= eps
            b_in = db <= eps
            if a_in:
                new_poly.append(a)
            if a_in ^ b_in:
                t = da / (da - db)
                new_poly.append(a + t * (b - a))

        if len(new_poly) < 3:
            continue

        base_idx = len(v_list)
        for p in new_poly:
            v_list.append(p.tolist())
        idx = list(range(base_idx, base_idx + len(new_poly)))
        for m in range(1, len(idx) - 1):
            f_list.append([idx[0], idx[m], idx[m + 1]])

    return np.array(v_list, float), np.array(f_list, int)


def _clip_to_brillouin_zone(
    vertices: np.ndarray, faces: np.ndarray, b_vectors: Iterable[Iterable[float]]
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from seekpath.brillouinzone.brillouinzone import get_BZ
    except ImportError as exc:
        raise ImportError(
            "seekpath is required for clip_bz=True. Install with `pip install seekpath`."
        ) from exc

    b1, b2, b3 = map(np.array, b_vectors)
    bz = get_BZ(b1, b2, b3)
    bz_faces = bz["faces"]
    center = np.array(bz["triangles_vertices"]).mean(axis=0)

    def face_plane(face_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p0, p1, p2 = face_vertices[0], face_vertices[1], face_vertices[2]
        n = np.cross(p1 - p0, p2 - p0)
        n = n / (np.linalg.norm(n) + 1e-30)
        if np.dot(n, center - p0) > 0:
            n = -n
        return n, p0

    v_out, f_out = vertices, faces
    for face in bz_faces:
        face = np.array(face)
        n, p0 = face_plane(face)
        v_out, f_out = _clip_triangles_against_plane(v_out, f_out, n, p0)

    return v_out, f_out


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


def compute_fermi_surface_mesh(
    energy: np.ndarray,
    b_vectors: Iterable[Iterable[float]],
    fermi_energy: Optional[float] = None,
    supercell_size: Tuple[int, int, int] = (2, 2, 2),
    drop_periodic: bool = True,
    clip_bz: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from skimage import measure
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required to compute the Fermi surface. Install with `pip install scikit-image`."
        ) from exc

    if drop_periodic:
        if energy.shape[0] > 1 and energy.shape[1] > 1 and energy.shape[2] > 1:
            energy = energy[:-1, :-1, :-1]

    if fermi_energy is None:
        fermi_energy = 0.5 * (float(energy.min()) + float(energy.max()))

    sx, sy, sz = supercell_size
    energy_sc = np.tile(energy, supercell_size)
    nx, ny, nz = energy.shape
    spacing = (1.0 / nx, 1.0 / ny, 1.0 / nz)

    verts_frac, faces, _, _ = measure.marching_cubes(
        volume=energy_sc, level=fermi_energy, spacing=spacing
    )

    origin = np.array([-(sx // 2), -(sy // 2), -(sz // 2)], dtype=float)
    verts_frac = verts_frac + origin

    b1, b2, b3 = map(np.array, b_vectors)
    k1, k2, k3 = verts_frac[:, 0], verts_frac[:, 1], verts_frac[:, 2]
    verts_abs = (
        k1[:, None] * b1[None, :]
        + k2[:, None] * b2[None, :]
        + k3[:, None] * b3[None, :]
    )

    if clip_bz:
        verts_abs, faces = _clip_to_brillouin_zone(verts_abs, faces, b_vectors)

    return verts_abs, faces


def add_brillouin_zone(
    viewer,
    b_vectors,
    name="brillouin-zone",
    color=None,
    opacity=0.2,
    show_edges=True,
    edge_color=None,
    material_type="Standard",
    append=True,
):
    vertices, faces = compute_brillouin_zone_mesh(b_vectors)
    mesh_color = color or [1.0, 0.0, 0.0, opacity]
    if len(mesh_color) == 3:
        mesh_color = [mesh_color[0], mesh_color[1], mesh_color[2], opacity]
    edge_color = edge_color or [0.0, 0.0, 0.0, 1.0]
    if len(edge_color) == 3:
        edge_color = [edge_color[0], edge_color[1], edge_color[2], 1.0]
    setting = {
        "name": name,
        "vertices": vertices.reshape(-1).tolist(),
        "faces": faces.reshape(-1).tolist(),
        "color": [float(c) for c in mesh_color],
        "position": [0.0, 0.0, 0.0],
        "materialType": material_type,
        "showEdges": bool(show_edges),
        "edgeColor": [float(c) for c in edge_color],
    }
    settings = []
    if append and isinstance(viewer.any_mesh.settings, list):
        settings = list(viewer.any_mesh.settings)
    settings.append(setting)
    viewer.any_mesh.settings = settings
    return setting


def add_reciprocal_axes(
    viewer,
    b_vectors,
    name="reciprocal-axes",
    color="#000000",
    radius=0.02,
    factor=1.0,
    append=True,
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
    if append and isinstance(existing_settings, dict):
        settings = dict(existing_settings)
    settings[name] = setting
    viewer.avr.vf.settings = settings
    return setting


def add_fermi_surface_from_bxsf(
    viewer,
    file_path,
    band_index: int = None,
    fermi_energy: Optional[float] = None,
    supercell_size: tuple = (2, 2, 2),
    drop_periodic: bool = True,
    clip_bz: bool = False,
    combine_bands: bool = True,
    name: str = None,
    color: list = None,
    opacity: float = 0.6,
    material_type: str = "Standard",
    show_bz: bool = True,
    show_reciprocal_axes: bool = True,
):
    """Compute Fermi surface meshes from a BXSF file and render via AnyMesh."""
    from ase import Atoms

    bxsf_data = parse_bxsf(file_path)
    fermi, b_vectors = bxsf_data.fermi_energy, bxsf_data.rec_lattice
    fermi_energy = fermi if fermi_energy is None else fermi_energy
    if band_index is not None:
        fs_bands = [band_index]
    else:
        fs_bands = fermi_crossing_bands(bxsf_data)
    if not fs_bands:
        raise ValueError(
            "No Fermi-crossing bands found. Specify band_index to render a single band."
        )
    bands_by_index = {band.index: band for band in bxsf_data.bands}
    missing = [idx for idx in fs_bands if idx not in bands_by_index]
    if missing:
        raise ValueError(f"Band indices not found in BXSF data: {missing}.")

    def _mesh_for_band(band):
        return compute_fermi_surface_mesh(
            energy=band.energies,
            b_vectors=b_vectors,
            fermi_energy=fermi_energy,
            supercell_size=supercell_size,
            drop_periodic=drop_periodic,
            clip_bz=clip_bz,
        )

    if color is None:
        mesh_color = [0.96, 0.62, 0.04, opacity]
    elif isinstance(color, (list, tuple)):
        if len(color) == 3:
            mesh_color = [color[0], color[1], color[2], opacity]
        elif len(color) == 4:
            mesh_color = list(color)
        else:
            raise ValueError(
                "color must be RGB or RGBA like [r, g, b] or [r, g, b, a]."
            )
    else:
        raise ValueError("color must be a list or tuple of RGB(A) values.")
    settings = []
    if combine_bands or len(fs_bands) == 1:
        verts_list = []
        faces_list = []
        vert_offset = 0
        for idx in fs_bands:
            band = bands_by_index[idx]
            vertices, faces = _mesh_for_band(band)
            verts_list.append(vertices)
            if faces.size:
                faces_list.append(faces + vert_offset)
            vert_offset += vertices.shape[0]
        if not verts_list:
            raise ValueError("No Fermi surface vertices were generated.")
        vertices = np.vstack(verts_list)
        if faces_list:
            faces = np.vstack(faces_list)
        else:
            faces = np.zeros((0, 3), dtype=int)
        default_name = (
            f"fermi-band-{fs_bands[0]}" if len(fs_bands) == 1 else "fermi-bands"
        )
        settings.append(
            {
                "name": name or default_name,
                "vertices": vertices.reshape(-1).tolist(),
                "faces": faces.reshape(-1).tolist(),
                "color": [float(c) for c in mesh_color],
                "position": [0.0, 0.0, 0.0],
                "materialType": material_type,
            }
        )
    else:
        for idx in fs_bands:
            band = bands_by_index[idx]
            vertices, faces = _mesh_for_band(band)
            band_name = name or f"fermi-band-{idx}"
            if name is not None and len(fs_bands) > 1:
                band_name = f"{name}-{idx}"
            settings.append(
                {
                    "name": band_name,
                    "vertices": vertices.reshape(-1).tolist(),
                    "faces": faces.reshape(-1).tolist(),
                    "color": [float(c) for c in mesh_color],
                    "position": [0.0, 0.0, 0.0],
                    "materialType": material_type,
                }
            )
    atoms = Atoms(symbols=[], positions=[], cell=b_vectors, pbc=True)
    viewer.from_ase(atoms)
    viewer.avr.cell.settings["showCell"] = False
    mesh_settings = []
    if isinstance(viewer.any_mesh.settings, list):
        mesh_settings = list(viewer.any_mesh.settings)
    mesh_settings.extend(settings)
    viewer.any_mesh.settings = mesh_settings
    if show_reciprocal_axes:
        add_reciprocal_axes(viewer, b_vectors)
    if show_bz:
        add_brillouin_zone(viewer, b_vectors)
    if len(settings) == 1:
        return settings[0]
    return settings
