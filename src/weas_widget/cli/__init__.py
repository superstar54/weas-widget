import click
import json
import os
import webbrowser
from pathlib import Path
from string import Template

import numpy as np
from ase.io import read
from ase.io.cube import read_cube_data

from ..config import CONFIG_DIR, DEFAULT_PORT
from ..server import run_http_server
from ..utils import ASEAdapter, create_volume_data


def auto_find_isovalue(volume):
    """
    Automatically determine a suitable isovalue for isosurface rendering.

    Uses mean and standard deviation or percentiles.
    """
    values = np.array(volume["values"])

    # Use percentile-based approach to avoid outliers affecting the isovalue
    iso_value = np.percentile(np.abs(values), 85)  # 85th percentile threshold
    return round(float(iso_value), 5)  # Round for better formatting


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option(
    "--style",
    default=None,
    help="Model style (0 = ball, 1 = ball+stick, 2 = polyhedra).",
)
@click.option(
    "--color-type", default=None, help="Color scheme (e.g., 'CPK' or 'VESTA')."
)
@click.option(
    "--boundary", default=None, help="Boundary settings for periodic structures."
)
@click.option("--phonon", is_flag=True, help="Enable phonon mode visualization.")
@click.option(
    "--eigenvectors", default=None, help="Eigenvectors for phonon visualization."
)
@click.option("--kpoint", default="[0, 0, 0]", help="K-point for phonon visualization.")
@click.option("--amplitude", default=2, help="Phonon amplitude.")
@click.option("--nframes", default=50, help="Number of frames in phonon animation.")
@click.option("--use-server", is_flag=True, help="Serve the file via HTTP.")
def weas(
    filename,
    style,
    color_type,
    boundary,
    phonon,
    eigenvectors,
    kpoint,
    amplitude,
    nframes,
    use_server,
):
    """
    CLI to visualize atomic structures (XYZ, CIF, CUBE) using WEAS.
    """

    # Identify file format
    extension = os.path.splitext(filename)[1].lower()

    snapshot_json = "null"
    atoms_json = None
    volume_json = "null"
    isovalue = None
    atoms = None

    # Read file using ASE or import a saved viewer state
    if extension == ".json":
        with open(filename, "r", encoding="utf-8") as f:
            snapshot = json.load(f)
        version = snapshot.get("version")
        if version not in ("weas_state_v1", "weas_widget_state_v1"):
            raise click.ClickException(
                "Invalid WEAS state file: missing or unsupported 'version'."
            )
        snapshot_json = json.dumps(snapshot)
    elif extension == ".cube":
        volume, atoms = read_cube_data(filename)
        atoms_json = json.dumps(ASEAdapter.to_weas(atoms))
        volume_json = create_volume_data(volume, cell=atoms.get_cell().array.tolist())
        # Automatically determine isovalue
        isovalue = auto_find_isovalue(volume_json)
    else:
        atoms = read(filename, index=":")
        atoms_json = json.dumps(ASEAdapter.to_weas(atoms))
        atoms = atoms[0]  # Use the first frame for visualization

    # Set default settings based on file format
    default_settings = {
        ".xyz": {
            "style": 1 if style is None else style,  # Ball + stick
            "color_type": "CPK" if color_type is None else color_type,
            "boundary": None if boundary is None else boundary,
        },
        ".cif": {
            "style": 2 if style is None else style,  # Polyhedral representation
            "color_type": "VESTA" if color_type is None else color_type,
            "showBondedAtoms": True,
            "boundary": "[[-0.01, 1.01], [-0.01, 1.01], [-0.01, 1.01]]"
            if boundary is None
            else boundary,
        },
        ".cube": {
            "style": 0 if style is None else style,  # Ball representation
            "color_type": "VESTA" if color_type is None else color_type,
            "boundary": None if boundary is None else boundary,
        },
        ".json": {
            "style": None if style is None else style,
            "color_type": None if color_type is None else color_type,
            "boundary": None if boundary is None else boundary,
        },
    }

    # Use default settings if no user input
    settings = default_settings.get(
        extension, default_settings[".xyz"]
    )  # Default to XYZ settings

    # Convert JSON-like strings to proper format
    boundary_json = (
        json.dumps(json.loads(settings["boundary"])) if settings["boundary"] else "null"
    )
    eigenvectors_json = json.dumps(json.loads(eigenvectors)) if eigenvectors else "null"
    kpoint_json = json.dumps(json.loads(kpoint))

    viewer_style = "null" if settings["style"] is None else settings["style"]
    model_style = "null" if settings["style"] is None else settings["style"]
    color_type = settings["color_type"] if settings["color_type"] is not None else ""
    show_bonded_atoms = "true" if settings.get("showBondedAtoms", False) else "false"

    js_template_path = Path(__file__).with_name("template.js")
    if atoms_json is None:
        atoms_json = "null"
    js_template = Template(js_template_path.read_text(encoding="utf-8"))
    js_script = js_template.substitute(
        viewer_style=viewer_style,
        snapshot_json=snapshot_json,
        atoms_json=atoms_json,
        model_style=model_style,
        color_type=color_type,
        show_bonded_atoms=show_bonded_atoms,
        boundary_json=boundary_json,
        volume_json=volume_json,
        isovalue="null" if isovalue is None else isovalue,
        phonon=phonon,
        eigenvectors_json=eigenvectors_json,
        kpoint_json=kpoint_json,
        amplitude=amplitude,
        nframes=nframes,
    )

    template_path = Path(__file__).with_name("template.html")
    template = Template(template_path.read_text(encoding="utf-8"))
    html_content = template.substitute(
        title="WEAS Visualization",
        js_script=js_script,
    )

    formula = atoms.get_chemical_formula() if atoms is not None else "weas_state"
    html_filename = os.path.join(CONFIG_DIR, f"{formula}.html")

    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    if use_server:
        run_http_server(DEFAULT_PORT)
        # webbrowser.open(f"http://localhost:{PORT}/{os.path.basename(html_filename)}")
    else:
        webbrowser.open("file://" + os.path.abspath(html_filename))


__all__ = ["weas"]
