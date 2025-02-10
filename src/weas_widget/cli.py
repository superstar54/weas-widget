import click
import os
import webbrowser
import json
from ase.io import read
from weas_widget.utils import ASEAdapter, create_volume_data
from ase.io.cube import read_cube_data
import numpy as np
from .server import run_http_server
from .config import CONFIG_DIR, DEFAULT_PORT


def auto_find_isovalue(volume):
    """
    Automatically determine a suitable isovalue for isosurface rendering.

    Uses mean and standard deviation or percentiles.
    """
    values = np.array(volume["values"])

    # Use percentile-based approach to avoid outliers affecting the isovalue
    iso_value = np.percentile(
        np.abs(values), 85
    )  # Take the 85th percentile as threshold
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

    # Read file using ASE
    if extension == ".cube":
        volume, atoms = read_cube_data(filename)
        atoms_json = json.dumps(ASEAdapter.to_weas(atoms))
        volume_json = create_volume_data(volume, cell=atoms.get_cell().array.tolist())
        # Automatically determine isovalue
        isovalue = auto_find_isovalue(volume_json)
    else:
        atoms = read(filename, index=":")
        atoms_json = json.dumps(ASEAdapter.to_weas(atoms))
        atoms = atoms[0]  # Use the first frame for visualization
        volume_json = "null"  # No volumetric data for XYZ and CIF
        isovalue = None  # No volumetric data

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

    # Generate JavaScript for the WEAS Viewer
    js_script = f"""
    <script type="module">
      import * as THREE from "https://unpkg.com/three@0.152.0/build/three.module.js";
      import {{ WEAS, Atoms }} from "https://unpkg.com/weas/dist/weas.mjs";

      window.THREE = THREE;

      const domElement = document.getElementById("weas");

      const viewerConfig = {{
        _modelStyle: {settings["style"]},
        logLevel: "debug"
      }};

      const guiConfig = {{
        controls: {{
          enabled: true,
          atomsControl: true,
          colorControl: true,
          cameraControls: true
        }},
        legend: {{
          enabled: true,
          position: "bottom-right"
        }},
        timeline: {{
          enabled: true
        }},
        buttons: {{
          enabled: true,
          fullscreen: true,
          undo: true,
          redo: true,
          download: true,
          measurement: true
        }}
      }};

      const editor = new WEAS({{ domElement, viewerConfig, guiConfig }});
      window.editor = editor;

      // Load atoms directly from Python (ASE parsed)
      let trajectory = {atoms_json};
      let atoms;

      // Convert atoms to WEAS format
      if (Array.isArray(trajectory)) {{
          trajectory = trajectory.map((atom) => new Atoms(atom));
          atoms = trajectory[0];
      }} else {{
          atoms = new Atoms(trajectory);
          trajectory = atoms;
      }}

      editor.avr.atoms = trajectory;
      editor.avr.modelStyle = {settings["style"]};
      editor.avr.colorType = "{settings["color_type"]}";
      editor.avr.showBondedAtoms = {"true" if settings.get("showBondedAtoms", False) else "false"};

      // Apply boundary settings if necessary
      if ({boundary_json} !== null) {{
        editor.avr.boundary = {boundary_json};
      }}

      // Handle Cube files (Isosurface visualization)
      let volume_json = {volume_json};
      if (volume_json !== null) {{
        console.log("Volumetric data loaded.");
        editor.avr.volumetricData = volume_json;
        editor.avr.isosurfaceManager.fromSettings({{
          positive: {{ isovalue: {isovalue}, mode: 1, step_size: 1 }},
          negative: {{ isovalue: -{isovalue}, color: "#ff0000", mode: 1 }}
        }});
        editor.avr.isosurfaceManager.drawIsosurfaces();
      }}

      editor.avr.drawModels();

      // If phonon mode is enabled, apply phonon settings
      if ("{phonon}" === "True") {{
        const eigenvectors = {eigenvectors_json};
        const kpoint = {kpoint_json};

        editor.avr.fromPhononMode({{
          atoms: atoms,
          eigenvectors: eigenvectors,
          amplitude: {amplitude},
          factor: 1,
          nframes: {nframes},
          kpoint: kpoint,
          repeat: [4, 4, 1],
          color: "#ff0000",
          radius: 0.1
        }});

        editor.avr.frameDuration = 50;
        editor.avr.showBondedAtoms = false;
      }}

    </script>
    """

    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8"/>
      <title>WEAS Visualization</title>
    </head>
    <body>
      <div id="weas" style="width:100%; height:800px;"></div>
      {js_script}
    </body>
    </html>
    """

    formula = atoms.get_chemical_formula()
    html_filename = os.path.join(CONFIG_DIR, f"{formula}.html")

    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    if use_server:
        run_http_server(DEFAULT_PORT)
        # webbrowser.open(f"http://localhost:{PORT}/{os.path.basename(html_filename)}")
    else:
        webbrowser.open("file://" + os.path.abspath(html_filename))
