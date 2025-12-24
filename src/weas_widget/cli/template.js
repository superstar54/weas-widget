<script type="module">
  import * as THREE from "https://unpkg.com/three@0.152.0/build/three.module.js";
  import { WEAS, Atoms } from "https://unpkg.com/weas@0.2.1/dist/index.mjs";

  window.THREE = THREE;

  const domElement = document.getElementById("weas");

  const viewerConfig = {
    logLevel: "debug"
  };
  if ($viewer_style !== null) {
    viewerConfig._modelStyle = $viewer_style;
  }

  const editor = new WEAS({ domElement, viewerConfig });
  window.editor = editor;

  const snapshot = $snapshot_json;
  if (snapshot !== null) {
    editor.importState(snapshot);
  } else {
    // Load atoms directly from Python (ASE parsed)
    let trajectory = $atoms_json;
    let atoms;

    // Convert atoms to WEAS format
    if (Array.isArray(trajectory)) {
      trajectory = trajectory.map((atom) => new Atoms(atom));
      atoms = trajectory[0];
    } else {
      atoms = new Atoms(trajectory);
      trajectory = atoms;
    }

    editor.avr.atoms = trajectory;
    editor.avr.modelStyle = $model_style;
    editor.avr.colorType = "$color_type";
    editor.avr.showBondedAtoms = $show_bonded_atoms;

    // Apply boundary settings if necessary
    if ($boundary_json !== null) {
      editor.avr.boundary = $boundary_json;
    }

    // Handle Cube files (Isosurface visualization)
    let volume_json = $volume_json;
    if (volume_json !== null) {
      console.log("Volumetric data loaded.");
      editor.avr.volumetricData = volume_json;
      editor.avr.isosurfaceManager.setSettings({
        positive: { isovalue: $isovalue, mode: 1, step_size: 1 },
        negative: { isovalue: -$isovalue, color: "#ff0000", mode: 1 }
      });
    }

    editor.avr.drawModels();
  }

  // If phonon mode is enabled, apply phonon settings
  if (snapshot === null && "$phonon" === "True") {
    const eigenvectors = $eigenvectors_json;
    const kpoint = $kpoint_json;

    editor.avr.fromPhononMode({
      atoms: atoms,
      eigenvectors: eigenvectors,
      amplitude: $amplitude,
      factor: 1,
      nframes: $nframes,
      kpoint: kpoint,
      repeat: [4, 4, 1],
      color: "#ff0000",
      radius: 0.1
    });

    editor.avr.frameDuration = 50;
    editor.avr.showBondedAtoms = false;
  }

</script>
