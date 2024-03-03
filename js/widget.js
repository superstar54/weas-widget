// if we want test weas package, then use the following import
// clone the weas repo and import the weas module
// import * as weas from "../../weas/src/index.js";
// if not, then use the release version from unpkg
import * as weas from "https://unpkg.com/weas@0.0.8-a/dist/weas.mjs";
import "./widget.css";



function render({ model, el }) {
    let avr; // Declare avr here
    let viewerElement = document.createElement("div");
    // Stop propagation of mouse and keyboard events from the viewer to jupyter notebook
    // to avoid conflicts with the notebook's keyboard shortcuts
    preventEventPropagation(viewerElement);
    viewerElement.style.cssText = "position: relative; width: 600px; height: 400px;";
    const viewerStyle = model.get("viewerStyle");
    // set the style ortherwise use the default value
    if (viewerStyle) {
        viewerElement.style.width = viewerStyle.width;
        viewerElement.style.height = viewerStyle.height;
    }
    el.appendChild(viewerElement);
    // Function to render atoms
    const renderAtoms = () => {
        // load init parameters from the model
        const atomsData = model.get("atoms");
        let atoms;
        // if atomsData is an array, then create an array of Atoms objects
        if (Array.isArray(atomsData)) {
            atoms = atomsData.map((data) => new weas.Atoms(data));
        } else {
            atoms = new weas.Atoms(atomsData);
        }
        // console.log("atoms: ", atoms);
        const guiConfig = model.get("guiConfig");
        const viewerConfig = {
             debug: model.get("debug"),
            _modelStyle: model.get("modelStyle"),
            _colorBy: model.get("colorBy"),
            _colorType: model.get("colorType"),
            _colorRamp: model.get("colorRamp"),
            _materialType: model.get("materialType"),
            _atomLabelType: model.get("atomLabelType"),
            _showCell: model.get("showCell"),
            _showBondedAtoms: model.get("showBondedAtoms"),
            _boundary: model.get("boundary"),

        };
        avr = new weas.AtomsViewer(viewerElement, atoms, viewerConfig, guiConfig);
        avr.selectedAtomsIndices = model.get("selectedAtomsIndices");
        // avr.atomScales = model.get("atomScales");
        // avr.modelSticks = model.get("modelSticks");
        // avr.modelPolyhedras = model.get("modelPolyhedras");
        // volumetric data
        avr.isosurfaceManager.volumetricData = createVolumeData(model.get("volumetricData"), atoms.cell);
        avr.isosurfaceManager.fromSettings(model.get("isoSettings"));
        // vector field
        avr.VFManager.fromSettings(model.get("vectorField"));
        avr.showVectorField = model.get("showVectorField")
        // mesh primitives
        avr.meshPrimitive.fromSettings(model.get("meshPrimitives"));

        avr.drawModels();
        avr.render();
        return avr;
    };
    // Initial rendering
    setTimeout(() => {
        avr = renderAtoms();
            }, 10
    );
    // js task
    model.on("change:js_task", () => {
        const task = model.get("js_task");
        function run_task(task) {
            switch (task.name) {
                case "drawModels":
                    avr.drawModels();
                    break;
                case "exportImage":
                    const imageData = avr.tjs.exportImage(task.kwargs.resolutionScale);
                    model.set("imageData", imageData);
                    model.save_changes();
                    break;
                case "downloadImage":
                    avr.tjs.downloadImage(task.kwargs.filename);
                    break;
                case "setCameraPosition":
                    avr.tjs.updateCameraAndControls(avr.atoms.getCenterOfGeometry(), task.kwargs.position);
                    break;
            }
        }
        run_task(task);
    });
    // Listen for changes in the 'atoms' property
    model.on("change:atoms", () => {
        const data = model.get("atoms");
        // if uuid of data and avr.atoms are not undefined and are the same, then skip
        if (data.uuid && avr.atoms.uuid && avr.atoms.uuid === data.uuid) {
            return;
        }
        const atoms = new weas.Atoms(data);
        // Re-render with the new atoms data
        avr.updateAtoms(atoms);
        console.log("update viewer from Python.");
    });
    // Listen for the custom 'atomsUpdated' event
    viewerElement.addEventListener('atomsUpdated', (event) => {
        // event detail is a trajectory: a array of atoms data
        // loop all the atoms and export to a dict
        const trajectory = [];
        event.detail.forEach((atomsData) => {
            trajectory.push(atomsData.to_dict());
        });
        trajectory.uuid = avr.uuid;
        model.set("atoms", trajectory);
        model.save_changes();
        // console.log("updatedAtoms: ", trajectory);
        console.log("Updated atoms from event.")
    });
    // Listen for the custom 'viewerUpdated' event
    // this include modelStyle, colorType, materialType, atomLabelType, etc
    viewerElement.addEventListener('viewerUpdated', (event) => {
        const data = event.detail; // event.detail contains the updated data
        // loop through the data and update the model
        for (const key in data) {
            model.set(key, data[key]);
        }
        model.save_changes();
        console.log("Updated viewer: ", data);
    });
    // Listen for changes in the 'viewer' property
    model.on("change:modelStyle", () => {avr.modelStyle = model.get("modelStyle");});
    model.on("change:colorType", () => {avr.colorType = model.get("colorType");});
    model.on("change:materialType", () => {avr.materialType = model.get("materialType");});
    model.on("change:atomLabelType", () => {avr.atomLabelType = model.get("atomLabelType");});
    model.on("change:showCell", () => {avr.showCell = model.get("showCell");});
    model.on("change:showBondedAtoms", () => {avr.showBondedAtoms = model.get("showBondedAtoms");});
    model.on("change:atomScales", () => {avr.atomScales = model.get("atomScales");});
    model.on("change:modelSticks", () => {avr.modelSticks = model.get("modelSticks");});
    model.on("change:modelPolyhedras", () => {avr.modelPolyhedras = model.get("modelPolyhedras");});
    model.on("change:selectedAtomsIndices", () => {avr.selectedAtomsIndices = model.get("selectedAtomsIndices");});
    model.on("change:boundary", () => {avr.boundary = model.get("boundary");});
    // volumetric data
    model.on("change:volumetricData", () => {
        const data = model.get("volumetricData");
        avr.isosurfaceManager.volumetricData = createVolumeData(data);
    });
    model.on("change:isoSettings", () => {
        const isoSettings = model.get("isoSettings");
        avr.isosurfaceManager.fromSettings(isoSettings);
        avr.isosurfaceManager.drawIsosurfaces();
    });

    // Vector field
    model.on("change:vectorField", () => {
        const data = model.get("vectorField");
        avr.VFManager.fromSettings(data);
        avr.VFManager.drawVectorFields();
    });
    // mesh primitives
    model.on("change:meshPrimitives", () => {
        const data = model.get("meshPrimitives");
        console.log("meshPrimitives: ", data);
        avr.meshPrimitive.fromSettings(data);
        avr.meshPrimitive.drawMesh();
    });
}
function createVolumeData(data, cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
    // get the dimensions
    const dims = [data.values.length, data.values[0].length, data.values[0][0].length];
    // flatten the 3d data to 1d
    const values = [].concat.apply([], [].concat.apply([], data.values));
    return {dims, values, cell: cell, origin: [0, 0, 0]};
}

function preventEventPropagation(element) {
    const stopPropagation = (e) => e.stopPropagation();
    ["click", "keydown", "keyup", "keypress"].forEach((eventType) => {
      element.addEventListener(eventType, stopPropagation, false);
    });
  }

export default { render }
