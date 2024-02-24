// use the latest version of weas from unpkg
import * as weas from "https://unpkg.com/weas/dist/weas.mjs";
export function render({ model, el }) {
    let avr; // Declare avr here
    let viewerElement = document.createElement("div");
    viewerElement.style.cssText = "position: relative; width: 600px; height: 400px;";
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
        avr = new weas.AtomsViewer(viewerElement, atoms, guiConfig);
        avr.modelStyle = model.get("modelStyle");
        avr.colorType = model.get("colorType");
        avr.materialType = model.get("materialType");
        avr.atomLabelType = model.get("atomLabelType");
        avr.showCell = model.get("showCell");
        avr.showBondedAtoms = model.get("showBondedAtoms");
        avr.selectedAtomsIndices = model.get("selectedAtomsIndices");
        avr.boundary = model.get("boundary");
        // avr.atomScales = model.get("atomScales");
        // avr.modelSticks = model.get("modelSticks");
        // avr.modelPolyhedras = model.get("modelPolyhedras");
        // volumetric data
        avr.isosurfaceManager.volumetricData = createVolumeData(model.get("volumetricData"), atoms.cell);
        avr.isosurfaceManager.fromSettings(model.get("isoSettings"));
        // vector field
        avr.VFManager.fromSettings(model.get("vectorField"));
        avr.showVectorField = model.get("showVectorField")

        avr.drawModels();
        avr.render();
        return avr;
    };
    // Initial rendering
    setTimeout(() => {
        avr = renderAtoms();
            }, 10
    ); // Delay rendering by 10ms
    // Listen for changes in the '_update' property
    model.on("change:_drawModels", () => {
        avr.drawModels();
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
        const updatedAtoms = event.detail.to_dict(); // event.detail contains the updated atoms
        updatedAtoms.uuid = avr.atoms.uuid;
        model.set("atoms", updatedAtoms);
        model.save_changes();
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

    // export image
    model.on("change:_exportImage", () => {
        const imageData = avr.tjs.exportImage();
        model.set("imageData", imageData);
        model.save_changes();
    });
    // download image
    model.on("change:_downloadImage", () => {
        const filename = model.get("_imageFileName");
        console.log("filename: ", filename);
        avr.tjs.downloadImage(filename);
    });
}

function createVolumeData(data, cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
    // get the dimensions
    const dims = [data.values.length, data.values[0].length, data.values[0][0].length];
    // flatten the 3d data to 1d
    const values = [].concat.apply([], [].concat.apply([], data.values));
    return {dims, values, cell: cell, origin: [0, 0, 0]};
}
