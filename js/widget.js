// if we want test weas package, clone the weas repo and import the weas module, then use the following import
// import * as weas from "../../weas/src/index.js";
// if not, then use the following import
import * as weas from "weas";
import "./widget.css";



function render({ model, el }) {
    let editor;
    let domElement = document.createElement("div");
    el.appendChild(domElement);
    // To scope styles to just elements added by this widget, adding a class to the root el.
    el.classList.add("weas-widget");
    // Stop propagation of mouse and keyboard events from the viewer to jupyter notebook
    // to avoid conflicts with the notebook's keyboard shortcuts
    preventEventPropagation(domElement);
    domElement.style.cssText = "position: relative; width: 600px; height: 400px;";
    const viewerStyle = model.get("viewerStyle");
    // set the style ortherwise use the default value
    if (viewerStyle) {
        domElement.style.width = viewerStyle.width;
        domElement.style.height = viewerStyle.height;
    }
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
        if (guiConfig.legend) {
            guiConfig.legend.enabled = model.get("showAtomLegend");
        } else {
            guiConfig.legend = {enabled: model.get("showAtomLegend"),
                position: "bottom-right",
            };
        }

        const viewerConfig = {
             logLevel: model.get("logLevel"),
            _modelStyle: model.get("modelStyle"),
            _colorBy: model.get("colorBy"),
            _colorType: model.get("colorType"),
            _colorRamp: model.get("colorRamp"),
            _radiusType: model.get("radiusType"),
            _materialType: model.get("materialType"),
            _atomLabelType: model.get("atomLabelType"),
            _showCell: model.get("showCell"),
            _showBondedAtoms: model.get("showBondedAtoms"),
            _hideLongBonds: model.get("hideLongBonds"),
            _showHydrogenBonds: model.get("showHydrogenBonds"),
            _boundary: model.get("boundary"),
            _currentFrame: model.get("currentFrame"),
        };
        editor = new weas.WEAS({ domElement, atoms, viewerConfig, guiConfig });
        // window.editor = editor; // for debugging
        editor.avr.selectedAtomsIndices = model.get("selectedAtomsIndices");
        // editor.avr.atomScales = model.get("atomScales");
        // editor.avr.modelSticks = model.get("modelSticks");
        // editor.avr.modelPolyhedras = model.get("modelPolyhedras");
        // species settings
        editor.avr.atomManager.fromSettings(model.get("speciesSettings"));
        // bond settings
        // console.log("bondSettings: ", model.get("bondSettings"));
        editor.avr.bondManager.fromSettings(model.get("bondSettings"));
        // highlight settings
        // console.log("highlightSettings: ", model.get("highlightSettings"));
        editor.avr.highlightManager.fromSettings(model.get("highlightSettings"));
        // volumetric data
        editor.avr.volumetricData = createVolumeData(model.get("volumetricData"), atoms.cell);
        // isosurface
        editor.avr.isosurfaceManager.fromSettings(model.get("isoSettings"));
        // volume slice
        editor.avr.volumeSliceManager.fromSettings(model.get("sliceSettings"));
        // vector field
        editor.avr.VFManager.fromSettings(model.get("vectorField"));
        editor.avr.showVectorField = model.get("showVectorField");
        editor.avr.drawModels();
        // mesh primitives
        editor.instancedMeshPrimitive.fromSettings(model.get("instancedMeshPrimitive"));
        editor.instancedMeshPrimitive.drawMesh();
        //
        const phonon = model.get("phonon");
        console.log("phonon: ", phonon);
        // if phone is not empty object, then create phonon mode
        if (Object.keys(phonon).length > 0) {
            editor.avr.fromPhononMode({
                ...phonon,
                atoms: atoms,
            });
        }
        // camera settings
        const cameraSetting = model.get("cameraSetting");
        editor.tjs.updateCameraAndControls(cameraSetting);
        editor.render();
        return editor;
    };
    // Initial rendering
    editor = renderAtoms();
    // js task
    model.on("change:js_task", () => {
        const task = model.get("js_task");
        // if task is {}, then skip
        if (Object.keys(task).length === 0) {
            return;
        }
        run_task(editor, task, model);
    });
    // Listen for changes in the 'atoms' property
    model.on("change:atoms", () => {
        const data = model.get("atoms");
        // if uuid of data and editor.avr.atoms are not undefined and are the same, then skip
        if (data.uuid && editor.avr.atoms.uuid && editor.avr.atoms.uuid === data.uuid) {
            return;
        }
        const atoms = new weas.Atoms(data);
        // Re-render with the new atoms data
        editor.avr.atoms = atoms;
        console.log("update viewer from Python.");
    });
    // Listen for changes in the 'objectUpdated' property
    // disable this event, because it will be triggered too many times in animation
    // domElement.addEventListener('weas', (event) => {
    //     const detail = event.detail; // event.detail contains the updated data
    //     model.set("python_task", event.detail);
    //     model.save_changes();
    //     console.log("Get event from weas: ");
    // });
    model.on("change:python_task", () => {
        const python_task = model.get("python_task");
        console.log("on change, python_task: ", python_task)
      });
    // Listen for the custom 'atomsUpdated' event
    domElement.addEventListener('atomsUpdated', (event) => {
        // event detail is a trajectory: a array of atoms data
        // loop all the atoms and export to a dict
        const trajectory = [];
        event.detail.forEach((atomsData) => {
            trajectory.push(atomsData.toDict());
        });
        trajectory.uuid = editor.avr.uuid;
        model.set("atoms", trajectory);
        model.save_changes();
        // console.log("updatedAtoms: ", trajectory);
        console.log("Updated atoms from event.")
    });
    // Listen for the custom 'viewerUpdated' event
    // this include modelStyle, colorType, materialType, atomLabelType, etc
    domElement.addEventListener('viewerUpdated', (event) => {
        const data = event.detail; // event.detail contains the updated data
        // loop through the data and update the model
        for (const key in data) {
            model.set(key, data[key]);
        }
        model.save_changes();
        console.log("Updated viewer: ", data);
    });
    // Listen for changes in the 'viewer' property
    model.on("change:modelStyle", () => {editor.avr.modelStyle = model.get("modelStyle");});
    model.on("change:colorType", () => {editor.avr.colorType = model.get("colorType");});
    model.on("change:materialType", () => {editor.avr.materialType = model.get("materialType");});
    model.on("change:atomLabelType", () => {editor.avr.atomLabelType = model.get("atomLabelType");});
    model.on("change:showCell", () => {editor.avr.showCell = model.get("showCell");});
    model.on("change:showBondedAtoms", () => {editor.avr.showBondedAtoms = model.get("showBondedAtoms");});
    model.on("change:atomScales", () => {editor.avr.atomScales = model.get("atomScales");});
    model.on("change:modelSticks", () => {editor.avr.modelSticks = model.get("modelSticks");});
    model.on("change:modelPolyhedras", () => {editor.avr.modelPolyhedras = model.get("modelPolyhedras");});
    model.on("change:selectedAtomsIndices", () => {editor.avr.selectedAtomsIndices = model.get("selectedAtomsIndices");});
    model.on("change:boundary", () => {editor.avr.boundary = model.get("boundary");});
    // frame
    model.on("change:currentFrame", () => {
        editor.avr.currentFrame = model.get("currentFrame");
    });
    // bond settings
    model.on("change:bondSettings", () => {
        const data = model.get("bondSettings");
        editor.avr.bondManager.fromSettings(data);
    });
    // volumetric data
    model.on("change:volumetricData", () => {
        const data = model.get("volumetricData");
        console.log("volumetricData: ", data);
        editor.avr.isosurfaceManager.volumetricData = createVolumeData(data, editor.avr.atoms.cell);
        console.log("volumeData: ", editor.avr.isosurfaceManager.volumetricData);
    });
    model.on("change:isoSettings", () => {
        const isoSettings = model.get("isoSettings");
        console.log("isoSettings: ", isoSettings);
        editor.avr.isosurfaceManager.fromSettings(isoSettings);
        editor.avr.isosurfaceManager.drawIsosurfaces();
        console.log("drawIsosurfaces");
    });

    // Vector field
    model.on("change:vectorField", () => {
        const data = model.get("vectorField");
        editor.avr.VFManager.fromSettings(data);
        editor.avr.VFManager.drawVectorFields();
    });
    // instanced mesh primitives
    model.on("change:instancedMeshPrimitive", () => {
        const data = model.get("instancedMeshPrimitive");
        console.log("instancedMeshPrimitive: ", data);
        editor.instancedMeshPrimitive.fromSettings(data);
        editor.instancedMeshPrimitive.drawMesh();
    });

    // any mesh
    model.on("change:anyMesh", () => {
        const data = model.get("anyMesh");
        console.log("anyMesh: ", data);
        editor.anyMesh.fromSettings(data);
        editor.anyMesh.drawMesh();
    });

    // camera settings
    model.on("change:cameraSetting", () => {
        console.log("cameraSetting changed.")
        const cameraSetting = model.get("cameraSetting");
        editor.tjs.updateCameraAndControls(cameraSetting);
    });
    model.on("change:cameraZoom", () => {
        const cameraZoom = model.get("cameraZoom");
        editor.tjs.camera.updateZoom(cameraZoom);
        editor.tjs.render();
    });
    model.on("change:cameraPosition", () => {
        const cameraPosition = model.get("cameraPosition");
        editor.tjs.camera.updatePosition(cameraPosition[0], cameraPosition[1], cameraPosition[2]);
        editor.tjs.render();
    });
    model.on("change:cameraLookAt", () => {
        const cameraLookAt = model.get("cameraLookAt");
        editor.tjs.controls.target.set(cameraLookAt[0], cameraLookAt[1], cameraLookAt[2]);
        editor.tjs.controls.update();
        editor.tjs.render();
    });
    // frame
    model.on("change:showAtomLegend", () => {
        editor.avr.guiManager.guiConfig.legend.enabled = model.get("showAtomLegend");
        editor.avr.guiManager.updateLegend();
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


// Function to safely resolve the method from the string path
function resolveFunctionFromString(editor, path) {
    const parts = path.split('.');
    const methodName = parts.pop(); // Separate the method name from the path
    const context = parts.reduce((acc, part) => acc && acc[part], editor);

    if (context && typeof context[methodName] === 'function') {
      return context[methodName].bind(context); // Bind the method to its context
    } else {
      console.error('Method not found or is not a function');
      return null;
    }
  }

  function run_task(editor, task, model) {
    console.log("task: ", task);
    switch (task.name) {
        case "exportImage":
            const imageData = editor.tjs.exportImage(task.kwargs.resolutionScale);
            model.set("imageData", imageData);
            model.save_changes();
            break;
        // all the other tasks, run editor.task.name with task.args and task.kwargs
        default:
            // Extract the method based on the 'name' path
            const method = resolveFunctionFromString(editor, task.name);

            if (typeof method === 'function') {
                // Prepare args and kwargs
                const args = task.args || [];
                const kwargs = task.kwargs || {};
                // Handle both args and kwargs if method supports it
                if (args.length > 0) {
                    method.apply(null, [...args, kwargs]);
                } else {
                    method(kwargs);
                }
            } else {
                console.error('Method not found or is not a function');
            }
            break;
    }
}

export default { render }
