// if we want test weas package, clone the weas repo and import the weas module, then use the following import
// import * as weas from "../../weas-js/src/index.js";
// import "../../weas-js/src/style.css";
// if not, then use the following import
import * as weas from "weas";
import "./widget.css";



function render({ model, el }) {
    let editor;
    let domElement = document.createElement("div");
    el.appendChild(domElement);
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
        // console.log("viewerUpdated: ", data);
        for (const key in data) {
            // skip atomScales, modelSticks, modelPolyhedras
            if (key === "atomScales" || key === "modelSticks" || key === "modelPolyhedras") {
                continue;
            }
            model.set(key, data[key]);
        }
        model.save_changes();
        console.log("Updated viewer: ", data);
    });
    // To scope styles to just elements added by this widget, adding a class to the root el.
    el.classList.add("weas-widget");
    domElement.classList.add("weas-viewer");
    // Stop propagation of mouse and keyboard events from the viewer to jupyter notebook
    // to avoid conflicts with the notebook's keyboard shortcuts
    preventEventPropagation(domElement);
    domElement.style.position = "relative";
    const vs = model.get("viewerStyle") || {};
    domElement.style.width = vs.width || "600px";
    domElement.style.height = vs.height || "400px";
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
        if (guiConfig.atomLegend) {
            guiConfig.atomLegend.enabled = model.get("showAtomLegend");
        } else {
            guiConfig.atomLegend = {enabled: model.get("showAtomLegend"),
                position: "bottom-right",
            };
        }

        const viewerConfig = {
            logLevel: model.get("logLevel"),
            modelStyle: model.get("modelStyle"),
            colorBy: model.get("colorBy"),
            colorType: model.get("colorType"),
            colorRamp: model.get("colorRamp"),
            radiusType: model.get("radiusType"),
            materialType: model.get("materialType"),
            atomLabelType: model.get("atomLabelType"),
            showBondedAtoms: model.get("showBondedAtoms"),
            cellSettings: model.get("cellSettings"),
            bondSettings: {
                hideLongBonds: model.get("hideLongBonds"),
                showHydrogenBonds: model.get("showHydrogenBonds"),
                showOutBoundaryBonds: model.get("showOutBoundaryBonds"),
            },
            boundary: model.get("boundary"),
            currentFrame: model.get("currentFrame"),
            continuousUpdate: model.get("continuousUpdate"),
        };
        editor = new weas.WEAS({ domElement, atoms, viewerConfig, guiConfig });
        window.editor = editor; // for debugging
        // volumetric data
        setVolumetricData(editor, model.get("volumetricData"), atoms.cell);
        editor.avr.transaction(() => {
            const initialState = {
                modelStyle: model.get("modelStyle"),
                colorBy: model.get("colorBy"),
                colorType: model.get("colorType"),
                colorRamp: model.get("colorRamp"),
                radiusType: model.get("radiusType"),
                materialType: model.get("materialType"),
                atomLabelType: model.get("atomLabelType"),
                showBondedAtoms: model.get("showBondedAtoms"),
                boundary: model.get("boundary"),
                selectedAtomsIndices: model.get("selectedAtomsIndices") || [],
            };
            const atomScales = model.get("atomScales");
            if (atomScales.length > 0) {
                initialState.atomScales = atomScales;
            }
            const modelSticks = model.get("modelSticks");
            if (modelSticks.length > 0) {
                initialState.modelSticks = modelSticks;
            }
            const modelPolyhedras = model.get("modelPolyhedras");
            if (modelPolyhedras.length > 0) {
                initialState.modelPolyhedras = modelPolyhedras;
            }
            editor.avr.applyState(initialState, { redraw: "full" });
        });
        editor.state.transaction(() => {
            console.log("set plugin settings");
            const bondSettings = model.get("bondSettings") || {};
            const hideLongBonds = model.get("hideLongBonds");
            const showHydrogenBonds = model.get("showHydrogenBonds");
            const showOutBoundaryBonds = model.get("showOutBoundaryBonds");
            const highlightSettings = model.get("highlightSettings") || {};
            const isoSettings = model.get("isoSettings") || {};
            const sliceSettings = model.get("sliceSettings") || {};
            const vectorField = model.get("vectorField") || {};
            const showVectorField = model.get("showVectorField");
            const cellSettings = model.get("cellSettings") || {};
            editor.state.set({
                bond: {
                    settings: bondSettings,
                    hideLongBonds,
                    showHydrogenBonds,
                    showOutBoundaryBonds,
                },
            });
            editor.state.set({ plugins: { highlight: { settings: highlightSettings } } });
            editor.state.set({
                cell: cellSettings,
                plugins: {
                    isosurface: { settings: isoSettings },
                    volumeSlice: { settings: sliceSettings },
                    vectorField: { settings: vectorField, show: showVectorField },
                    instancedMeshPrimitive: { settings: model.get("instancedMeshPrimitive") || [] },
                    anyMesh: { settings: model.get("anyMesh") || [] },
                    text: { settings: model.get("text") || [] },
                    species: { settings: model.get("speciesSettings") || {} },
                },
            });
        });

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
        const measurementSettings = model.get("measurementSettings");
        if (measurementSettings && editor.state && typeof editor.state.set === "function") {
            editor.state.set({ plugins: { measurement: { settings: measurementSettings } } });
        }
        const animationState = model.get("animationState") || {};
        if (typeof animationState.frameDuration === "number") {
            editor.avr.frameDuration = animationState.frameDuration;
        }
        if (typeof animationState.currentFrame === "number") {
            editor.avr.currentFrame = animationState.currentFrame;
        }
        if (animationState.isPlaying) {
            editor.avr.play();
        } else if (animationState.isPlaying === false) {
            editor.avr.pause();
        }
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
    const initialTask = model.get("js_task");
    if (initialTask && Object.keys(initialTask).length > 0) {
        run_task(editor, initialTask, model);
    }
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

    // Listen for changes in the 'viewer' property
    model.on("change:modelStyle", () => {editor.avr.applyState({ modelStyle: model.get("modelStyle") }, { redraw: "full" });});
    model.on("change:colorType", () => {editor.avr.applyState({ colorType: model.get("colorType") }, { redraw: "full" });});
    model.on("change:colorBy", () => {editor.avr.applyState({ colorBy: model.get("colorBy") }, { redraw: "full" });});
    model.on("change:colorRamp", () => {editor.avr.applyState({ colorRamp: model.get("colorRamp") }, { redraw: "full" });});
    model.on("change:materialType", () => {editor.avr.applyState({ materialType: model.get("materialType") }, { redraw: "full" });});
    model.on("change:atomLabelType", () => {editor.avr.applyState({ atomLabelType: model.get("atomLabelType") }, { redraw: "render" });});
    model.on("change:showBondedAtoms", () => {editor.avr.applyState({ showBondedAtoms: model.get("showBondedAtoms") }, { redraw: "full" });});
    model.on("change:continuousUpdate", () => {editor.avr.applyState({ continuousUpdate: model.get("continuousUpdate") }, { redraw: "none" });});
    model.on("change:atomScales", () => {editor.avr.applyState({ atomScales: model.get("atomScales") }, { redraw: "full" });});
    model.on("change:modelSticks", () => {editor.avr.applyState({ modelSticks: model.get("modelSticks") }, { redraw: "full" });});
    model.on("change:modelPolyhedras", () => {editor.avr.applyState({ modelPolyhedras: model.get("modelPolyhedras") }, { redraw: "full" });});
    model.on("change:selectedAtomsIndices", () => {editor.avr.applyState({ selectedAtomsIndices: model.get("selectedAtomsIndices") || [] }, { redraw: "render" });});
    model.on("change:boundary", () => {editor.avr.applyState({ boundary: model.get("boundary") }, { redraw: "full" });});
    // frame
    model.on("change:currentFrame", () => {
        console.log("change:currentFrame", model.get("currentFrame"));
        editor.avr.currentFrame = model.get("currentFrame");
    });
    // bond settings
    model.on("change:bondSettings", () => {
        const data = model.get("bondSettings") || {};
        editor.state.set({ bond: { settings: data } });
    });
    model.on("change:hideLongBonds", () => {
        const hideLongBonds = model.get("hideLongBonds");
        editor.state.set({ bond: { hideLongBonds } });
    });
    model.on("change:showHydrogenBonds", () => {
        const showHydrogenBonds = model.get("showHydrogenBonds");
        editor.state.set({ bond: { showHydrogenBonds } });
    });
    model.on("change:showOutBoundaryBonds", () => {
        const showOutBoundaryBonds = model.get("showOutBoundaryBonds");
        editor.state.set({ bond: { showOutBoundaryBonds } });
    });
    // species settings
    model.on("change:speciesSettings", () => {
        const data = model.get("speciesSettings") || {};
        editor.state.set({ plugins: { species: { settings: data } } });
    });
    // highlight settings
    model.on("change:highlightSettings", () => {
        const data = model.get("highlightSettings") || {};
        editor.state.set({ plugins: { highlight: { settings: data } } });
    });
    // cell settings
    model.on("change:cellSettings", () => {
        const data = model.get("cellSettings") || {};
        editor.state.set({ cell: data });
    });
    // volumetric data
    model.on("change:volumetricData", () => {
        const data = model.get("volumetricData");
        console.log("volumetricData: ", data);
        setVolumetricData(editor, data, editor.avr.atoms.cell);
        console.log("volumeData: ", editor.avr.volumetricData);
    });
    model.on("change:isoSettings", () => {
        const isoSettings = model.get("isoSettings") || {};
        editor.state.set({ plugins: { isosurface: { settings: isoSettings } } });
    });
    // volume slice
    model.on("change:sliceSettings", () => {
        const data = model.get("sliceSettings") || {};
        editor.state.set({ plugins: { volumeSlice: { settings: data } } });
    });

    // Vector field
    model.on("change:vectorField", () => {
        const data = model.get("vectorField") || {};
        editor.state.set({ plugins: { vectorField: { settings: data } } });
    });
    model.on("change:showVectorField", () => {
        const show = model.get("showVectorField");
        editor.state.set({ plugins: { vectorField: { show } } });
    });
    // instanced mesh primitives
    model.on("change:instancedMeshPrimitive", () => {
        const data = model.get("instancedMeshPrimitive") || [];
        console.log("instancedMeshPrimitive: ", data);
        editor.state.set({ plugins: { instancedMeshPrimitive: { settings: data } } });
    });

    // any mesh
    model.on("change:anyMesh", () => {
        const data = model.get("anyMesh") || [];
        console.log("anyMesh: ", data);
        editor.state.set({ plugins: { anyMesh: { settings: data } } });
    });

    // text labels
    model.on("change:text", () => {
        const data = model.get("text") || [];
        console.log("text: ", data);
        editor.state.set({ plugins: { text: { settings: data } } });
    });

    // camera settings
    model.on("change:cameraSetting", () => {
        console.log("cameraSetting changed.")
        const cameraSetting = model.get("cameraSetting");
        editor.tjs.updateCameraAndControls(cameraSetting);
    });
    model.on("change:cameraType", () => {
        const cameraType = model.get("cameraType");
        const cameraSetting = model.get("cameraSetting");
        editor.tjs.cameraType = cameraType;
        editor.tjs.updateCameraAndControls(cameraSetting || {});
    });
    model.on("change:cameraZoom", () => {
        const cameraZoom = model.get("cameraZoom");
        editor.tjs.camera.updateZoom(cameraZoom);
        editor.requestRedraw("render");
    });
    model.on("change:cameraPosition", () => {
        const cameraPosition = model.get("cameraPosition");
        editor.tjs.camera.updatePosition(cameraPosition[0], cameraPosition[1], cameraPosition[2]);
        editor.requestRedraw("render");
    });
    model.on("change:cameraLookAt", () => {
        const cameraLookAt = model.get("cameraLookAt");
        editor.tjs.controls.target.set(cameraLookAt[0], cameraLookAt[1], cameraLookAt[2]);
        editor.tjs.controls.update();
        editor.requestRedraw("render");
    });

    let suppressMeasurementSync = false;
    const syncMeasurementToModel = (next) => {
        if (suppressMeasurementSync) {
            return;
        }
        model.set("measurementSettings", next?.settings || {});
        model.save_changes();
    };
    if (editor.state && typeof editor.state.subscribe === "function") {
        editor.state.subscribe("plugins.measurement", (next) => {
            syncMeasurementToModel(next);
        });
    }
    model.on("change:measurementSettings", () => {
        const measurement = model.get("measurementSettings");
        suppressMeasurementSync = true;
        if (editor.state && typeof editor.state.set === "function") {
            editor.state.set({ plugins: { measurement: { settings: measurement } } });
        }
        setTimeout(() => { suppressMeasurementSync = false; }, 0);
    });

    let suppressAnimationSync = false;
    const syncAnimationToModel = (next) => {
        if (suppressAnimationSync) {
            return;
        }
        model.set("animationState", next || {});
        if (next && typeof next.currentFrame === "number") {
            model.set("currentFrame", next.currentFrame);
        }
        model.save_changes();
    };
    if (editor.state && typeof editor.state.subscribe === "function") {
        editor.state.subscribe("animation", (next) => {
            syncAnimationToModel(next);
        });
    }
    model.on("change:animationState", () => {
        const animation = model.get("animationState") || {};
        suppressAnimationSync = true;
        if (typeof animation.frameDuration === "number") {
            editor.avr.frameDuration = animation.frameDuration;
        }
        if (typeof animation.currentFrame === "number") {
            editor.avr.currentFrame = animation.currentFrame;
        }
        if (animation.isPlaying) {
            editor.avr.play();
        } else if (animation.isPlaying === false) {
            editor.avr.pause();
        }
        setTimeout(() => { suppressAnimationSync = false; }, 0);
    });
    // frame
    model.on("change:showAtomLegend", () => {
        editor.avr.guiManager.guiConfig.atomLegend.enabled = model.get("showAtomLegend");
        editor.avr.guiManager.updateLegend();
    });
}
function createVolumeData(data, cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
    if (!data || !data.values || !Array.isArray(data.values) || data.values.length === 0) {
        return null;
    }
    // get the dimensions
    const dims = [data.values.length, data.values[0].length, data.values[0][0].length];
    // flatten the 3d data to 1d
    const values = [].concat.apply([], [].concat.apply([], data.values));
    return {dims, values, cell: cell, origin: [0, 0, 0]};
}

function setVolumetricData(editor, data, cell) {
    if (!editor || !editor.avr) {
        return;
    }
    const volumeData = createVolumeData(data, cell);
    if (!volumeData) {
        return;
    }
    if (typeof editor.avr.setVolumetricData === "function") {
        editor.avr.setVolumetricData(volumeData);
        return;
    }
    editor.avr.volumetricData = volumeData;
    if (editor.avr.isosurfaceManager && typeof editor.avr.isosurfaceManager.drawIsosurfaces === "function") {
        editor.avr.isosurfaceManager.drawIsosurfaces();
    }
    if (editor.avr.volumeSliceManager && typeof editor.avr.volumeSliceManager.drawSlices === "function") {
        editor.avr.volumeSliceManager.drawSlices();
    }
    if (typeof editor.avr.requestRedraw === "function") {
        editor.avr.requestRedraw("render");
    }
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
        case "exportState":
            const snapshot = editor.exportState();
            model.set("stateSnapshot", snapshot);
            model.save_changes();
            break;
        // all the other tasks, run editor.task.name with task.args and task.kwargs
        default:
            // Extract the method based on the 'name' path
            const method = resolveFunctionFromString(editor, task.name);

            if (typeof method === 'function') {
                // Prepare args and kwargs
                const args = task.args || [];
                const kwargs = task.kwargs || null;
                // Handle both args and kwargs if method supports it
                if (args.length > 0) {
                    const finalArgs = [...args];
                    if (kwargs && Object.keys(kwargs).length > 0) {
                        finalArgs.push(kwargs);
                    }
                    method.apply(null, finalArgs);
                } else {
                    if (kwargs && Object.keys(kwargs).length > 0) {
                        method(kwargs);
                    } else {
                        method();
                    }
                }
            } else {
                console.error('Method not found or is not a function');
            }
            break;
    }
    model.set("js_task", {});
    model.save_changes();
}

export default { render }
