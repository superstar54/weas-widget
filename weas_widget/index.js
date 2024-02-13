import * as weas from "https://unpkg.com/weas@0.0.6-e/dist/weas.mjs";
export function render({ model, el }) {
    let avr; // Declare avr here
    let viewerElement = document.createElement("div");
    viewerElement.style.cssText = "position: relative; width: 400px; height: 400px;";
    el.appendChild(viewerElement);
    const renderAtoms = () => {
        const data = model.get("atoms");
        const atoms = new weas.Atoms(data);
        const bjs = new weas.BlendJS(viewerElement);
        avr = new weas.AtomsViewer(bjs, atoms);
        avr.drawModels();
        bjs.render();
        return avr;
    };
    // Initial rendering
    setTimeout(() => {
        avr = renderAtoms();
            }, 10
    ); // Delay rendering by 10ms
    // Listen for changes in the 'atoms' property
    model.on("change:atoms", () => {
        const data = model.get("atoms");
        const atoms = new weas.Atoms(data);
        // Re-render with the new atoms data
        avr.atoms = atoms;
        // uuid is used to identify the atoms object in the viewer
        // so that we can delete it later
        atoms.uuid = avr.uuid;
        avr.drawModels();
    });
    // Listen for the custom 'atomsUpdated' event
    viewerElement.addEventListener('atomsUpdated', (event) => {
        const updatedAtoms = event.detail.to_dict(); // event.detail contains the updated atoms
        model.set("atoms", updatedAtoms);
        model.save_changes();
        console.log("Updated atoms: ", updatedAtoms);
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

}
