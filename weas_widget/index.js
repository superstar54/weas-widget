import * as weas from "https://unpkg.com/weas@0.0.5-b/dist/weas.mjs";
export function render({ model, el }) {
    let avr; // Declare avr here
    let viewerElement = document.createElement("div");
    viewerElement.style.cssText = "position: absolte; width: 100%; height: 500px;";
    el.appendChild(viewerElement);
    const renderAtoms = () => {
        const data = model.get("atoms");
        const atoms = new weas.Atoms(data);
        const bjs = new weas.BlendJS(viewerElement);
        avr = new weas.AtomsViewer(bjs, atoms);
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
        avr.drawModel();
    });
    // Listen for the custom 'atomsUpdated' event
    viewerElement.addEventListener('atomsUpdated', (event) => {
        const updatedAtoms = event.detail.to_dict(); // event.detail contains the updated atoms
        model.set("atoms", updatedAtoms);
        model.save_changes();
        console.log("Updated atoms: ", updatedAtoms);
    });

}
