import * as weas from "https://unpkg.com/weas@0.0.3/dist/weas.mjs";
export function render({ model, el }) {
    let getAtomsData = () => model.get("atoms");
    let viewerElement = document.createElement("div");
    viewerElement.style.cssText = "position: relative; width: 500px; height: 500px;";
    el.appendChild(viewerElement);
    const renderAtoms = () => {
        const data = model.get("atoms");
        const atoms = new weas.Atoms(data);
        const bjs = new weas.BlendJS(viewerElement);
        const avr = new weas.AtomsViewer(bjs, atoms);
        bjs.render();
    };
    // Initial rendering
    setTimeout(() => {
        renderAtoms();
            }, 10
    ); // Delay rendering by 500ms
    // Listen for changes in the 'atoms' property
    model.on("change:atoms", () => {
        // Clear the existing content of viewerElement
        viewerElement.innerHTML = '';
        // Re-render with the new atoms data
        renderAtoms();
    });
}
