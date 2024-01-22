import anywidget
import traitlets

class WeasWidget(anywidget.AnyWidget):
    _esm = """
    import * as weas from "https://unpkg.com/weas@0.0.3/dist/weas.mjs";
    export function render({ model, el }) {
        let getAtomsData = () => model.get("atoms");
        let viewerElement = document.createElement("div");
        viewerElement.style.cssText = "position: relative; width: 500px; height: 500px;";
        el.appendChild(viewerElement);
        // Use the atoms data from the model
        const data = getAtomsData();

        const atoms = new weas.Atoms(data);
        setTimeout(() => {
                const bjs = new weas.BlendJS(viewerElement);
                const avr = new weas.AtomsViewer(bjs, atoms);
                bjs.render();
            }, 10
        ); // Delay rendering by 500ms
    }
    """
    atoms = traitlets.Dict().tag(sync=True)




def show_ase(atoms):
    species = {}
    cell = atoms.get_cell().array.flatten().tolist()
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    numbers = atoms.get_atomic_numbers()
    speciesArray = symbols
    for i in range(len(symbols)):
        species[symbols[i]] = [symbols[i], numbers[i]]
    atoms={"species": species, "cell": cell, "positions": positions, "speciesArray": speciesArray}
    viewer = WeasWidget(atoms=atoms)
    return viewer
