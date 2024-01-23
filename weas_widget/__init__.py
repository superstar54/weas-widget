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
    """
    atoms = traitlets.Dict().tag(sync=True)

    def from_ase(self, atoms):
        # Convert an ASE Atoms object to the widget's format
        species = {}
        cell = atoms.get_cell().array.flatten().tolist()
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        numbers = atoms.get_atomic_numbers()
        speciesArray = symbols
        for i in range(len(symbols)):
            species[symbols[i]] = [symbols[i], numbers[i]]
        atoms={"species": species, "cell": cell, "positions": positions, "speciesArray": speciesArray}
        self.atoms = atoms

    def from_pymatgen(self, structure):
        # Convert a Pymatgen Structure object to the widget's format
        atoms_data = self._convert_to_dict(structure)
        self.atoms = atoms_data


