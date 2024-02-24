
# Welcome to WEAS Widget!
[![PyPI version](https://badge.fury.io/py/weas-widget.svg)](https://badge.fury.io/py/weas-widget)
[![Docs status](https://readthedocs.org/projects/weas-widget/badge)](http://weas-widget.readthedocs.io/)
[![Unit test](https://github.com/superstar54/weas-widget/actions/workflows/ci.yml/badge.svg)](https://github.com/superstar54/weas-widget/actions/workflows/ci.yml)

A widget to visualize and edit atomistic structures in Jupyter Notebook. It uses [WEAS](https://github.com/superstar54/weas) (Web Environment For Atomistic Structure) in the backend.


<img src="docs/source/_static/images/example-adsorption.gif"  width="100%"/>


Features:

- Model: space-filling, ball-stick, polyhedral.
- Supported File type: cif, xyz.
- Edit structure: move, rotate, delete and replace atoms.
- Support periodic boundary conditions
- Animation
- Isosurface
- Vector field, e.g., magnetic moment


## Installation

Use the pip:

```console
    pip install weas-widget
```

To install the latest version from source, first clone the repository and then install using pip:

```console
    $ git clone https://github.com/superstar54/weas-widget
    $ pip install -e weas-widget
```


## Edit the structure with mouse and keyboard
WEAS supports editing the atoms directly in the GUI and synchronizing with the structure of the Python object.

### Select Atoms
There are two methods for selecting atoms:
- Pick Selection: Click directly on an atom to select it.
- Range Selection: Hold the `Shift` key and drag the right mouse button to select a group of atoms.



### Move, Rotate selected atoms

Press the transform shortcut, and move your mouse.

|Operation | Shortcut|
|----------|---------|
| Move     | `g`   |
| Rotate   | `r`   |


### Delete selected atoms
Press the ``x`` key to delete the selected atoms


### Export edited atoms
One can export the edited atoms to ASE or Pymatgen

## Example

### Load structure
One can load a structure from ASE or Pymatgen

```python
from ase.build import molecule
from weas_widget import WeasWidget
atoms = molecule("C2H6SO")
viewer = WeasWidget()
viewer.from_ase(atoms)
viewer
```

<img src="docs/source/_static/images/example-c2h6so.png"  width="300px"/>



### Crystal view
For a nice visualization of a crystal, one usually shows the polyhedra and the atoms on the unit cell boundary, as well as the bonded atoms outside the cell.

```python
from weas_widget import WeasWidget
viewer1 = WeasWidget()
viewer1.load_example("tio2.cif")
viewer1.modelStyle = 2
viewer1.boundary = [[-0.1, 1.1], [-0.1, 1.1], [-0.1, 1.1]]
viewer1.showBondedAtoms = True
viewer1.colorType = "VESTA"
viewer1
```

<img src="docs/source/_static/images/example-tio2.png"  width="300px"/>


### Isosurface

```python
from ase.build import molecule
from weas_widget import WeasWidget
from ase.io.cube import read_cube_data
volume, atoms = read_cube_data("h2o-homo.cube")
viewer = WeasWidget()
viewer.from_ase(atoms)
viewer.volumetricData = {"values": volume}
viewer.isoSettings = [{"isovalue": 0.0001, "mode": 0}]
viewer
```
<img src="docs/source/_static/images/example-isosurface.png"  width="300px"/>


### Magnetic moment
Show the magnetic moment as a vector field.

```python
from ase.build import bulk
from weas_widget import WeasWidget
import numpy as np
atoms = bulk("Fe", cubic=True)
atoms*=[2, 2, 1]
atoms.set_array("moment", np.ones(len(atoms)))
viewer = WeasWidget()
viewer.from_ase(atoms)
viewer.modelStyle = 1
viewer
```
<img src="docs/source/_static/images/example-magnetic-moment.png"  width="300px"/>


### Phonon
One can visualize the phonon dispersion via lattice vibrations. One only need to use the eigenstates (calculated with an external software) to generate the trajectory.

```python
import numpy as np
from ase.build import bulk
from weas_widget import WeasWidget
from weas_widget.utils import generate_phonon_trajectory

atoms = bulk("Fe", cubic=True)
eigenvector = np.array([[0, -0.0, 0.5], [0, 0.0, -0.5]])
trajectory = generate_phonon_trajectory(atoms, eigenvector, repeat=[4, 4, 1])
viewer = WeasWidget()
viewer.from_ase(trajectory)
# set a vector field to show the arrow
viewer.vectorField = [{"origins": "positions", "vectors": "movement", "radius": 0.1}]
viewer
```

<img src="docs/source/_static/images/example-phonon.gif"  width="300px"/>



### Download image

```python
viewer.download_image("filename.png")
```

## How to use

Please visit: https://weas-widget.readthedocs.io/en/latest/index.html



## Contact
* Xing Wang  <xingwang1991@gmail.com>

## License
[MIT](http://opensource.org/licenses/MIT)
