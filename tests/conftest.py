import pytest
from weas_widget import WeasWidget
from ase.io import read
from ase.build import molecule
from ase.build import bulk


@pytest.fixture
def h2o():
    # setup fixture h2o
    atoms = molecule("H2O")
    viewer = WeasWidget(from_ase=atoms)
    yield viewer


@pytest.fixture
def c2h6so():
    atoms = molecule("C2H6SO")
    viewer = WeasWidget(from_ase=atoms)
    yield viewer


@pytest.fixture
def ch4():
    atoms = molecule("CH4")
    viewer = WeasWidget(from_ase=atoms)
    yield viewer


@pytest.fixture
def au():
    atoms = bulk("Au", cubic=True)
    viewer = WeasWidget(from_ase=atoms)
    yield viewer


@pytest.fixture
def tio2():
    atoms = read("../tests/datas/tio2.cif")
    viewer = WeasWidget(from_ase=atoms)
    yield viewer


@pytest.fixture
def h2o_homo():
    from ase.io.cube import read_cube_data

    volume, atoms = read_cube_data("h2o-homo.cube")
    viewer = WeasWidget(from_ase=atoms)
    viewer.volumetricData = {"values": volume}
    viewer.isoSettings = [{"isovalue": 0.0001, "mode": 0}]
    yield viewer
