class Optimization:
    def __init__(self, _widget):
        self._widget = _widget

    def optmize(self, forcefield="mmff94", steps=500):
        """_summary_

        Args:
            forcefield (str, optional): _description_. Defaults to "mmff94".
            steps (int, optional): _description_. Defaults to 500.
        """
        from openbabel import openbabel as ob

        # from batoms.utils import read_from_pybel
        mol = self.as_pybel()
        for step in range(steps):
            mol.localopt(forcefield, 1)
            positions = []
            for atom in ob.OBMolAtomIter(mol.OBMol):
                positions.append([atom.GetX(), atom.GetY(), atom.GetZ()])
            # species, positions, arrays, cell, pbc, info = read_from_pybel(mol)
            self.positions = positions

    def as_pybel(self, export_bond=False):
        """Convert an Batoms object to an OBMol object.

        Returns:
            OBMOL: OBMOL
        """
        from openbabel import pybel
        from ase.data import atomic_numbers

        mol = pybel.ob.OBMol()
        arrays = self.arrays
        natom = len(self)
        for i in range(natom):
            a = mol.NewAtom()
            a.SetAtomicNum(atomic_numbers[arrays["elements"][i]])
            a.SetVector(
                arrays["positions"][i][0],
                arrays["positions"][i][1],
                arrays["positions"][i][2],
            )
        if export_bond:
            bond_arrays = self.bond.arrays
            nbond = len(bond_arrays)
            for i in range(nbond):
                mol.AddBond(
                    int(bond_arrays["atoms_index1"][i]) + 1,
                    int(bond_arrays["atoms_index2"][i]) + 1,
                    bond_arrays["order"][i],
                )
        else:
            mol.ConnectTheDots()
            mol.PerceiveBondOrders()
        mol = pybel.Molecule(mol)
        return mol
