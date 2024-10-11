import numpy as np
from ase import Atoms
import cmath
from typing import List, Optional, Tuple


class Phonon:
    """
    A class to represent phonon vibrations and trajectories for a given atomic structure.

    Attributes:
        atoms (Atoms): The atomic structure.
        kpoint (Optional[List[float]]): The k-point in reciprocal space.
        eigenvectors (Optional[List[List[Tuple[float, float]]]]): Eigenvectors representing phonon modes.
        add_atom_phase (bool): Whether to add phase based on atom positions.
        vibrations (List[List[complex]]): Calculated vibration phases for each atom.
    """

    def __init__(
        self,
        atoms: Atoms,
        kpoint: Optional[List[float]] = None,
        eigenvectors: Optional[List[List[Tuple[float, float]]]] = None,
        add_atom_phase: bool = True,
    ):
        """
        Initialize the Phonon object.

        Args:
            atoms (Atoms): The atomic structure.
            kpoint (Optional[List[float]]): The k-point in reciprocal space.
            eigenvectors (Optional[List[List[Tuple[float, float]]]]): Eigenvectors for phonon modes.
            add_atom_phase (bool): Whether to add phase based on atom positions.
        """
        self.atoms = atoms
        self.kpoint = kpoint or [0, 0, 0]
        self.eigenvectors = eigenvectors
        self.add_atom_phase = add_atom_phase
        self.vibrations = []

    def calculate_vibrations(self, repeat: Tuple[int, int, int] = (1, 1, 1)) -> None:
        """
        Compute the initial phases and vibrations for the phonon modes.

        Args:
            repeat (Tuple[int, int, int]): Repeat factors for the atomic cell in x, y, and z directions.
        """
        nx, ny, nz = repeat
        fractional_positions = self.atoms.get_scaled_positions()
        natoms = len(self.atoms.positions)
        atom_phase = []

        if self.add_atom_phase:
            atom_phase = [np.dot(self.kpoint, pos) for pos in fractional_positions]
        else:
            atom_phase = [0] * natoms

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    for i in range(natoms):
                        sprod = np.dot(self.kpoint, [ix, iy, iz]) + atom_phase[i]
                        phase = cmath.rect(1.0, sprod * 2.0 * np.pi)
                        self.vibrations.append(
                            [
                                phase * complex(vector[0], vector[1])
                                for vector in self.eigenvectors[i]
                            ]
                        )

    def get_trajectory(
        self,
        amplitude: float,
        nframes: int,
        kpoint: Optional[List[float]] = None,
        eigenvectors: Optional[List[List[Tuple[float, float]]]] = None,
        atoms: Optional[Atoms] = None,
        repeat: Tuple[int, int, int] = None,
        add_atom_phase: Optional[bool] = None,
    ) -> List[Atoms]:
        """
        Generate the trajectory of the phonon mode over time.

        Args:
            amplitude (float): Amplitude of the vibration.
            nframes (int): Number of frames for the trajectory.
            kpoint (Optional[List[float]]): K-point in reciprocal space.
            eigenvectors (Optional[List[List[Tuple[float, float]]]]): Eigenvectors for phonon modes.
            atoms (Optional[Atoms]): Atomic structure.
            repeat (Tuple[int, int, int]): Repeat factors for the atomic cell in x, y, and z directions.
            add_atom_phase (Optional[bool]): Whether to add phase based on atom positions.

        Returns:
            List[Atoms]: List of Atoms objects representing the trajectory frames.
        """
        if atoms is not None:
            self.atoms = atoms
        if kpoint is not None:
            self.kpoint = kpoint
        if eigenvectors is not None:
            self.eigenvectors = eigenvectors
        if self.kpoint is None or self.eigenvectors is None:
            raise ValueError("kpoint and eigenvectors must be provided")
        if add_atom_phase is not None:
            self.add_atom_phase = add_atom_phase

        repeat = repeat or [1, 1, 1]
        self.calculate_vibrations(repeat)
        trajectory = []
        times = [2 * np.pi * (i / nframes) for i in range(nframes)]

        for t in times:
            new_atoms = self.atoms * repeat
            phase = cmath.rect(amplitude, t)
            movement = []

            for i in range(len(new_atoms.positions)):
                displacement = [phase * v for v in self.vibrations[i]]
                displacement_real = [d.real for d in displacement]
                new_atoms.positions[i] = [
                    pos + disp / 5
                    for pos, disp in zip(new_atoms.positions[i], displacement_real)
                ]
                movement.append(displacement_real)

            new_atoms.set_array("movement", np.array(movement))
            trajectory.append(new_atoms)

        return trajectory


def generate_phonon_trajectory(
    atoms: Atoms,
    eigenvectors: List[List[Tuple[float, float]]],
    kpoint: List[float] = None,
    amplitude: float = 1,
    nframes: int = 20,
    repeat: Tuple[int, int, int] = None,
):
    """Generate a trajectory of atoms vibrating along the given eigenvectors.
    Args:
        atoms: ASE Atoms object
        eigenvectors: Eigenvectors of the dynamical matrix
        kpoint (Optional[List[float]]): K-point in reciprocal space.
        amplitude: Amplitude of the vibration
        nframes (int): Number of frames for the trajectory.
    Returns:
        A list of ASE Atoms objects representing the trajectory
    """
    phonon = Phonon(atoms, eigenvectors=eigenvectors)
    trajectory = phonon.get_trajectory(amplitude, nframes, kpoint=kpoint, repeat=repeat)
    return trajectory
