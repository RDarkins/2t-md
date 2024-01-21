from itertools import product
import math
from numbers import Number
from typing import Generator, IO, Iterable, Optional, Union
import sys

import numpy as np


class TwoTempSystem:
    """This class defines a general system composed of single-component
    nuclei and a continuous electronic temperature field.

    Attributes:
        N: Number of nuclei.
        Ninv: 1 / N.
        mass: Mass of each nucleus.
        massinv: 1 / mass.
        positions: Nuclear positions with dimensions (N, 3).
        velocities: Nuclear velocities with dimensions (N, 3).
        forces: Nuclear forces with dimensions (N, 3).
        electronic_temp: Temperature of electrons.
    """
    
    def __init__(self, initial_positions: Iterable[np.ndarray],
                 mass: float,
                 initial_temp: float = 0.):
        """Initialise both subsystems in equilibrium with each other.
        
        Args:
            initial_positions: Initial positions of all nuclei.
            mass: Mass of each nucleus.
            initial_temp: Initial temperature of both subsystems.
        """
        self.positions = np.vstack(list(initial_positions))
        self._N = self.positions.shape[0]
        self._Ninv = 1. / float(self._N)
        self.velocities = np.zeros((self._N, 3))
        self.forces = np.zeros((self._N, 3))
        self._mass = mass
        self._massinv = 1. / mass
        self.electronic_temp = None
        self.set_temp(initial_temp)

    def zero_forces(self) -> None:
        """Set all nuclear forces to zero."""
        self.forces.fill(0.)

    @property
    def N(self) -> int:
        return self._N

    @property
    def Ninv(self) -> float:
        return self._Ninv

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def massinv(self) -> float:
        return self._massinv

    @mass.setter
    def mass(self, mass: float) -> None:
        self._mass = mass
        self._massinv = 1. / mass

    def set_temp(self, temp: float) -> None:
        """Set the temperature of both subsystems.
        
        Args:
            temp: New temperature.
        """
        self.electronic_temp = temp
        self.set_nuclear_temp(temp)

    def nuclear_kinetic_energy(self) -> float:
        """Total kinetic energy of all nuclei."""
        return 0.5 * self.mass * np.sum(self.velocities**2)

    def nuclear_temp(self) -> float:
        """Compute the current nuclear temperature using equipartition."""
        return np.sum(self.velocities**2) * self.mass / (3. * (self.N-1))

    def set_nuclear_temp(self, temp: float) -> None:
        """Draw random nuclear velocities from the Maxwell-Boltzmann
        distribution with the specified temperature.
        
        Args:
            temp: New nuclear temperature.
        """
        self.velocities = np.random.normal(size=(self.N,3))
        self.scale_nuclear_temp(temp)

    def scale_nuclear_temp(self, temp: float) -> None:
        """Use velocity rescaling to attain the desired nuclear temperature.

        Args:
            temp: New nuclear temperature.
        """
        current_temp = self.nuclear_temp()
        scale_factor = math.sqrt(temp / current_temp)
        self.velocities *= scale_factor

    def dump_xyz(self, stream: Optional[IO] = None) -> None:
        """Output the nuclear configuration in the xyz format.

        Args:
            stream: Stream to write to. If None, it will print to stdout.
        """
        print(self.N, end='\n\n', file=stream)
        for row in self.positions:
            print(f"X {row[0]} {row[1]} {row[2]}", file=stream)


class TwoTempNanoparticle(TwoTempSystem):
    """Two-temperature model of a spherical nanoparticle with a simple
    cubic lattice."""
    
    def __init__(self, radius: float,
                 lattice_param: float = 1.,
                 mass: float = 1.,
                 temp: float = 0.):
        """Generate the initial configuration.
        
        Args:
            radius: Radius of the nanoparticle. 
            lattice_param: Lattice parameter of the internal lattice.
            mass: Mass of each nucleus.
            temp: Initial temperature of both subsystems.
        """
        def lattice_generator() -> Generator[np.ndarray, None, None]:
            n = int(radius / lattice_param)
            latt_range = np.arange(-n * lattice_param,
                                   (n+1) * lattice_param,
                                   lattice_param
                                   )
            radius_sq = radius * radius
            # Loop over a cube-shaped lattice
            for point in product(latt_range, latt_range, latt_range):
                # Cut out a sphere
                norm_sq = np.dot(point, point)
                if norm_sq < radius_sq:
                    yield point
                    
        super().__init__(lattice_generator(), mass, temp)
