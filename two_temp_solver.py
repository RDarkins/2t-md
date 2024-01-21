import math
from typing import Any, Dict, Optional

import numpy as np

from corrective import Corrective
from material import Material
from radiation_source import RadiationSource
import symplectic_integrator as symps
from two_temp_system import TwoTempSystem


class TwoTempSolver:
    """Simulate the time evolution of a two temperature system exposed to a
    radiation source. This class is the heart of the package. It supports the
    use of a corrective energy term to maintain a positive heat capacity.
    
    Attributes:
        ttsys: Two temperature system to integrate.
        material: Defines the material (interatomic potentials, etc).
        radiation: Radiation source for supplying energy to the electrons.
        symp: Symplectic integrator for the nuclei.
        corrective: Corrective for maintaining positive heat capacities.
        time: The current time during the simulation.
        use_trotter: If True, use Trotter-Suzuki decomposition and employ
            an ensemble average for the electron-phonon coupling. Offers
            improved numerical stability at the cost of energy drift.
    """
    
    ttsys: TwoTempSystem
    material: Material
    radiation: RadiationSource
    symp: symps.Symplectic
    corrective: Corrective
    time: float = 0.
    use_trotter: bool
    _thermostat_forces: np.ndarray = None
    radiation_energy_dumped: float = 0.
    _gamma1: float = 0.
    _gamma2: float = 0.
    _trotter_G: float = 0.

    def __init__(self, ttsys: TwoTempSystem,
                 material: Material,
                 radiation: Optional[RadiationSource] = None,
                 corrective: Optional[Corrective] = None,
                 symp_type: str = 'vv',
                 use_trotter: bool = False):
        """Initialise the two temperature solver.

        Args:
            ttsys: Two temperature system to integrate.
            material: Defines the material (interatomic potentials, etc).
            radiation: Radiation source for supplying energy to the electrons.
            corrective: Corrective for maintaining positive heat capacities.
            symp_type: Which symplectic integrator to apply to the nuclei.
                       Acceptable values:
                       - 'vv': Velocity Verlet
                       - 'lf': Leapfrog Verlet
            use_trotter: Use Trotter-Suzuki decomposition, else use Euler.
        """
        symp_type = symp_type.lower()
        if symp_type == 'vv':
            # Velocity Verlet
            self.symp = symps.VVSymplectic()
        elif symp_type == 'lf':
            # Leapfrog Verlet
            self.symp = symps.LFSymplectic()
        else:
            raise ValueError(f"Invalid symplectic integrator type: {symp_type}")
        self.use_trotter = use_trotter
        self.ttsys = ttsys
        self.material = material
        self.radiation = radiation or None
        self.corrective = corrective or None

    def update(self, interval: float,
               dt: float = None,
               temp: float = None) -> Dict[str, Any]:
        """Integrate the two temperature system.

        Args:
            interval: How long to integrate for.
            dt: If specified, break the interval into time steps of dt. If
                interval is not a multiple of dt, interval will be overshot.
            temp: If specified, rescale the temperature after every time step
                to this value. Useful for equilibration.
        
        Returns:
            A dictionary of state variables at the end of the interval.
        """
        dt = dt or interval
        nsteps = math.ceil(interval / dt)
        self._update_forces()
        self._ep_coupling_init(dt)
        
        # Multi-step integration
        for step in range(nsteps):
            self._update_nuclear(dt)
            self._update_electronic(dt)
            self.time += dt

            # Impose temperature after each step
            if temp is not None:
                self.ttsys.scale_nuclear_temp(temp)
                self.ttsys.electronic_temp = temp
                
        return {
            't': self.time,
            'tot': self._total_energy(),
            'u': self.material.nuclear_potential(
                self.ttsys, precomputed=True)[0],
            'ke': self.ttsys.nuclear_kinetic_energy(),
            'ee': self.material.electronic_energy(
                self.ttsys.electronic_temp)[0] * self.ttsys.N,
            'w': (self.corrective.energy(self.ttsys, self.material)
                  if self.corrective is not None else 0.),
            'ce': self.material.heat_capacity(
                self.ttsys, precomputed=True)[0],
            'cce': (self.corrective.heat_capacity(self.ttsys, self.material)
                    if self.corrective is not None else 0.),
            'te': self.ttsys.electronic_temp,
            'tn': self.ttsys.nuclear_temp(),
            'q': self._radiation_power(),
            'qe': self.radiation_energy_dumped * self.ttsys.N,
            'cfg': self.ttsys.positions.copy()
        }

    def _total_energy(self) -> float:
        """Sum all Hamiltonian contributions."""
        tot = 0.
        tot += self.ttsys.nuclear_kinetic_energy()
        tot += self.material.nuclear_potential(
            self.ttsys, precomputed=True)[0]
        tot += self.material.electronic_energy(
            self.ttsys.electronic_temp)[0] * self.ttsys.N
        if self.corrective is not None:
            tot += self.corrective.energy(self.ttsys, self.material)
        return tot

    def _update_nuclear(self, dt: float) -> None:
        """Integrate the nuclear subsystem by a single time step.

        Args:
            dt: Time step.
        """
        self.symp.update(self.ttsys, dt, self._update_forces)

    def _update_electronic(self, dt: float) -> None:
        """Integrate the electronic subsystem by a single time step.
        
        The nuclear subsystem must be integrated beforehand.

        Args:
            dt: Time step.
        """
        if self.use_trotter:
            # Trotter-Suzuki decomposition
            
            Tn = self.ttsys.nuclear_temp()

            def half_step_linear() -> None:
                heat_capacity = self._heat_capacity()
                radiation_power = self._radiation_power()
                del_temp = (self._trotter_G*Tn + radiation_power)/heat_capacity
                del_temp *= dt * 0.5
                self.ttsys.electronic_temp += del_temp
                self.radiation_energy_dumped += radiation_power * dt * 0.5

            def full_step_exp() -> None:
                heat_capacity = self._heat_capacity()
                coupling_factor = self._trotter_G / heat_capacity
                self.ttsys.electronic_temp *= math.exp(-coupling_factor * dt)
            
            half_step_linear()
            full_step_exp()
            half_step_linear()
        else:
            # Euler method
            
            heat_capacity = self._heat_capacity()
            radiation_power = self._radiation_power()
            coupling_power = np.sum(
                self._thermostat_forces * self.ttsys.velocities)
            coupling_power *= self.ttsys.Ninv
            del_temp = (radiation_power - coupling_power) / heat_capacity * dt
            self.ttsys.electronic_temp += del_temp
            self.radiation_energy_dumped += radiation_power * dt
                        
    def _update_forces(self) -> None:
        """Recompute the nuclear forces in ttsys."""
        self.ttsys.zero_forces()
        self.material.precompute(self.ttsys)
        
        # Interatomic
        self.ttsys.forces += self.material.nuclear_forces(
            self.ttsys, precomputed=True)
        
        # Langevin thermostat
        self._thermostat_forces = self._langevin_forces()
        self.ttsys.forces += self._thermostat_forces
        
        # Corrective
        if self.corrective is not None:
            self.ttsys.forces += self.corrective.nuclear_forces(
                self.ttsys, self.material)

    def _ep_coupling_init(self, dt: float) -> None:
        """Initialise electron-phonon force evaluations.

        Args:
            dt: The integration time step that will be used.
        """
        self._gamma1 = -self.material.ep_coupling
        self._gamma2 = math.sqrt(
            24. * self.material.ep_coupling * self.ttsys.electronic_temp / dt)
        self._trotter_G = 3. * self.material.ep_coupling / self.ttsys.mass

    def _langevin_forces(self) -> np.ndarray:
        """Forces on the nuclei from the Langevin thermostat."""
        return (self._gamma1 * self.ttsys.velocities + 
                self._gamma2 * np.random.uniform(
                    -0.5, 0.5, size=self.ttsys.forces.shape))

    def _heat_capacity(self) -> float:
        """Heat capacity plus corrective contribution at the present moment."""
        heat_capacity = self.material.heat_capacity(
            self.ttsys, precomputed=True)[0]
        if self.corrective is not None:
            heat_capacity += self.corrective.heat_capacity(
                self.ttsys, self.material)
        return heat_capacity

    def _radiation_power(self) -> float:
        """Radiation power at the present moment."""
        return (self.radiation.power(self.time)
                if self.radiation is not None else 0.0)
