import abc
from typing import Callable

from two_temp_system import TwoTempSystem


class Symplectic(abc.ABC):
    """Base class to define a nuclear symplectic integrator."""
    
    @abc.abstractmethod
    def update(self, ttsys: TwoTempSystem,
               dt: float,
               update_forces: Callable[[], None]) -> None:
        """Advance the system by a single time step.

        Args:
            ttsys: Two temp system to be integrated.
            dt: Time step.
            update_forces: A function that updates the forces in ttsys.
        """
        raise NotImplementedError()


class VVSymplectic(Symplectic):
    """Velocity Verlet symplectic integrator."""
    
    def update(self, ttsys: TwoTempSystem,
               dt: float,
               update_forces: Callable[[], None]) -> None:
        """Advance the system by a single time step using velocity Verlet.
        
        Args:
            ttsys: Two temp system to be integrated.
            dt: Time step.
            update_forces: A function that updates the forces in ttsys.
        """
        reuse = dt * 0.5 * ttsys.massinv
        ttsys.positions += ttsys.velocities * dt + ttsys.forces * dt * reuse
        ttsys.velocities += ttsys.forces * reuse
        update_forces()
        ttsys.velocities += ttsys.forces * reuse


class LFSymplectic(Symplectic):
    """Leapfrog Verlet symplectic integrator."""
    
    def update(self, ttsys: TwoTempSystem,
               dt: float,
               update_forces: Callable[[], None]) -> None:
        """Advance the system by a single time step using kick-drift-kick.
        
        Args:
            ttsys: Two temp system to be integrated.
            dt: Time step.
            update_forces: A function that updates the forces in ttsys.
        """
        reuse = dt * 0.5 * ttsys.massinv
        ttsys.velocities += ttsys.forces * reuse
        ttsys.positions += ttsys.velocities * dt
        update_forces()
        ttsys.velocities += ttsys.forces * reuse

