import abc
import math
from typing import Any, Callable, Optional, Tuple

import numpy as np

from two_temp_system import TwoTempSystem


class Material(abc.ABC):
    """This base class defines the material properties.
    
    Specifically it defines the interatomic potentials, the electronic
    energy, the electron-phonon coupling, and the electronic heat capacity.

    Attributes:
        ep_coupling: Langevin damping constant (dimension mass/time).
    """
    
    ep_coupling: float
    
    def __init__(self, ep_coupling: float):
        self.ep_coupling = ep_coupling

    def precompute(self, ttsys: TwoTempSystem) -> None:
        """Any method in this class that accepts a `precomputed` argument
        will call this method if `precomputed` is False.

        It is intended to precompute expensive terms that depend on
        ttsys.positions and that are independent of the electronic temp,
        for optimisation purposes. Its use is optional.

        Args:
            ttsys: Two temperature system.
        """
        pass

    @staticmethod
    def _precomputable(method: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to call precompute() if `precomputed` is False."""
        def wrapper(self: 'Material', ttsys: TwoTempSystem,
                    electronic_temp: Optional[np.ndarray] = None,
                    precomputed: bool = False):
            if not precomputed:
                self.precompute(ttsys)
            return method(self, ttsys, electronic_temp)
        return wrapper

    @_precomputable
    def nuclear_potential(self, ttsys: TwoTempSystem,
                          electronic_temp: Optional[np.ndarray] = None,
                          precomputed: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nuclear potential energy U and its derivatives w.r.t. T_e.
        
        Args:
            ttsys: Two temperature system.
            electronic_temp: An optional array of electronic temperatures. If
                None, the electronic temperature specified in ttsys is used.
            precomputed: If False, call the precompute() method first.

        Returns:
            A tuple (U, U_d, U_dd) where U_d is dU/dT_e etc. Each element is
            evaluated for each electronic temperature requested.
        """
        return self._nuclear_potential_impl(ttsys, electronic_temp)

    @_precomputable
    def nuclear_forces(self, ttsys: TwoTempSystem,
                       electronic_temp: Optional[np.ndarray] = None,
                       precomputed: bool = False) -> np.ndarray:
        """Force -grad U on each nucleus.
        
        Args:
            ttsys: Two temperature system.
            electronic_temp: An optional array of electronic temperatures. If
                None, the electronic temperature specified in ttsys is used.
            precomputed: If False, call the precompute() method first.

        Returns:
            If electronic_temp is none, the forces are returned in the format
            of ttsys.forces. Otherwise an array is returned where the i-th
            element equals the forces (in the format of ttsys.forces) evaluated
            at the temperature electronic_temp[i].
        """
        return self._nuclear_forces_impl(ttsys, electronic_temp)

    @_precomputable
    def heat_capacity(self, ttsys: TwoTempSystem,
                      electronic_temp: Optional[np.ndarray] = None,
                      precomputed: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Electronic heat capacity per nucleus and its derivative w.r.t. T_e.
        
        Args:
            ttsys: Two temperature system.
            electronic_temp: An optional array of electronic temperatures. If
                None, the electronic temperature specified in ttsys is used.
            precomputed: If False, call the precompute() method first.

        Returns:
            A tuple (Ce, dCe/dTe) where each element is an array evaluated at 
            the corresponding temperatures in electronic_temp.
        """
        electronic_temp = (ttsys.electronic_temp if electronic_temp is None
                           else electronic_temp)
        _, U_d, U_dd = self.nuclear_potential(
            ttsys, electronic_temp, precomputed=True)
        _, E_d, E_dd = self.electronic_energy(electronic_temp)
        U_d *= ttsys.Ninv
        U_dd *= ttsys.Ninv
        return (U_d + E_d), (U_dd + E_dd)

    @abc.abstractmethod
    def _nuclear_potential_impl(self, ttsys: TwoTempSystem,
                                electronic_temp: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Implementation of nuclear_potential(), to be overridden. The precompute()
        method is guaranteed to be called before this method.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _nuclear_forces_impl(
            self, ttsys: TwoTempSystem, electronic_temp: Optional[np.ndarray]
    ) -> np.ndarray:
        """Implementation of nuclear_forces(), to be overridden. The precompute()
        method is guaranteed to be called before this method.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def electronic_energy(self, electronic_temp: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Electronic energy per nucleus and its derivatives w.r.t. T_e.

        Args:
            electronic_temp: Electronic temperature.

        Returns:
            A tuple (E, E_d, E_dd) where E_d is the first order derivative, etc.
        """
        raise NotImplementedError()


class ToyMaterial(Material):
    """Toy model material contrived to produce a negative electronic heat
    capacity at high electronic temperatures.
    
    Lennard-Jones interatomic potential of the form

        U(r) = A(T_e) * [1/r^{12} - 2/r^{6}]

    where T_e is the electronic temp, r is the interatomic distance, and
    
        A(T_e) = A_0 if T_e < excited_temp
                 A_0 * [1 + 1/2*(T_e - excited_temp)^2] else

    The electronic energy takes the form

        E(T_e) = E_0 * (T_e + T_e^2)

    Attributes:
        A0: The Lennard-Jones coefficient below the excited temperature.
        E0: The electronic energy coefficient.
        excited_temp: The electronic temperature beyond which U changes shape.
    """
    
    A0: float
    E0: float
    excited_temp: float    
    _precomputed_pot: float = 0.
    _precomputed_forces: np.ndarray = None

    def __init__(self, ep_coupling: float = 0.1,
                 A0: float = 1.,
                 E0: float = 1.,
                 excited_temp: float = 1.):
        super().__init__(ep_coupling)
        self.A0 = A0
        self.E0 = E0
        self.excited_temp = excited_temp

    def precompute(self, ttsys: TwoTempSystem) -> None:
        """Evaluate the electronic temperature independent parts of
        the Lennard-Jones potential U.
        
        Args:
            ttsys: Two temperature system.
        """
        
        # Here we evaluate:
        #     _precomputed_pot = U / A(T_e)
        #     _precomputed_forces[i] = -grad_i U / A(T_e)
        
        self._precomputed_pot = 0.
        self._precomputed_forces = np.empty_like(ttsys.forces)
        for i in range(ttsys.N):
            disp_i = ttsys.positions - ttsys.positions[i]
            disp_i[i][0] = 1. # Avoid divide by zero
            r2inv_i = 1. / np.sum(disp_i**2, axis=1)
            r2inv_i[i] = 0. # Remove self-interaction
            r6inv_i = r2inv_i * r2inv_i * r2inv_i
            pot_i = np.sum(r6inv_i * (r6inv_i - 2.))
            force_ij = 12. * r6inv_i * (r6inv_i - 1.) * r2inv_i
            force_i = - np.sum(disp_i * force_ij[:, np.newaxis], axis=0)
            self._precomputed_pot += pot_i
            self._precomputed_forces[i] = force_i
        self._precomputed_pot *= 0.5 # Correct for full neighbour list

    def _nuclear_potential_impl(self, ttsys: TwoTempSystem,
                                electronic_temp: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nuclear potential energy U and its derivatives w.r.t. T_e. See
        nuclear_potential()."""
        electronic_temp = (ttsys.electronic_temp if electronic_temp is None
                           else electronic_temp)
        A, A_d, A_dd = self._A(electronic_temp)
        U = A * self._precomputed_pot
        U_d = A_d * self._precomputed_pot
        U_dd = A_dd * self._precomputed_pot
        return U, U_d, U_dd

    def _nuclear_forces_impl(
            self, ttsys: TwoTempSystem, electronic_temp: Optional[np.ndarray]
    ) -> np.ndarray:
        """Force -grad U on each nucleus. See nuclear_forces()."""
        if electronic_temp is None:
            A, *_ = self._A(ttsys.electronic_temp)
            return A * self._precomputed_forces
        else:
            A_array, *_ = self._A(electronic_temp)
            return A_array[:, np.newaxis, np.newaxis] * self._precomputed_forces

    def electronic_energy(self, electronic_temp: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Electronic energy per nucleus and its T_e derivatives.

        Args:
            electronic_temp: Electronic temperatures to evaluate.

        Returns:
            A tuple (E, E_d, E_dd) where E_d is the first order derivative, etc.
        """
        E = self.E0*electronic_temp + self.E0*electronic_temp*electronic_temp
        E_d = self.E0 + 2.*self.E0*electronic_temp
        E_dd = 2. * self.E0
        return E, E_d, E_dd

    def _A(self, electronic_temp: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Electronic temperature dependent Lennard-Jones coefficient A and
        its derivatives w.r.t. T_e.
        
        Args:
            electronic_temp: Electronic temperature.

        Returns:
            (A, A_d, A_dd) where A_d is the first order derivative etc.
        """
        result = (
            np.full_like(electronic_temp, self.A0),
            np.zeros_like(electronic_temp),
            np.zeros_like(electronic_temp)
        )
        result = np.where(
            electronic_temp > self.excited_temp,
            (
                0.5*self.A0*(electronic_temp-self.excited_temp)**2 + self.A0,
                self.A0*(electronic_temp-self.excited_temp),
                np.full_like(electronic_temp, self.A0)
            ),
            result)
        return result
