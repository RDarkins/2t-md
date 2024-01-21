import abc
import math
from typing import Optional, Tuple

import numpy as np
import scipy.integrate
import scipy.interpolate

from material import Material
from two_temp_system import TwoTempSystem


class Corrective(abc.ABC):
    """Deriving classes introduce an auxiliary energy term to the Hamiltonian
    that may be a functional of the interatomic potential. This corrective
    energy is intended to maintain a positive heat capacity at all times.
    """
    
    @abc.abstractmethod
    def energy(self, ttsys: TwoTempSystem, material: Material) -> float:
        """Auxiliary energy to be added to the Hamiltonian.

        Args:
            ttsys: Two temperature system.
            material: Material.
        Returns:
            Corrective auxiliary energy.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def heat_capacity(self, ttsys: TwoTempSystem, material: Material) -> float:
        """Contribution of the auxiliary energy to the heat capacity.

        Args:
            ttsys: Two temperature system.
            material: Material.
        Returns:
            The derivative of energy() w.r.t. the electronic temperature.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def nuclear_forces(self, ttsys: TwoTempSystem,
                       material: Material) -> np.ndarray:
        """Forces on the nuclei due to the auxiliary energy.
        
        Args:
            ttsys: Two temperature system.
            material: Material.
        Returns:
            The nuclear forces, -grad energy(), in the format of ttsys.forces.
        """
        raise NotImplementedError()


def _integrate_with_spline(x_samples: np.ndarray,
                           y_samples: np.ndarray,
                           x0: float,
                           x1: float,
                           delta_x: float):
    """Fit a cubic spline to {x_samples, y_samples} and integrate over
    the range [x0, x1] using the Simpson method, sampling the spline
    at intervals of delta_x. The interval [x0, x1] must fall within the
    range of x_samples.

    Adaptive integration should not be used because y_samples may be
    high-dimensional.
    """
    finer_x_samples = np.arange(x0, x1 + 0.5* delta_x, delta_x)
    finer_x_samples = np.append(finer_x_samples, x1)
    interp = scipy.interpolate.CubicSpline(x_samples, y_samples)
    finer_y_samples = interp(finer_x_samples)
    return scipy.integrate.simpson(finer_y_samples, finer_x_samples, axis=0)


class DarkinsCorrective(Corrective):
    """This class implements the corrective scheme presented in ref [1].

    References:
        [1] Darkins, et al. Phys. Rev. B 98.2 (2018): 024304.

    Attributes:
        multiplier: Coefficient for evaluating epsilon.
        delta_sample: Sample forces in temperature increments of delta_sample.
        delta_integrate: Integrate forces over temperature using spline-
            generated samples at intervals of delta_integrate.
    """

    multiplier: float    
    delta_sample: float
    delta_integrate: float

    def __init__(self, multiplier: float,
                 delta_sample: float,
                 delta_integrate: Optional[float] = None):
        self.multiplier = multiplier
        self.delta_sample = delta_sample
        self.delta_integrate = delta_integrate or delta_sample

    def energy(self, ttsys: TwoTempSystem, material: Material) -> float:
        """The auxiliary energy W is defined by equation 8 in ref [1]:

            W = \int_{0}^{T_e} _w(s) ds

        where T_e is the electronic temperature and _w() is defined below.

        Args:
            ttsys: Two temperature system.
            material: Material.
        
        Returns:
            Corrective auxiliary energy.
        """
        temp_samples = np.arange(
            0., ttsys.electronic_temp + self.delta_sample*1.5, self.delta_sample)
        integrand_samples = self._w(temp_samples, ttsys, material)[0]
        integral = _integrate_with_spline(
            temp_samples, integrand_samples,
            0., ttsys.electronic_temp,
            self.delta_integrate) * ttsys.N
        return integral

    def heat_capacity(self, ttsys: TwoTempSystem, material: Material) -> float:
        """The heat capacity evaluates to _w(T_e).

        Args:
            ttsys: Two temperature system.
            material: Material.
        
        Returns:
            The derivative of energy() w.r.t. the electronic temperature.
        """
        return self._w(ttsys.electronic_temp, ttsys, material)[0]

    def nuclear_forces(self, ttsys: TwoTempSystem,
                       material: Material) -> np.ndarray:
        """The material forces are evaluated across a range of temperatures
        and then integrated with splines. See equation 10 in ref [1].
        
        Args:
            ttsys: Two temperature system.
            material: Material.
        
        Returns:
            The nuclear forces, -grad energy(), in the format of ttsys.forces.
        """
        temp_samples = np.arange(
            0., ttsys.electronic_temp + self.delta_sample*1.5, self.delta_sample)
        w, w_x, w_xx, w_xy = self._w(temp_samples, ttsys, material)
        _, Ce_d = material.heat_capacity(
            ttsys, temp_samples, precomputed=True)
        _, eps_d = self._eps(temp_samples, ttsys, material)
        mat_forces = material.nuclear_forces(
            ttsys, temp_samples, precomputed=True)
        coeff = w_xx * Ce_d + w_xy * eps_d
        integrand_samples = coeff[:, np.newaxis, np.newaxis] * mat_forces
        integral = _integrate_with_spline(
            temp_samples, integrand_samples,
            0., ttsys.electronic_temp,
            self.delta_integrate)
        corrective_forces = w_x[-1] * mat_forces[-1] - integral
        return corrective_forces

    def _w(self, electronic_temp: np.ndarray,
           ttsys: TwoTempSystem,
           material: Material
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the w(x, y) function, and some partial derivatives, defined in
        equation A1 of ref [1], where x = heat capacity(T_e) and y = eps(T_e).

        Args:
            electronic_temp: An array of electronic temperatures.
            ttsys: Two temperature system.
            material: Material.
        Returns:
            A tuple (w, w_x, w_xx, w_xy) where w_x is the partial derivative
            dw(x,y)/dx, etc, and each element is an array evaluated at the
            corresponding temperatures defined in electronic_temp.
        """
        x, *_ = material.heat_capacity(ttsys, electronic_temp, precomputed=True)
        y, *_ = self._eps(electronic_temp, ttsys, material)
        yinv = 1. / y
        sin1 = np.sin(math.pi*x*yinv)
        sin3 = np.sin(3.*math.pi*x*yinv)
        cos1 = np.cos(math.pi*x*yinv)
        cos3 = np.cos(3.*math.pi*x*yinv)
        f1_16 = 1. / 16.
        f3_16 = 3. / 16.
        f9_16 = 9. / 16.
        w = tuple(np.zeros_like(electronic_temp) for _ in range(4))
        w = np.where(
            x <= 0.,
            (
                0.5*y - x,
                np.full_like(x, -1.),
                np.zeros_like(x),
                np.zeros_like(x)
            ),
            w)
        w = np.where(
            (x > 0.) & (x < y),
            (
                -f9_16*y/math.pi*sin1 + f1_16*y/(3.*math.pi)*sin3 - 0.5*x + 0.5*y,
                -f9_16*cos1 + f1_16*cos3 - 0.5,
                f9_16*math.pi*yinv*sin1 -f3_16*math.pi*yinv*sin3,
                (-f9_16*math.pi*x*sin1 + f3_16*math.pi*x*sin3)*yinv*yinv
            ),
            w)
        return w

    def _eps(self, electronic_temp: np.ndarray,
             ttsys: TwoTempSystem,
             material: Material
    ) -> Tuple[np.ndarray, np.ndarray]:
        """If the heat capacity drops below this function, the corrective term
        kicks in. This eps function is proportional to dE/dT_e.
        
        Args:
            electronic_temp: An array of electronic temperatures.
            ttsys: Two temperature system.
            material: Material.
        Returns:
            A tuple (eps, eps_d) where eps_d is d(eps)/dT_e.
        """
        _, E_d, E_dd = material.electronic_energy(electronic_temp)
        eps = self.multiplier * E_d
        eps_d = self.multiplier * E_dd
        return eps, eps_d
