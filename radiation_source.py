import math


class RadiationSource:
    """A class to define radiation sources. Zero radiation by default."""
    
    def power(self, time: float) -> float:
        """Radiation power Q delivered to the electrons as a function of time.

        Args:
            time: Time.
        """
        return 0.


class Laser(RadiationSource):
    """Laser pulse with a Gaussian time profile."""
    
    _time0: float
    _fwhm: float
    _fluence: float
    _coeff1: float
    _coeff2: float
    
    def __init__(self, time0: float, fwhm: float, fluence: float):
        """Setup the laser.

        Args:
            time0: Time of maximum power.
            fwhm: Full width at half maximum.
            fluence: Absorbed fluence per atom.
        """
        self._time0 = time0
        self._fwhm = fwhm
        self._fluence = fluence
        self._coeff1 = fluence / fwhm * math.sqrt(4.*math.log(2.)/math.pi)
        self._coeff2 = 4. * math.log(2.) / fwhm**2

    def power(self, time: float) -> float:
        """Radiation power Q delivered to the electrons as a function of time.
        
        Args:
            time: Time.
        """
        return self._coeff1 * math.exp(-self._coeff2 * (time-self._time0)**2)
