from abc import ABC

from .effective_area import EffectiveArea
from .energy_resolution import EnergyResolution
from .angular_resolution import AngularResolution

"""
Detector modules, bringing together 
effective area and energy resolution.
"""


class Detector(ABC):
    """
    Abstract base class for a neutrino detector.
    """

    @property
    def effective_area(self):

        return self._effective_area

    @effective_area.setter
    def effective_area(self, value):

        if not isinstance(value, EffectiveArea):

            raise ValueError(str(value) + " is not an instance of EffectiveArea")

        else:

            self._effective_area = value

    @property
    def energy_resolution(self):

        return self._energy_resolution

    @energy_resolution.setter
    def energy_resolution(self, value):

        if not isinstance(value, EnergyResolution):

            raise ValueError(str(value) + " is not an instance of EnergyResolution")

    @property
    def angular_resolution(self):

        return self._angular_resolution

    @angular_resolution.setter
    def angular_resolution(self, value):

        if not isinstance(value, AngularResolution):

            raise ValueError(str(value) + " is nmot an instance of AngularResolution.")

        else:

            self._angular_resolution = value


class IceCube(Detector):
    """
    IceCube detector.
    """

    def __init__(self, effective_area, energy_resolution, angular_resolution):
        """
        IceCube detector.

        :param effective_area: instance of EffectiveArea.
        :param energy_resolution: instance of EnergyResolution.
        :param angular_resolution: instance of AngularResolution.
        """

        self._effective_area = effective_area

        self._energy_resolution = energy_resolution

        self._angular_resolution = angular_resolution

        super().__init__()
