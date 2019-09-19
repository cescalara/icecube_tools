import numpy as np
from abc import ABC, abstractmethod

from effective_area import EffectiveArea
from energy_resolution import EnergyResolution

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

            raise ValueError(str(value) + ' is not an instance of EffectiveArea')
        
        else:

            self._effective_area = value

    @property
    def energy_resolution(self):

        return self._energy_resolution

    @energy_resolution.setter
    def energy_resolution(self, value):

        if not isinstance(value, EnergyResolution):

            raise ValueError(str(value) + ' is not an instance of EnergyResolution')
        
          

class IceCube(Detector):
    """
    IceCube detector.
    """

    def __init__(self, effective_area, energy_resolution):
        """
        IceCube detector.

        :param effective_area: instance of EffectiveArea.
        :param energy_resolution: instance of EnergyResolution.
        """

        super().__init__()

        self.effective_area = effective_area

        self.energy_resolution = energy_resolution
        
