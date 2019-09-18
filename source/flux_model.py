import numpy as np
from abc import ABC, abstractmethod

"""
Module for simple flux models used in 
neutrino detection calculations
"""

class FluxModel(ABC):
    """
    Abstract base class for flux models.
    """        
        
    @abstractmethod
    def spectrum(self):

        pass

    @abstractmethod
    def integrated_spectrum(self):

        pass


    
class PowerLawFlux(FluxModel):
    """
    Power law flux models.
    """

    def __init__(self, normalisation, normalisation_energy, index, lower_energy,
                 upper_energy=np.inf):
        """
        Power law flux models. 

        :param normalisation: Flux normalisation [TeV^-1 cm^-2 s^-1 sr^-1] or [TeV^-1 cm^-2 s^-1] for point sources.
        :param normalisation energy: Energy at which flux is normalised [TeV].
        :param index: Spectral index of the power law.
        :param lower_energy: Lower energy bound [TeV].
        :param upper_energy: Upper enegry bound [TeV], unbounded by default.
        """

        super().__init__()
        
        self._normalisation = normalisation

        self._normalisation_energy = normalisation_energy

        self._index = index

        self._lower_energy = lower_energy
        self._upper_energy = upper_energy


    def spectrum(self, energy):
        """
        dN/dEdAdt or dN/dEdAdtdO depending on flux_type.
        """

        return self._normalisation * np.power(energy / self._normalisation_energy, -self._index)

    
    def integrated_spectrum(self, lower_energy_bound, upper_energy_bound):
        """
        \int spectrum dE over finite energy bounds.
        
        :param lower_energy_bound: [TeV]
        :param upper_energy_bound: [TeV]
        """

        if lower_energy_bound < self._lower_energy:
            lower_energy_bound = self._lower_energy

        if upper_energy_bound > self._upper_energy:
            upper_energy_bound = self._upper_energy

        norm = self._normalisation / ( np.power(self._normalisation_energy, -self._index) * (1 - self._index) )

        return norm * ( np.power(upper_energy_bound, 1-self._index) - np.power(lower_energy_bound), 1-self._index )

        
    
