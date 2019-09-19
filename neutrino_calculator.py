import numpy as np

from source.source_model import Source, DIFFUSE, POINT
from effective_area import EffectiveArea

"""
Module for calculating the number of neutrinos,
given a flux model and effective area.
"""

M_TO_CM = 100.0
GEV_TO_TEV = 1.0e-3
YEAR_TO_SEC = 3.154e7

class NeutrinoCalculator():
    """
    Calculate the expected number of detected neutrinos.
    """

    def __init__(self, source, detector):
        """
        Calculate the expected number of detected neutrinos.
        
        :param source: A Source instance.
        :param effective_area: An EffectiveArea instance.
        """

        self._source = source

        self._effective_area = effective_area

    @property 
    def source(self):

        return self._source

    @source.setter
    def source(self, value):

        if not isinstance(value, Source):

            raise ValueError(str(value) + ' is not an instance of Source')

        else:

            self._source = value

    @property
    def effective_area(self):

        return self._effective_area

    @effective_area.setter
    def effective_area(self, value):

        if not isinstance(value, EffectiveArea):

            raise ValueError(str(value) + ' is not an instance of EffectiveArea')
        
        else:

            self._effective_area = value

            
    def _diffuse_calculation(self):

        dN_dt = 0
        for i, E in enumerate(self.effective_area.true_energy_bins[:-1]):

            Em = E * GEV_TO_TEV # TeV
            EM = self.effective_area.true_energy_bins[i+1] * GEV_TO_TEV # TeV
            
            for j, czm in enumerate(self.effective_area.cos_zenith_bins[:-1]):

                czM = self.effective_area.cos_zenith_bins[j+1]
    
                integrated_flux = ( self.source.flux_model.integrated_spectrum(Em, EM)
                                    * (czM - czm) * 2*np.pi )
                
                aeff = self._selected_effective_area_values[i][j] * M_TO_CM**2
                
                dN_dt += aeff * integrated_flux

        return dN_dt * self._time

    
    def _select_single_cos_zenith(self):

        # cos(zenith) = -sin(declination)
        cos_zenith = -np.sin(self.source.coordinate.dec.rad)

        selected_bin_index = np.digitize(cos_zenith, self.effective_area.cos_zenith_bins) - 1

        return selected_bin_index
    

    def _point_source_calculation(self):

        selected_bin_index = self._select_single_cos_zenith()

        dN_dt = 0
        
        for i, E in enumerate(self.effective_area.true_energy_bins[:-1]):

            Em = E * GEV_TO_TEV # TeV
            EM = self.effective_area.true_energy_bins[i+1] * GEV_TO_TEV # TeV

            integrated_flux = self.source.flux_model.integrated_spectrum(Em, EM)

            aeff = self._selected_effective_area_values.T[selected_bin_index][i] * M_TO_CM**2

            dN_dt += aeff * integrated_flux

        return dN_dt * self._time

        
        
    def __call__(self, time=1, min_energy=1e2, max_energy=1e9, min_cosz=-1, max_cosz=1):
        """
        Calculate the number of expected neutrinos, 
        taking into account the observation time and
        possible further constraints on the effective
        area as a function of energy and cos(zenith). 
        !! NB: We assume Aeff is zero outside of specified 
        energy and cos(zenith)!!
        :param time: Observation time in years.
        :param min_energy: Aeff energy lower bound [GeV].
        :param max_energy: Aeff energy upper bound [GeV].
        :param min_cosz: Aeff cos(zenith) lower bound.
        :param max_cosz: Aeff cos(zenith) upper bound.
        """

        self._time = time * YEAR_TO_SEC # s
        
        self._selected_effective_area_values = self.effective_area.values.copy()
        
        # @TODO: Add contribution from bins on boundary.
        self._selected_effective_area_values[self.effective_area.true_energy_bins[1:] < min_energy] = 0
        self._selected_effective_area_values[self.effective_area.true_energy_bins[:-1] > max_energy] = 0

        self._selected_effective_area_values.T[self.effective_area.cos_zenith_bins[1:] < min_cosz] = 0
        self._selected_effective_area_values.T[self.effective_area.cos_zenith_bins[:-1] > max_cosz] = 0
        
        if self.source.source_type == DIFFUSE:

            N = self._diffuse_calculation()
            
        elif self.source_model.source_type == POINT:

            N = self._point_source_calculation()

        else:

            raise ValueError(str(self.source.source_type) + ' is not recognised.')

        return N
