import numpy as np

from source.source_model import Source
from effective_area.effective_area import IceCubeEffectiveArea

"""
Module for calculating the number of neutrinos,
given a flux model and effective area.
"""

M_TO_CM = 100.0
GEV_TO_TEV = 1.0e-3


class NeutrinoCalculator():
    """
    Calculate the expected number of detected neutrinos.
    """

    def __init__(self, source_model, effective_area):
        """
        Calculate the expected number of detected neutrinos.
        
        :param source_model: A Source instance.
        :param effective_area: An IceCubeEffectiveArea instance.
        """

        self._source_model = source_model

        self._effective_area = effective_area

    @property 
    def source_model(self):

        return self._source_model

    @source_model.setter
    def source_model(self, value):

        if not isinstance(value, Source):

            raise ValueError(str(value) + ' is not an instance of Source')

        else:

            self._source_model = value

    @property
    def effective_area(self):

        return self._effective_area

    @effective_area.setter
    def effective_area(self, value):

        if not isinstance(value, IceCubeEffectiveArea):

            raise ValueError(str(value) + ' is not an instance of IceCubeEffectiveArea')
        
        else:

            self._effective_area = value

            
    def _diffuse_calculation(self):

        dN_dt = 0
        for i, E in enumerate(self.effective_area.true_energy_bins[:-1]):

            Em = E * GEV_TO_TEV # TeV
            EM = self.effective_area.true_energy_bins[i+1] * GEV_TO_TEV # TeV

            for j, czm in enumerate(self.effective_area.true_cosz_bins[:-1]):

                czM = self.effective_area.cos_zenith_bins[j+1]

                integrated_flux = ( self.flux_model.integrated_spectrum(Em, EM)
                                    * (czM - czm) * 2*np.pi )

                aeff = self.effective_area.values[i][j] * M_TO_CM**2

                dN_dt += aeff * integrated_flux

        return dN_dt

    
    def _select_single_cos_zenith(self):

        # cos(zenith) = -sin(declination)
        cos_zenith = -np.sin(self.coordinate.dec.rad)

        selected_bin_index = np.digitize(cos_zenith, self.effective_area.cos_zenith_bins) - 1

        return selected_bin_index
    

    def _point_source_calculation(self):

        selected_bin_index = self._select_single_cos_zenith()

        dN_dt = 0
        
        for i, E in enumerate(self.effective_area.true_energy_bins[:-1]):

            Em = E * GEV_TO_TEV # TeV
            EM = self.effective_area.true_energy_bins[i+1] * GEV_TO_TEV # TeV

            integrated_flux = self.flux_model.integrated_spectrum(Em, EM)

            aeff = self.effective_area.values.T[selected_bin_index][i] * M_TO_CM**2

            dN_dt += aeff * integrated_flux

        return dN_dt

        
        
    def __call__(self, min_energy=None, max_energy=None, min_cosz=None, max_cosz=None):
        """
        
        """

        pass

        

        
