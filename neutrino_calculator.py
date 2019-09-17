import numpy as np

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

    def __init__(self, flux_model, effective_area):
        """
        Calculate the expected number of detected neutrinos.
        
        :param flux_model: A FluxModel instance.
        :param effective_area: An IceCubeEffectiveArea instance.
        """

        self._flux_model = flux_model

        self._effective_area = effective_area
        
        
    def _diffuse_calculation(self):

        dN_dt = 0
        for i, E in enumerate(effective_area.true_energy_bins[:-1]):

            Em = E * GEV_TO_TEV # TeV
            EM = effective_area.true_energy_bins[i+1] * GEV_TO_TEV # TeV

            for j, czm in enumerate(effective_area.true_cosz_bins[:-1]):

                czM = effective_area.cos_zenith_bins[j+1]

                integrated_flux = ( flux_model.integrated_spectrum(Em, EM)
                                    * (czM - czm) * 2*np.pi )

                aeff = effective_area.values[i][j] * M_TO_CM**2

                dN_dt += aeff * integrated_flux

        return dN_dt
            

    def _point_source_calculation(self):

        pass

        
        
        
    def __call__(self, min_energy=None, max_energy=None, min_cosz=None, max_cosz=None):
        """
        
        """

        pass

        

        
