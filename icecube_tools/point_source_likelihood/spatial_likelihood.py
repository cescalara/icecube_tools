import numpy as np


"""
Module to compute the IceCube point source likelihood
using publicly available information.

Based on the method described in:
Braun, J. et al., 2008. Methods for point source analysis 
in high energy neutrino telescopes. Astroparticle Physics, 
29(4), pp.299–305.

Currently well-defined for searches with
Northern sky muon neutrinos.
"""

class SpatialGaussianLikelihood():
    """
    Spatial part of the point source likelihood.

    P(x_i | x_s) where x is the direction (unit_vector).
    """
    

    def __init__(self, angular_resolution):
        """
        Spatial part of the point source likelihood.
        
        P(x_i | x_s) where x is the direction (unit_vector).
        
        :param angular_resolution; Angular resolution of detector [deg]. 
        """

        # @TODO: Init with some sigma as a function of E?
        
        self._sigma = angular_resolution

    
    def __call__(self, event_coord, source_coord):
        """
        Use the neutrino energy to determine sigma and 
        evaluate the likelihood.

        P(x_i | x_s) = (1 / (2pi * sigma^2)) * exp( |x_i - x_s|^2/ (2*sigma^2) )

        :param event_coord: (ra, dec) of event [rad].
        :param source_coord: (ra, dec) of point source [rad].
        """

        sigma_rad = np.deg2rad(self._sigma)

        ra, dec = event_coord
                
        src_ra, src_dec = source_coord
        
        norm = 0.5 / (np.pi * sigma_rad**2)

        # Calculate the cosine of the distance of the source and the event on
        # the sphere.
        cos_r = np.cos(src_ra - ra) * np.cos(src_dec) * np.cos(dec) + np.sin(src_dec) * np.sin(dec)
        
        # Handle possible floating precision errors.
        if cos_r < -1.0:
            cos_r = 1.0
        if cos_r > 1.0:
            cos_r = 1.0

        r = np.arccos(cos_r)
         
        dist = np.exp( -0.5*(r / sigma_rad)**2 )

        return norm * dist
