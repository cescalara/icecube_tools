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


class PointSourceLikelihood():
    """
    Calculate the point source likelihood for a given 
    neutrino dataset - in terms of reconstructed 
    energies and arrival directions.
    """    
    
    def __init__(self, direction_likelihood, energy_likelihood, event_coords, energies, source_coord):
        """
        Calculate the point source likelihood for a given 
        neutrino dataset - in terms of reconstructed 
        energies and arrival directions.
        
        :param direction_likelihood: An instance of SpatialGaussianLikelihood.
        :param energy_likelihood: An instance of MarginalisedEnergyLikelihood.
        :param event_coords: List of (ra, dec) tuples for reconstructed coords.
        :param energies: The reconstructed nu energies.
        :param source_coord: (ra, dec) pf the point to test.
        """

        self._direction_likelihood = direction_likelihood 

        self._energy_likelihood = energy_likelihood
        
        self._band_width = 5 * self._direction_likelihood._sigma # degrees

        self._event_coords = event_coords
        
        self._energies = energies

        self._source_coord = source_coord

        self._bg_index = 3.7

        self._select_declination_band()


    def _select_declination_band(self):

        decs = np.array([_[1] for _ in self._event_coords])

        _, source_dec = self._source_coord

        dec_fac = np.deg2rad(self._band_width)
        
        selected = np.where((decs >= source_dec - dec_fac) & (decs <= source_dec + dec_fac) )[0]

        self._selected = selected
        
        self._selected_energies = self._energies[selected]

        self._selected_event_coords = [(ec[0], ec[1]) for ec in self._event_coords
                                       if (ec[1] >= source_dec - dec_fac) & (ec[1] <= source_dec + dec_fac)]
        
        self.N = len(selected)
        
        
    def _signal_likelihood(self, event_coord, source_coord, energy, index):

        return self._direction_likelihood(event_coord, source_coord) * self._energy_likelihood(energy, index)


    def _background_likelihood(self, energy):

        return self._energy_likelihood(energy, self._bg_index) / np.deg2rad(self._band_width)
        #return 1  / np.deg2rad(self._band_width)
 
        
    def __call__(self, ns, index):
        """
        Evaluate the PointSourceLikelihood for the given
        neutrino dataset.

        returns -log(likelihood_ratio) for minimization.

        Uses calculation described in:
        https://github.com/IceCubeOpenSource/SkyLLH/blob/master/doc/user_manual.pdf

        :param ns: Number of source counts.
        :param index: Spectral index of the source.
        """
        
        
        log_likelihood_ratio = 0.0
        
        for i in range(self.N):
            
            signal = self._signal_likelihood(self._selected_event_coords[i],
                                                             self._source_coord, self._selected_energies[i], index)

            bg = self._background_likelihood(self._selected_energies[i])

            chi = (1 / self.N) * (signal/bg - 1)
            
            log_likelihood_ratio += np.log(1 + ns * chi)
                        
        return -log_likelihood_ratio
        
                
class MarginalisedEnergyLikelihood():
    """
    Compute the marginalised energy likelihood by using a 
    simulation of a large number of reconstructed muon 
    neutrino tracks. 
    """
    
    
    def __init__(self, energy, sim_index=1.5, min_index=1.5, max_index=4.0, min_E=1e2, max_E=1e9):
        """
        Compute the marginalised energy likelihood by using a 
        simulation of a large number of reconstructed muon 
        neutrino tracks. 
        
        :param energy: Reconstructed muon energies (preferably many).
        :param sim_index: Spectral index of source spectrum in sim.
        """

        self._energy = energy

        self._sim_index = sim_index

        self._min_index = min_index
        self._max_index = max_index

        self._min_E = min_E
        self._max_E = max_E
        
        self._index_bins = np.linspace(min_index, max_index)

        self._energy_bins = np.linspace(np.log10(min_E), np.log10(max_E)) # GeV

        self._precompute_histograms()

        
    def _calc_weights(self, new_index):

        return  np.power(self._energy, self._sim_index - new_index)

    
    def _precompute_histograms(self):

        self._likelihood = []
        
        for index in self._index_bins:

            weights = self._calc_weights(index)

            hist, _ = np.histogram(np.log10(self._energy), bins=self._energy_bins,
                                   weights=weights, density=True)

            self._likelihood.append(hist)
        

    def __call__(self, E, new_index):
        """
        P(Ereco | index) = \int dEtrue P(Ereco | Etrue) P(Etrue | index)
        """

        if E < self._min_E or E > self._max_E:

            raise ValueError('Energy ' + str(E) + 'is not in the accepted range between '
                             + str(self._min_E) + ' and ' + str(self._max_E))

        if new_index < self._min_index or new_index > self._max_index:

            raise ValueError('Sepctral index ' + str(new_index) + ' is not in the accepted range between '
                             + str(self._min_index) + ' and ' + str(self._max_index))
        
        i_index = np.digitize(new_index, self._index_bins) - 1
        
        E_index = np.digitize(np.log10(E), self._energy_bins) - 1

        return self._likelihood[i_index][E_index]
        


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
