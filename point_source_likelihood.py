import numpy as np
from iminuit import Minuit

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
        
        self._band_width = 3 * self._direction_likelihood._sigma # degrees

        self._event_coords = event_coords
        
        self._energies = energies

        self._source_coord = source_coord

        self._bg_index = 3.7

        self._ns_min = 0.0
        self._ns_max = 100.0
        self._max_index = 3.0

        self._select_nearby_events()

        self.Ntot = len(self._energies)
        

    def _select_nearby_events(self):

        ras = np.array([_[0] for _ in self._event_coords])

        decs = np.array([_[1] for _ in self._event_coords])

        source_ra, source_dec = self._source_coord

        dec_fac = np.deg2rad(self._band_width)
        
        selected = list( set(np.where((decs >= source_dec - dec_fac) & (decs <= source_dec + dec_fac)
                            & (ras >= source_ra - dec_fac) & (ras <= source_ra + dec_fac))[0]) )

        selected_dec_band = np.where((decs >= source_dec - dec_fac) & (decs <= source_dec + dec_fac))[0]
        
        self._selected = selected
        
        self._selected_energies = self._energies[selected]

        self._selected_event_coords = [(ec[0], ec[1]) for ec in self._event_coords
                                       if (ec[1] >= source_dec - dec_fac) & (ec[1] <= source_dec + dec_fac)
                                       & (ec[0] >= source_ra - dec_fac) & (ec[0] <= source_ra + dec_fac)]
        
        self.Nprime = len(selected)

        self.N = len(selected_dec_band)
        
        
    def _signal_likelihood(self, event_coord, source_coord, energy, index):

        return self._direction_likelihood(event_coord, source_coord) * self._energy_likelihood(energy, index)


    def _background_likelihood(self, energy):

        return self._energy_likelihood(energy, self._bg_index) / (np.deg2rad(self._band_width*2) * 2*np.pi)
 
        
    def _get_neg_log_likelihood_ratio(self, ns, index):
        """
        Calculate the -log(likelihood_ratio) for minimization.

        Uses calculation described in:
        https://github.com/IceCubeOpenSource/SkyLLH/blob/master/doc/user_manual.pdf

        :param ns: Number of source counts.
        :param index: Spectral index of the source.
        """
        
        one_plus_alpha = 1e-10 
        alpha = one_plus_alpha - 1
        
        log_likelihood_ratio = 0.0
        
        for i in range(self.Nprime):
            
            signal = self._signal_likelihood(self._selected_event_coords[i],
                                             self._source_coord, self._selected_energies[i], index)

            bg = self._background_likelihood(self._selected_energies[i])

            chi = (1 / self.N) * (signal/bg - 1)

            alpha_i = ns * chi
               
            if (1 + alpha_i) < one_plus_alpha:

                alpha_tilde = (alpha_i - alpha) / one_plus_alpha 
                log_likelihood_ratio += np.log1p(alpha) + alpha_tilde - (0.5 * alpha_tilde**2) 

            else:
                
                log_likelihood_ratio += np.log1p(alpha_i)

        log_likelihood_ratio += (self.N - self.Nprime) * np.log1p(-ns / self.N)
            
        return -log_likelihood_ratio

    
    def _get_log_likelihood(self, ns=0.0, index=None):
        """
        Calculate -log(likelihood) where likelihood is the 
        full point source likelihood. Negative is reutrned for
        easy minimization.

        Evaluated at the best fit ns and index, this is the 
        maximum likelihood for the source + background hypothesis.
        Evaluated at ns=0, index=None, this is the likelihood 
        for the background only hypothesis.

        :param ns: Number of source counts.
        :param index: Spectral index of source.
        """

        log_likelihood = 0.0

        for i in range(self.N):

            if index:
                
                signal = self._signal_likelihood(self._selected_event_coords[i], self._source_coord, self._selected_energies[i], index)
                S_i = (ns / self.N) * signal

            else:

                S_i = 0
                
            bg = self._background_likelihood(self._selected_energies[i])

            B_i = (1 - ns/self.N) * bg
            
            log_likelihood += np.log(S_i + B_i)

        return -log_likelihood

    
    def __call__(self, ns, index):
        """
        Wrapper function for convenience.
        """

        return self._get_neg_log_likelihood_ratio(ns, index)

    
    def _minimize(self):
        """
        Minimize -log(likelihood_ratio) for the source hypothesis, 
        returning the best fit ns and index.
        """

        m = Minuit(self._get_neg_log_likelihood_ratio, ns=0.0, index=self._max_index,
                   error_ns=0.1, error_index=0.1, errordef=0.5,
                   limit_ns=(self._ns_min, self._ns_max),
                   limit_index=(self._energy_likelihood._min_index, self._max_index))
        m.tol = 10
        m.migrad()

        if not m.migrad_ok() or not m.matrix_accurate():

            m = Minuit(self._get_neg_log_likelihood_ratio, ns=0.0, index=self._max_index, fix_index=True,
                       error_ns=0.1, error_index=0.1, errordef=0.5,
                       limit_ns=(self._ns_min, self._ns_max),
                       limit_index=(self._energy_likelihood._min_index, self._max_index))
            m.tol = 10
            m.migrad()

        self._best_fit_ns = m.values['ns']
        self._best_fit_index = m.values['index']

        
    def _first_derivative_likelihood_ratio(self, ns=0, index=2.0):
        """
        First derivative of the likelihood ratio. 
        Equation 41 in
        https://github.com/IceCubeOpenSource/SkyLLH/blob/master/doc/user_manual.pdf.  
        """

        one_plus_alpha = 1e-10
        alpha = one_plus_alpha - 1
        
        self._first_derivative = []
        
        for i in range(self.Nprime):

            signal = self._signal_likelihood(self._selected_event_coords[i], self._source_coord, self._selected_energies[i], index) 

            bg = self._background_likelihood(self._selected_energies[i])
            
            chi_i = (1 / self.N) * ((signal/bg) - 1)

            alpha_i = ns * chi_i
               
            if (1 + alpha_i) < one_plus_alpha:

                alpha_tilde = (alpha_i - alpha) / one_plus_alpha
            
                self._first_derivative.append( (1 / one_plus_alpha) * (1 - alpha_tilde) * chi_i )

            else:

                self._first_derivative.append( chi_i / (1 + alpha_i) )

        self._first_derivative = np.array(self._first_derivative)
                
        return sum(self._first_derivative) - ((self.N  - self.Nprime) / (self.N - ns))


    def _second_derivative_likelihood_ratio(self, ns=0):
        """
        Second derivative of the likelihood ratio.
        Equation 44 in
        https://github.com/IceCubeOpenSource/SkyLLH/blob/master/doc/user_manual.pdf.
        """

        self._second_derivative = -(self._first_derivative)**2 
            
        return sum(self._second_derivative) - ((self.N - self.Nprime) / (self.N - ns)**2)
        
            
    def get_test_statistic(self):
        """
        Calculate the test statistic for the best fit ns
        """

        self._minimize()

        #if self._best_fit_ns == 0:

        #    first_der = self._first_derivative_likelihood_ratio(self._best_fit_ns, self._best_fit_index)
        #    second_der = self._second_derivative_likelihood_ratio(self._best_fit_ns)

        #    self.test_statistic = -2 * (first_der**2 / (4 * second_der))

        #   self.likelihood_ratio = np.exp(-self.test_statistic/2)
            
        #else:

        neg_log_lik = self._get_neg_log_likelihood_ratio(self._best_fit_ns, self._best_fit_index)
        
        self.likelihood_ratio = np.exp(neg_log_lik)
        
        self.test_statistic = -2 * neg_log_lik
        
        return self.test_statistic

    
                
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

        return np.power(self._energy, self._sim_index - new_index)

    
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


class SimplePointSourceLikelihood():

    
    def __init__(self, direction_likelihood, event_coords, source_coord):
        """
        Point source likelihood with only spatial information.
        """

        self._direction_likelihood = direction_likelihood

        self._band_width = 3 * direction_likelihood._sigma
        
        self._event_coords = event_coords

        self._source_coord = source_coord

        self._select_declination_band()

        self.Ntot = len(self._event_coords)
        

    def _signal_likelihood(self, event_coord, source_coord):

        return self._direction_likelihood(event_coord, source_coord)

    
    def _background_likelihood(self):

        return 1 / (np.deg2rad(self._band_width*2) * 2*np.pi) 


    def _select_declination_band(self):

        decs = np.array([_[1] for _ in self._event_coords])

        _, source_dec = self._source_coord

        dec_fac = np.deg2rad(self._band_width)
        
        selected = np.where((decs >= source_dec - dec_fac) & (decs <= source_dec + dec_fac) )[0]

        self._selected = selected
        
        self._selected_event_coords = [(ec[0], ec[1]) for ec in self._event_coords
                                       if (ec[1] >= source_dec - dec_fac) & (ec[1] <= source_dec + dec_fac)]
        
        self.N = len(selected)
  

    def __call__(self, ns):

        log_likelihood = 0.0
        
        for i in range(self.N):

            signal = (ns / self.N) * self._signal_likelihood(self._selected_event_coords[i],
                                                             self._source_coord)

            bg = (1 - (ns / self.N)) * self._background_likelihood()

            log_likelihood += np.log(signal + bg)

        return -log_likelihood



class SimpleWithEnergyPointSourceLikelihood():

    
    def __init__(self, direction_likelihood, energy_likelihood, event_coords, source_coord):
        """
        Simple version of point source likelihood.
        """

        self._direction_likelihood = direction_likelihood

        self._energy_likelihood = energy_likelihood
        
        self._band_width = 3 * direction_likelihood._sigma
        
        self._event_coords = event_coords

        self._source_coord = source_coord

        self._select_declination_band()

        self.Ntot = len(self._event_coords)

        self._bg_index = 3.7
        

    def _signal_likelihood(self, event_coord, source_coord, energy, index):

        return self._direction_likelihood(event_coord, source_coord) * self._energy_likelihood(energy, index)

    
    def _background_likelihood(self, energy):

        return 1 / (np.deg2rad(self._band_width*2) * 2*np.pi) * self._energy_likelihood(energy, self._bg_index) 


    def _select_declination_band(self):

        decs = np.array([_[1] for _ in self._event_coords])

        _, source_dec = self._source_coord

        dec_fac = np.deg2rad(self._band_width)
        
        selected = np.where((decs >= source_dec - dec_fac) & (decs <= source_dec + dec_fac) )[0]

        self._selected = selected
        
        self._selected_event_coords = [(ec[0], ec[1]) for ec in self._event_coords
                                       if (ec[1] >= source_dec - dec_fac) & (ec[1] <= source_dec + dec_fac)]
        
        self.N = len(selected)
  

    def __call__(self, ns):

        log_likelihood = 0.0
        
        for i in range(self.N):

            signal = (ns / self.N) * self._signal_likelihood(self._selected_event_coords[i],
                                                             self._source_coord)

            bg = (1 - (ns / self.N)) * self._background_likelihood()

            log_likelihood += np.log(signal + bg)

        return -log_likelihood
    

        
def reweight_spectrum(energies, sim_index, new_index, bins=int(1e3)):
    """
    Use energies from a simulation with a harder 
    spectral index for efficiency.

    The spectrum is then resampled from the 
    weighted histogram

    :param energies: Energies to be reiweghted.
    :sim_index: Spectral index of the simulation. 
    :new_index: Spectral index to reweight to.
    """

    weights = np.array([np.power(_, sim_index-new_index) for _ in energies])
    
    hist, bins = np.histogram(np.log10(energies), bins=bins, 
                              weights=weights, density=True)

    bin_midpoints = bins[:-1] + np.diff(bins)/2

    cdf = np.cumsum(hist)
    cdf = cdf / float(cdf[-1])

    values = np.random.rand(len(energies))

    value_bins = np.searchsorted(cdf, values)
    
    random_from_cdf = bin_midpoints[value_bins]

    energies = np.power(10, random_from_cdf)

    return energies
