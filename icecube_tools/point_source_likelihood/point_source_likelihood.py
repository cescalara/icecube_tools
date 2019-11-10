import numpy as np
from iminuit import Minuit

from .energy_likelihood import *
from .spatial_likelihood import *

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

        self._band_solid_angle = ((2 * self._band_width) / 180) * 4 * np.pi
        
        self._event_coords = event_coords
        
        self._energies = energies

        self._source_coord = source_coord

        self._bg_index = 3.8

        self._ns_min = 0.0
        self._ns_max = 100
        self._max_index = 3.8

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

        return self._energy_likelihood(energy, self._bg_index) / self._band_solid_angle
 
        
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

        Uses the iMiuint wrapper.
        """


        init_index = 2.0 #self._energy_likelihood._min_index + (self._max_index - self._energy_likelihood._min_index)/2 
        init_ns = self._ns_min + (self._ns_max - self._ns_min)/2 

        m = Minuit(self._get_neg_log_likelihood_ratio, ns=init_ns, index=init_index,
                   error_ns=0.1, error_index=1, errordef=0.5,
                   limit_ns=(self._ns_min, self._ns_max),
                   limit_index=(self._energy_likelihood._min_index, self._max_index))

        m.migrad()
        
        if not m.migrad_ok() or not m.matrix_accurate():
        
            # Fix the index as can be uninformative
            m.fixed['index'] = True
            m.migrad()
       
        self._best_fit_ns = m.values['ns']
        self._best_fit_index = m.values['index']

        
    def _minimize_grid(self):
        """
        Minimize -log(likelihood_ratio) for the source hypothesis, 
        returning the best fit ns and index.
        
        This simple grid method takes roughly the same time as minuit 
        and is more accurate... 
        """

        ns_grid = np.linspace(self._ns_min, self._ns_max, 10)
        index_grid = np.linspace(self._energy_likelihood._min_index, self._max_index, 10)

        out = np.zeros((len(ns_grid), len(index_grid)))
        for i, ns in enumerate(ns_grid):
            for j, index in enumerate(index_grid):
                out[i][j] = self._get_neg_log_likelihood_ratio(ns, index)

        sel = np.where(out==np.min(out))

        if len(sel[0]) > 1:

            self._best_fit_index = 3.7
            self._best_fit_ns = 0.0

        else:

            self._best_fit_ns = ns_grid[sel[0]][0]
            self._best_fit_index = index_grid[sel[1]][0]
            
    
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
        #self._minimize_grid()
        
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
    


