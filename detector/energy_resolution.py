import numpy as np
from scipy.stats import lognorm
from scipy.optimize import curve_fit
from abc import ABC, abstractmethod

from effective_area import R2015AeffReader

"""
Module for handling the energy resolution 
of IceCube using publicly available information.
"""

class EnergyResolution(ABC):
    """
    Abstract base class for energy resolution.
    Stores information on how the reconstructed 
    energy in the detector relates to the true 
    neutrino energy.
    """

    
    @property
    def values(self):
        """
        A 2D histogram of probabilities normalised 
        over reconstructed energy. 

        x-axis <=> true_energy
        y-axis <=> reco_energy
        """

        return self._values

    
    @values.setter
    def values(self, values):

        if len(np.shape(values)) >  2:

            raise ValueError(str(values) + ' is not a 2D array.')

        else:

            self._values = values

            
    @property
    def true_energy_bins(self):

        return self._true_energy_bins

    
    @true_energy_bins.setter
    def true_energy_bins(self, value):

        self._true_energy_bins = value


    @property
    def reco_energy_bins(self):

        return self._reco_energy_bins

    
    @reco_energy_bins.setter
    def reco_energy_bins(self, value):

        self._reco_energy_bins = value


    @abstractmethod
    def sample(self):

        pass
        
        
class NuMuEnergyResolution(EnergyResolution):
    """
    Muon neutrino energy resolution using public data.
    Makes use of the 2015 effective area release and its
    corresponding reader class.
    """

    def __init__(self, filename, **kwargs):
        """
        Muon neutrino energy resolution using public data.
        Makes use of the 2015 effective area release and its
        corresponding reader class.
        
        :param filename: Name of file to be read in.
        :param kwargs: year and/or nu_type can be specified.
        
        See release for more info.
        Link: https://icecube.wisc.edu/science/data/HE_NuMu_diffuse.
        """

        super().__init__()
        
        self._reader = R2015AeffReader(filename, **kwargs)

        self.true_energy_bins = self._reader.true_energy_bins
        
        self.reco_energy_bins = self._reader.reco_energy_bins

        self.values = self._integrate_out_cos_zenith()
        self.values = self._get_conditional()
        self.values = self._normalise_over_reco()

        self._fit_lognormal()
        self._fit_polynomial()
        
        
    def _integrate_out_cos_zenith(self):
        """
        We are only interested in the energy
        dependence.
        """

        dim_to_int = self._reader._label_order['cos_zenith']
        
        return np.sum(self._reader.effective_area_values, axis=dim_to_int)

        
    def _get_conditional(self):
        """
        From the joint distribution of Etrue and Ereco
        we want the conditional of Ereco | Etrue.
        """

        true_energy_dist = self.values.T.sum(axis=0)

        conditional = np.nan_to_num(self.values.T / true_energy_dist)

        return conditional.T

    
    def _normalise_over_reco(self):
        """
        Normalise over the reconstruted energy so
        at each Etrue bin the is a probability 
        distribution over Ereco.
        """

        normalised = np.zeros( (len(self.true_energy_bins[:-1]),
                                len(self.reco_energy_bins[:-1])) )
        
        for i, Etrue in enumerate(self.true_energy_bins[:-1]):

            norm = 0

            for j, Ereco in enumerate(self.reco_energy_bins[:-1]):

                delta_Ereco = self.reco_energy_bins[j+1] - Ereco

                norm += self.values[i][j] * delta_Ereco

            normalised[i] = self.values[i] / norm

        return normalised

            
    def _fit_lognormal(self):
        """
        Fit a lognormal distribution for each Etrue 
        and store its parameters. 
        """

        def _lognorm_wrapper(Ereco, mu, sigma):

            return lognorm.pdf(Ereco, sigma, loc=0, scale=mu)

        self.reco_energy_bin_cen = (self.reco_energy_bins[:-1] + self.reco_energy_bins[1:]) / 2

        self._mu = []
        self._sigma = []
        
        for i, Etrue in enumerate(self.true_energy_bins[:-1]):

            try:

                fit_vals, _ = curve_fit(_lognorm_wrapper, self.reco_energy_bin_cen,
                                        np.nan_to_num(self.values[i]), p0=(Etrue, 0.5))

                self._mu.append(fit_vals[0])
                self._sigma.append(fit_vals[1])
                
            except:

                self._mu.append(np.nan)
                self._sigma.append(np.nan)

                
    def _fit_polynomial(self):
        """
        Fit a polynomial to approximate the lognormal
        params at extreme energies where there are 
        little statistics.
        """

        # hard coded values for excluding low statistics
        imin = 5
        imax = 210

        # polynomial degree
        degree = 5

        true_energy_bin_cen = (self.true_energy_bins[:-1] + self.true_energy_bins[1:]) / 2

        mu_sel = np.where(np.isfinite(self._mu))

        Etrue_cen_mu = true_energy_bin_cen[mu_sel]
        mu = np.array(self._mu)[mu_sel]

        sigma_sel = np.where(np.isfinite(self._sigma))

        Etrue_cen_sigma = true_energy_bin_cen[sigma_sel]
        sigma = np.array(self._sigma)[sigma_sel]

        mu_pars = np.polyfit(np.log10(Etrue_cen_mu[imin:imax]), np.log10(mu[imin:imax]), degree)

        sigma_pars = np.polyfit(np.log10(Etrue_cen_sigma[imin:imax]), np.log10(sigma[imin:imax]), degree)
        
        self._mu_poly = np.poly1d(mu_pars)

        self._sigma_poly = np.poly1d(sigma_pars)

        
    def _get_lognormal_params(self, Etrue):
        """
        Returns params for lognormal representing 
        P(Ereco | Etrue).

        :param Etrue: The true neutrino energy [GeV]
        """

        mu = np.power( 10, self._mu_poly(np.log10(Etrue)) )

        sigma = np.power( 10, self._sigma_poly(np.log10(Etrue)) )

        return mu, sigma
        
        
    def sample(self, Etrue):
        """
        Sample a reconstructed energy given a true energy.
        """

        mu, sigma = self._get_lognormal_params(Etrue)

        return lognorm.rvs(sigma, loc=0, scale=mu)
