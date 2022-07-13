import numpy as np
from scipy.stats import lognorm
from scipy import stats
from scipy.optimize import curve_fit
from abc import ABC, abstractmethod

from icecube_tools.detector.effective_area import (
    R2015AeffReader,
    R2015_AEFF_FILENAME,
    #R2021AeffReader,
    #R2021_AEFF_FILENAME
)
from icecube_tools.detector.angular_resolution import (
    R2021AngResReader,
    R2021_ANG_RES_FILENAME,
)
from icecube_tools.utils.data import IceCubeData, find_files, data_directory

"""
Module for handling the energy resolution
of IceCube using publicly available information.
"""

GIVEN_ETRUE = 0
GIVEN_ERECO = 1

_supported_dataset_ids = ["20150820", "20210126"]


class EnergyResolutionBase(ABC):
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

        if len(np.shape(values)) > 2:

            raise ValueError(str(values) + " is not a 2D array.")

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


class EnergyResolution(EnergyResolutionBase):
    """
    Muon neutrino energy resolution using public data.
    Makes use of the 2015 effective area release and its
    corresponding reader class.

    Based on implementation by C. Haack (@chrhck).
    """

    supported_datasets = _supported_dataset_ids

    def __init__(self, filename, conditional=GIVEN_ETRUE, **kwargs):
        """
        Muon neutrino energy resolution using public data.
        Makes use of the 2015 effective area release and its
        corresponding reader class.

        Based on implementation by C. Haack (@chrhck).

        :param filename: Name of file to be read in.
        :param kwargs: year and/or nu_type can be specified.

        See release for more info.
        Link: https://icecube.wisc.edu/science/data/HE_NuMu_diffuse.
        """

        super().__init__()

        self._conditional = conditional

        self._reader = R2015AeffReader(filename, **kwargs)

        self.true_energy_bins = self._reader.true_energy_bins

        self.reco_energy_bins = self._reader.reco_energy_bins

        self.values = self._integrate_out_cos_zenith()
        self.values = self._get_conditional()
        self.values = self._normalise()

        self._fit_lognormal()
        self._fit_polynomial()

    @classmethod
    def from_dataset(cls, dataset_id, fetch=True, **kwargs):
        """
        Load energy resolution from publicly
        available data.

        :param dataset_id: Date identifying the dataset
        e.g. "20181018"
        :param fetch: If true, download dataset if missing
        """

        if dataset_id not in _supported_dataset_ids:

            raise NotImplementedError("This dataset is not currently supported")

        if fetch:

            data_interface = IceCubeData()

            dataset = data_interface.find(dataset_id)

            data_interface.fetch(dataset)

            dataset_dir = data_interface.get_path_to(dataset[0])

        else:

            dataset_dir = data_directory

        if dataset_id == "20150820":

            files = find_files(dataset_dir, R2015_AEFF_FILENAME)

            eres_file_name = files[0]

            return cls(eres_file_name, **kwargs)

        elif dataset_id == "20210126":

            files = find_files(dataset_dir, R2021_ANG_RES_FILENAME)
            eres_file_name = files[0]

            return R2021EnergyResolution(eres_file_name, **kwargs)

    def _integrate_out_cos_zenith(self):
        """
        We are only interested in the energy
        dependence.
        """

        dim_to_int = self._reader._label_order["cos_zenith"]

        return np.sum(self._reader.effective_area_values, axis=dim_to_int)

    def _get_conditional(self):
        """
        From the joint distribution of Etrue and Ereco
        we want the conditional of Ereco | Etrue OR Etrue | Ereco.
        """

        if self._conditional == GIVEN_ETRUE:

            true_energy_dist = self.values.T.sum(axis=0)

            # To avoid zero division
            true_energy_dist[true_energy_dist == 0] = 1e-10

            conditional = np.nan_to_num(self.values.T / true_energy_dist).T

        elif self._conditional == GIVEN_ERECO:

            reco_energy_dist = self.values.sum(axis=0)

            conditional = np.nan_to_num(self.values / reco_energy_dist)

        else:

            raise ValueError("conditional must be GIVEN_ETRUE or GIVEN_ERECO")

        return conditional

    def _normalise(self):
        """
        Normalise over the reconstruted energy so
        at each Etrue bin the is a probability
        distribution over Ereco.
        """

        if self._conditional == GIVEN_ETRUE:

            normalised = np.zeros(
                (len(self.true_energy_bins[:-1]), len(self.reco_energy_bins[:-1]))
            )

            for i, Etrue in enumerate(self.true_energy_bins[:-1]):

                norm = 0

                for j, Ereco in enumerate(self.reco_energy_bins[:-1]):

                    delta_Ereco = self.reco_energy_bins[j + 1] - Ereco

                    norm += self.values[i][j] * delta_Ereco

                # Avoid zero division
                if norm == 0:
                    norm = 1e-10

                normalised[i] = self.values[i] / norm

        elif self._conditional == GIVEN_ERECO:

            normalised = np.zeros(
                (len(self.true_energy_bins[:-1]), len(self.reco_energy_bins[:-1]))
            ).T

            for i, Ereco in enumerate(self.reco_energy_bins[:-1]):

                norm = 0

                for j, Etrue in enumerate(self.true_energy_bins[:-1]):

                    delta_Etrue = self.true_energy_bins[j + 1] - Etrue

                    norm += self.values.T[i][j] * delta_Etrue

                normalised[i] = self.values.T[i] / norm

            normalised = normalised.T

        return normalised

    def _fit_lognormal(self):
        """
        Fit a lognormal distribution for each Etrue
        and store its parameters.
        """

        def _lognorm_wrapper(E, mu, sigma):

            return lognorm.pdf(E, sigma, loc=0, scale=mu)

        self._mu = []
        self._sigma = []

        if self._conditional == GIVEN_ETRUE:

            self.reco_energy_bin_cen = (
                self.reco_energy_bins[:-1] + self.reco_energy_bins[1:]
            ) / 2

            for i, Etrue in enumerate(self.true_energy_bins[:-1]):

                try:

                    fit_vals, _ = curve_fit(
                        _lognorm_wrapper,
                        self.reco_energy_bin_cen,
                        np.nan_to_num(self.values[i]),
                        p0=(Etrue, 0.5),
                    )

                    self._mu.append(fit_vals[0])
                    self._sigma.append(fit_vals[1])

                except:

                    self._mu.append(np.nan)
                    self._sigma.append(np.nan)

        elif self._conditional == GIVEN_ERECO:

            self.true_energy_bin_cen = (
                self.true_energy_bins[:-1] + self.true_energy_bins[1:]
            ) / 2

            for i, Ereco in enumerate(self.reco_energy_bins[:-1]):

                try:

                    fit_vals, _ = curve_fit(
                        _lognorm_wrapper,
                        self.true_energy_bin_cen,
                        np.nan_to_num(self.values.T[i]),
                        p0=(Ereco, 0.5),
                    )

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

        # polynomial degree
        degree = 5

        mu_sel = np.where(np.isfinite(self._mu))
        mu = np.array(self._mu)[mu_sel]

        sigma_sel = np.where(np.isfinite(self._sigma))
        sigma = np.array(self._sigma)[sigma_sel]

        if self._conditional == GIVEN_ETRUE:

            # hard coded values for excluding low statistics
            imin = 5
            imax = 210

            true_energy_bin_cen = (
                self.true_energy_bins[:-1] + self.true_energy_bins[1:]
            ) / 2

            Etrue_cen_mu = true_energy_bin_cen[mu_sel]

            Etrue_cen_sigma = true_energy_bin_cen[sigma_sel]

            mu_pars = np.polyfit(
                np.log10(Etrue_cen_mu[imin:imax]), np.log10(mu[imin:imax]), degree
            )

            sigma_pars = np.polyfit(
                np.log10(Etrue_cen_sigma[imin:imax]), np.log10(sigma[imin:imax]), degree
            )

        elif self._conditional == GIVEN_ERECO:

            # hard coded values for exluding low statistics
            imin = 5
            imax = 45

            reco_energy_bin_cen = (
                self.reco_energy_bins[:-1] + self.reco_energy_bins[1:]
            ) / 2

            Ereco_cen_mu = reco_energy_bin_cen[mu_sel]

            Ereco_cen_sigma = reco_energy_bin_cen[sigma_sel]

            mu_pars = np.polyfit(
                np.log10(Ereco_cen_mu[imin:imax]), np.log10(mu[imin:imax]), degree
            )

            sigma_pars = np.polyfit(
                np.log10(Ereco_cen_sigma[imin:imax]), np.log10(sigma[imin:imax]), degree
            )

        self._mu_poly = np.poly1d(mu_pars)

        self._sigma_poly = np.poly1d(sigma_pars)

    def _get_lognormal_params(self, E):
        """
        Returns params for lognormal representing
        P(Ereco | Etrue) OR P(Etrue | Ereco).

        :param E: The true/reco energy if GIVEN_ETRUE/GIVEN_ERECO [GeV]
        """

        mu = np.power(10, self._mu_poly(np.log10(E)))

        sigma = np.power(10, self._sigma_poly(np.log10(E)))

        return mu, sigma

    def sample(self, E):
        """
        Sample a reco/true energy given a true/reco energy.
        """

        mu, sigma = self._get_lognormal_params(E)

        return lognorm.rvs(sigma, loc=0, scale=mu)


class R2021EnergyResolution(EnergyResolutionBase):
    """
    Class to handle energy resolution of 2021 data release.
    """

    def __init__(self, filename, conditional=GIVEN_ETRUE, **kwargs):
        """
        Energy resolution class for handling 10 year data released 2021.
        :param filename: Name of the file to be read in
        """

        self._filename = filename

        self._reader = R2021AngResReader(filename, **kwargs)

        self._true_energy_bins = self._reader.true_energy_bins

        self.declination_bins = self._reader.declination_bins

        self.dataset = self._reader.output
        #store marginal pdfs of reco energy for each true energy bin in a dict
        self.reco_energy_pdfs = {e: {} for e in range(self._true_energy_bins.shape[0])}

        self._values = []

        super().__init__()


    def _marginalisation(self, c_e, c_d, qoi="ERec"):
        """
        Function that marginalises over the smearing data provided for the 2021 release.
        Careful: Samples are drawn in logspace and converted to linspace upon return.
        :param int c_e: Index of energy bin
        :param int c_d: Index of declination bin
        :return: n, bins of the created distribution/histogram
        """
        
        if qoi == "ERec":
            needed_index = 4
        else:
            raise ValueError("Not other quantity of interest is available.")
        
        #do pre-selection: lowest energy and highest declination, save into new array
        reduced_data = self.dataset[np.intersect1d(np.argwhere(
            np.isclose(self.dataset[:, 0], self.true_energy_bins[c_e])),
                                np.argwhere(
            np.isclose(self.dataset[:, 2], np.rad2deg(self.declination_bins[c_d]))))]
        
        bins = np.array(sorted(list(set(reduced_data[:, needed_index]).union(
                    set(reduced_data[:, needed_index+1])))))
        
        frac_counts = np.zeros(bins.shape[0]-1)
       
        #marginalise over uninteresting quantities
        for c_b, b in enumerate(bins[:-1]):
            indices = np.nonzero(np.isclose(b, reduced_data[:, needed_index]))
            frac_counts[c_b] = np.sum(reduced_data[indices, -1])
        return frac_counts, bins


    def _return_bins(self, energy, declination):
        """
        Returns the lower bin edges and their indices for given energy and declination.
        :param float energy: Energy in GeV
        :param float declination: Declination in rad
        :return: Index of energy, energy at lower bin edge, index of declination, declination at lower bin edge
        :raises ValueError: if energy is outside of IRF-file range
        :raises ValueError: if declination is outside of $[-\pi/2, \pi/2]$
        """

        if energy >= self.true_energy_bins[0] and energy <= self.true_energy_bins[-1]:
            c_e = np.digitize(energy, self.true_energy_bins)
            #Need to get the index of lower bin edge.
            #np.digitize returns one too much, two if energy=highest bin edge
            if c_e < self.true_energy_bins.shape[0]:
                c_e -= 1
            else:
                c_e -= 2
            e = self.true_energy_bins[c_e]
        else:
            raise ValueError("Energy out of bounds.")

        if declination >= self.declination_bins[0] and declination <= self.declination_bins[-1]:
            c_d = np.digitize(declination, self.declination_bins)
            #Same procedure
            if c_d < self.declination_bins.shape[0]:
                c_d -= 1
            else:
                c_d -= 2
            d = self.declination_bins[c_d]
        else:
            raise ValueError("Declination out of bounds.")

        return c_e, e, c_d, d


    def sample(self, energy, declination):
        """
        Sample reconstructed energy given true energy and declination.
        :param energy: True $\log_{10}(E/\mathrm{GeV})$ to be sampled
        :param declination: True declination in radians to be sampled
        :return: Energy in GeV
        """

        c_e, e, c_d, d = self._return_bins(energy, declination)

        try:
            Erec = self.reco_energy_pdfs[c_e][c_d].rvs(size=1)[0]
        except KeyError:
            n, bins = self._marginalisation(c_e, c_d)
            self.reco_energy_pdfs[c_e][c_d] = stats.rv_histogram((n, bins))
            Erec = self.reco_energy_pdfs[c_e][c_d].rvs(size=1)[0]   # draws log(angle) values
        return np.power(10, Erec)

