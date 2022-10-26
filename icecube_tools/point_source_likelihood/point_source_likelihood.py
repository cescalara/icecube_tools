import numpy as np
from iminuit import Minuit
import logging

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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PointSourceLikelihood:
    """
    Calculate the point source likelihood for a given
    neutrino dataset - in terms of reconstructed
    energies and arrival directions.
    Based on what is described in Braun+2008 and
    Aartsen+2018.
    """

    def __init__(
        self,
        direction_likelihood,
        energy_likelihood,
        ras,
        decs,
        energies,
        source_coord,
        ang_errs=[],
        which='both',
        bg_energy_likelihood=None,
        index_prior=None,
        band_width_factor=3.0,
    ):
        """
        Calculate the point source likelihood for a given
        neutrino dataset - in terms of reconstructed
        energies and arrival directions.

        :param direction_likelihood: An instance of SpatialLikelihood.
        :param energy_likelihood: An instance of MarginalisedEnergyLikelihood.
        :param event_coords: List of (ra, dec) tuples for reconstructed coords.
        :param energies: The reconstructed nu energies.
        :param source_coord: (ra, dec) pf the point to test.
        :param index_prior: Optional prior on the spectral index, instance of Prior.
        """

        if which not in ["both", "energy", "spatial"]:
            raise ValueError("No other type of likelihood available.")
        else:
            self.which = which
            logger.info(f"Using {which} likelihoods.")

        self._direction_likelihood = direction_likelihood

        self._energy_likelihood = energy_likelihood

        self._bg_energy_likelihood = bg_energy_likelihood

        """
        try:
            if 3.7 in energy_likelihood.index_list:
                self._bg_energy_likelihood = energy_likelihood.likelihood['3.7']
        except AttributeError:
            pass
        """
        
        if isinstance(
            self._direction_likelihood, EnergyDependentSpatialGaussianLikelihood
        ):

            self._band_width = (
                band_width_factor * self._direction_likelihood.get_low_res()
            )

        else:

            self._band_width = (
                band_width_factor * self._direction_likelihood._sigma
            )  # degrees

        self._dec_low = source_coord[1] - np.deg2rad(self._band_width)

        self._dec_high = source_coord[1] + np.deg2rad(self._band_width)

        if self._dec_low < np.arcsin(-1.0) or np.isnan(self._dec_low):
            self._dec_low = np.arcsin(-1.0)

        if self._dec_high > np.arcsin(1.0) or np.isnan(self._dec_high):
            self._dec_high = np.arcsin(1.0)

        self._band_solid_angle = (
            2 * np.pi * (np.sin(self._dec_high) - np.sin(self._dec_low))
        )

        self._ra_low = source_coord[0] - np.deg2rad(self._band_width)

        self._ra_high = source_coord[0] + np.deg2rad(self._band_width)

        self._ras = ras

        self._decs = decs

        self._energies = energies

        self._source_coord = source_coord

        self._index_prior = index_prior

        self._ang_errs = ang_errs

        # Sensible values based on Braun+2008
        # and Aartsen+2018 analyses
        self._bg_index = 3.7
        self._ns_min = 0.0
        
        try:
            self._max_index = self._energy_likelihood._max_index
            self._min_index = self._energy_likelihood._min_index
        except AttributeError:
            self._max_index = 3.95

        self._select_nearby_events()

        # Can't have more source events than actual events...
        self._ns_max = self.N

        self.Ntot = len(self._energies)


    def _select_nearby_events(self):
        source_ra, source_dec = self._source_coord
        dec_fac = np.deg2rad(self._band_width)
        selected = list(
            set(
                np.where(
                    (self._decs >= self._dec_low)
                    & (self._decs <= self._dec_high)
                    & (self._ras >= self._ra_low)
                    & (self._ras <= self._ra_high)
                )[0]
            )
        )
        selected_dec_band = np.where(
            (self._decs >= self._dec_low) & (self._decs <= self._dec_high)
        )[0]

        self._selected = selected

        self._selected_ras = self._ras[selected]

        self._selected_decs = self._decs[selected]

        self._selected_energies = self._energies[selected]

        if isinstance(self._ang_errs, np.ndarray):
            self._selected_ang_errs = self._ang_errs[selected]
        else:
            self._selected_ang_errs = [1] * len(selected)

        self.Nprime = len(selected)

        self.N = len(selected_dec_band)


    def _signal_likelihood(self, ra, dec, source_coord, energy, index, ang_err=1):
        """
        Calculate the signal likelihood of a given event.
        :param ra: RA of event
        :param dec: DEC of event
        :param source_coord: Tuple of source coordinate (ra, dec)
        :param energy: Energy of event in GeV
        :param index: Spectral index of source model
        :param ang_err: Angular error on the event, defaults to 1 degree.
        """

        

        if isinstance(
            self._direction_likelihood, EnergyDependentSpatialGaussianLikelihood
        ):
            def spatial():
                return self._direction_likelihood(
                    (ra, dec), source_coord, energy, index
                )

            def en():
                return self._energy_likelihood(energy, index, dec)


        elif isinstance(
            self._direction_likelihood, EventDependentSpatialGaussianLikelihood
        ):
            def spatial():
                return self._direction_likelihood(
                    ang_err, (ra, dec), source_coord
                )
            
            def en():
                return self._energy_likelihood(energy, index, dec)

        else:
            def spatial():
                return self._direction_likelihood(
                    (ra, dec), source_coord
                )
            def en():
                return self._energy_likelihood(energy, index, dec)
        
        if self.which == 'spatial':
            output = spatial()
        elif self.which == 'energy':
            output = en()
        else:
            output = en() * spatial()

        return output


    def _background_likelihood(self, energy, dec):
        """
        Calculate the background likelihood for an event of given energy.
        :param energy: Energy of event in GeV
        """

        if self._bg_energy_likelihood is not None:
            def en():
                return self._bg_energy_likelihood(energy) 
            
        else:
            def en():
                return self._energy_likelihood(energy, self._bg_index, dec)
        
        def spatial():
            return 1. / self._band_solid_angle

        #Check which part is used for likelihood calculation
        if self.which == 'spatial':
            output = spatial()
        elif self.which == 'energy':
            output = en()
        else:
            output = en() * spatial()

        if output == 0.0:
            output = 1e-10

        return output


    def _func_to_minimize(self, ns, index):
        """
        Calculate the -log(likelihood_ratio) for minimization.

        Uses calculation described in:
        https://github.com/IceCubeOpenSource/SkyLLH/blob/master/doc/user_manual.pdf

        If there is a prior, it is added here, as this is equivalent to maximising
        the likelihood.

        :param ns: Number of source counts.
        :param index: Spectral index of the source.
        """

        one_plus_alpha = 1e-10
        alpha = one_plus_alpha - 1
        if isinstance(self._energy_likelihood, MarginalisedIntegratedEnergyLikelihood):
            index_list = [index]
        else:
            idx = np.digitize(index, self._energy_likelihood.index_list)
            llhs = np.zeros(2)
            index_list = self._energy_likelihood.index_list[idx-1:idx+1]

        for c, indx in enumerate(index_list):
            log_likelihood_ratio = 0.0
            for i in range(self.Nprime):
                signal = self._signal_likelihood(
                    self._selected_ras[i],
                    self._selected_decs[i],
                    self._source_coord,
                    self._selected_energies[i],
                    indx,
                    ang_err=self._selected_ang_errs[i]
                )

                bg = self._background_likelihood(
                    self._selected_energies[i],
                    self._selected_decs[i]
                )

                chi = (1 / self.N) * (signal / bg - 1)

                alpha_i = ns * chi

                if (1 + alpha_i) < one_plus_alpha:

                    alpha_tilde = (alpha_i - alpha) / one_plus_alpha
                    log_likelihood_ratio += (
                        np.log1p(alpha) + alpha_tilde - (0.5 * alpha_tilde ** 2)
                    )

                else:

                    log_likelihood_ratio += np.log1p(alpha_i)

            log_likelihood_ratio += (self.N - self.Nprime) * np.log1p(-ns / self.N)

            if isinstance(self._energy_likelihood, MarginalisedIntegratedEnergyLikelihood):
                #exit for integrated likelihood after one iteration, since that's all that's needed
                if self._index_prior:
                    log_likelihood_ratio += np.log(self._index_prior(indx))
                return -log_likelihood_ratio
            else:
                #continue through loop, pick value in the middle at the end
                llhs[c] = log_likelihood_ratio

        log_likelihood_ratio = np.interp(index, index_list, llhs)
        if self._index_prior:
            log_likelihood_ratio += np.log(self._index_prior(indx))
        
        return -log_likelihood_ratio


    def _func_to_minimize_sp(self, ns, index=2.0):
        """
        Calculate the -log(likelihood_ratio) for minimization using energy only.

        Uses calculation described in:
        https://github.com/IceCubeOpenSource/SkyLLH/blob/master/doc/user_manual.pdf

        If there is a prior, it is added here, as this is equivalent to maximising
        the likelihood.

        :param ns: Number of source counts.
        :param index: Dummy argument
        """
        
        one_plus_alpha = 1e-10
        alpha = one_plus_alpha - 1

        log_likelihood_ratio = 0.0
        for i in range(self.Nprime):
            signal = self._signal_likelihood(
                self._selected_ras[i],
                self._selected_decs[i],
                self._source_coord,
                self._selected_energies[i],
                2.0,
                ang_err=self._selected_ang_errs[i]
            )

            bg = self._background_likelihood(self._selected_energies[i], self._selected_decs[i])

            chi = (1 / self.N) * (signal / bg - 1)

            alpha_i = ns * chi

            if (1 + alpha_i) < one_plus_alpha:

                alpha_tilde = (alpha_i - alpha) / one_plus_alpha
                log_likelihood_ratio += (
                    np.log1p(alpha) + alpha_tilde - (0.5 * alpha_tilde ** 2)
                )

            else:

                log_likelihood_ratio += np.log1p(alpha_i)

        log_likelihood_ratio += (self.N - self.Nprime) * np.log1p(-ns / self.N)

        return -log_likelihood_ratio


    def __call__(self, ns, index):
        """
        Wrapper function for convenience.
        """

        return self._func_to_minimize(ns, index)


    def _minimize(self):
        """
        Minimize -log(likelihood_ratio) for the source hypothesis,
        returning the best fit ns and index.

        Uses the iMiuint wrapper.
        """

        init_index = self._energy_likelihood._min_index + (self._max_index - self._energy_likelihood._min_index)/2
        init_ns = self._ns_min + (self._ns_max - self._ns_min) / 2

        if self.which == 'spatial':
            #Spatial-only likelihood needs special function, because no spectral index is used
            logger.info("Using only spatial information.")
            func_to_minimize = self._func_to_minimize_sp
        else:
            func_to_minimize = self._func_to_minimize
            logger.info("Using all information.")

        m = Minuit(
            func_to_minimize,
            ns=init_ns,
            index=init_index,
        )
        # m.fixed["index"] = True
        m.limits["ns"] = (self._ns_min, self._ns_max)
        m.errors["index"] = 0.1242
        m.errors["ns"] = 1
        m.limits["index"] = (self._energy_likelihood._min_index, self._energy_likelihood._max_index)
        if self.which == 'spatial':
            m.fixed["index"] = True
        m.errordef = 0.5
        m.migrad()

        if not m.valid or not m.fmin.has_accurate_covar or not m.fmin.has_covariance:

            # Fix the index as can be uninformative
            m.fixed["index"] = True
            m.migrad()

        self._best_fit_ns = m.values["ns"]
        self._best_fit_index = m.values["index"]
        return m


    def _minimize_grid(self):
        """
        Minimize -log(likelihood_ratio) for the source hypothesis,
        returning the best fit ns and index.

        This simple grid method takes roughly the same time as minuit
        and is more accurate...
        """

        ns_grid = np.linspace(self._ns_min, self._ns_max, 10)
        index_grid = np.linspace(
            self._energy_likelihood._min_index, self._max_index, 10
        )

        out = np.zeros((len(ns_grid), len(index_grid)))
        for i, ns in enumerate(ns_grid):
            for j, index in enumerate(index_grid):
                out[i][j] = self._func_to_minimize(ns, index)

        sel = np.where(out == np.min(out))

        if len(sel[0]) > 1:

            self._best_fit_index = 3.7
            self._best_fit_ns = 0.0

        else:

            self._best_fit_ns = ns_grid[sel[0]][0]
            self._best_fit_index = index_grid[sel[1]][0]
        self.grid = out


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

            signal = self._signal_likelihood(
                self._selected_ras[i],
                self._selected_decs[i],
                self._source_coord,
                self._selected_energies[i],
                index,
                ang_err=self._selected_ang_errs[i],
            )

            bg = self._background_likelihood(self._selected_energies[i])

            chi_i = (1 / self.N) * ((signal / bg) - 1)

            alpha_i = ns * chi_i

            if (1 + alpha_i) < one_plus_alpha:

                alpha_tilde = (alpha_i - alpha) / one_plus_alpha

                self._first_derivative.append(
                    (1 / one_plus_alpha) * (1 - alpha_tilde) * chi_i
                )

            else:

                self._first_derivative.append(chi_i / (1 + alpha_i))

        self._first_derivative = np.array(self._first_derivative)

        return sum(self._first_derivative) - ((self.N - self.Nprime) / (self.N - ns))


    def _second_derivative_likelihood_ratio(self, ns=0):
        """
        Second derivative of the likelihood ratio.
        Equation 44 in
        https://github.com/IceCubeOpenSource/SkyLLH/blob/master/doc/user_manual.pdf.
        """

        self._second_derivative = -((self._first_derivative) ** 2)

        return sum(self._second_derivative) - (
            (self.N - self.Nprime) / (self.N - ns) ** 2
        )


    def get_test_statistic(self):
        """
        Calculate the test statistic for the best fit ns
        """

        _ = self._minimize()
        # self._minimize_grid()

        # For resolving the TS peak at zero
        # if self._best_fit_ns == 0:

        #    first_der = self._first_derivative_likelihood_ratio(self._best_fit_ns, self._best_fit_index)
        #    second_der = self._second_derivative_likelihood_ratio(self._best_fit_ns)

        #    self.test_statistic = -2 * (first_der**2 / (4 * second_der))

        #   self.likelihood_ratio = np.exp(-self.test_statistic/2)

        # else:

        neg_log_lik = self._func_to_minimize(
            self._best_fit_ns, self._best_fit_index
        )

        self.likelihood_ratio = np.exp(neg_log_lik)

        self.test_statistic = -2 * neg_log_lik

        return self.test_statistic


class SpatialOnlyPointSourceLikelihood:
    """
    Calculate the point source likelihood for a given
    neutrino dataset - in terms of reconstructed
    arrival directions.

    This class is exactly as in PointSourceLikelihood,
    but without the energy depedence.

    Should be removed at some point, this case is already
    included in the main class with the keyword "which",
    defaulting to "both".
    """

    def __init__(self, direction_likelihood, event_coords, source_coord):
        """
        Calculate the point source likelihood for a given
        neutrino dataset - in terms of reconstructed
        energies and arrival directions.

        :param direction_likelihood: An instance of SpatialGaussianLikelihood.
        :param event_coords: List of (ra, dec) tuples for reconstructed coords.
        :param source_coord: (ra, dec) pf the point to test.
        """

        self._direction_likelihood = direction_likelihood

        self._band_width = 3 * self._direction_likelihood._sigma  # degrees

        dec_low = source_coord[1] - np.deg2rad(self._band_width)
        dec_high = source_coord[1] + np.deg2rad(self._band_width)
        self._band_solid_angle = 2 * np.pi * (np.sin(dec_high) - np.sin(dec_low))

        self._event_coords = event_coords

        self._source_coord = source_coord

        self._bg_index = 3.7

        self._ns_min = 0.0
        self._ns_max = 100
        self._max_index = 3.7

        self._select_nearby_events()

        self.Ntot = len(self._event_coords)

    def _select_nearby_events(self):

        ras = np.array([_[0] for _ in self._event_coords])

        decs = np.array([_[1] for _ in self._event_coords])

        source_ra, source_dec = self._source_coord

        dec_fac = np.deg2rad(self._band_width)

        selected = list(
            set(
                np.where(
                    (decs >= source_dec - dec_fac)
                    & (decs <= source_dec + dec_fac)
                    & (ras >= source_ra - dec_fac)
                    & (ras <= source_ra + dec_fac)
                )[0]
            )
        )

        selected_dec_band = np.where(
            (decs >= source_dec - dec_fac) & (decs <= source_dec + dec_fac)
        )[0]

        self._selected = selected

        self._selected_event_coords = [
            (ec[0], ec[1])
            for ec in self._event_coords
            if (ec[1] >= source_dec - dec_fac)
            & (ec[1] <= source_dec + dec_fac)
            & (ec[0] >= source_ra - dec_fac)
            & (ec[0] <= source_ra + dec_fac)
        ]

        self.Nprime = len(selected)

        self.N = len(selected_dec_band)

    def _signal_likelihood(self, event_coord, source_coord):

        return self._direction_likelihood(event_coord, source_coord)

    def _background_likelihood(self):

        return 1.0 / self._band_solid_angle

    def _get_neg_log_likelihood_ratio(self, ns):
        """
        Calculate the -log(likelihood_ratio) for minimization.

        Uses calculation described in:
        https://github.com/IceCubeOpenSource/SkyLLH/blob/master/doc/user_manual.pdf

        :param ns: Number of source counts.
        """

        one_plus_alpha = 1e-10
        alpha = one_plus_alpha - 1

        log_likelihood_ratio = 0.0

        for i in range(self.Nprime):

            signal = self._signal_likelihood(
                self._selected_event_coords[i], self._source_coord
            )

            bg = self._background_likelihood()

            chi = (1 / self.N) * (signal / bg - 1)

            alpha_i = ns * chi

            if (1 + alpha_i) < one_plus_alpha:

                alpha_tilde = (alpha_i - alpha) / one_plus_alpha
                log_likelihood_ratio += (
                    np.log1p(alpha) + alpha_tilde - (0.5 * alpha_tilde ** 2)
                )

            else:

                log_likelihood_ratio += np.log1p(alpha_i)

        log_likelihood_ratio += (self.N - self.Nprime) * np.log1p(-ns / self.N)

        return -log_likelihood_ratio

    def __call__(self, ns):
        """
        Wrapper function for convenience.
        """

        return self._get_neg_log_likelihood_ratio(ns)

    def _minimize(self):
        """
        Minimize -log(likelihood_ratio) for the source hypothesis,
        returning the best fit ns and index.

        Uses the iMiuint wrapper.
        """

        init_ns = self._ns_min + (self._ns_max - self._ns_min) / 2
        
        m = Minuit(
            self._func_to_minimize,
            ns=init_ns,
        )
        m.limits["ns"] = (self._ns_min, self._ns_max)
        m.errors["ns"] = 1
        m.errordef = 0.5
        m.migrad()

        self._best_fit_ns = m.values["ns"]

        return m


    def get_test_statistic(self):
        """
        Calculate the test statistic for the best fit ns
        """

        self._minimize()

        neg_log_lik = self._get_neg_log_likelihood_ratio(self._best_fit_ns)

        self.likelihood_ratio = np.exp(neg_log_lik)

        self.test_statistic = -2 * neg_log_lik

        return self.test_statistic


class TimeDependentPointSourceLikelihood:
    def __init__(
        self,
        source_coords,
        periods,
        event_files,
        energy_likelihood: MarginalisedEnergyLikelihood,
        path=None,
        index_list=None,
        which="both"):
        """
        Create likelihood covering multiple data taking periods.
        :param source_coords: Tuple of ra, dec.
        :param periods: List of str of period names, eg. `IC40`
        :param event_files: List of event files corresponding the the above periods.
        :energy_likelihood: Class inheriting from MarginalisedEnergyLikelihood.
        :param index_list: List of indices covered by the events used to build the energy likelihood.
        :param path: Path to directory where the simulated events (see above) are located.
        :param which: String, `both`, `spatial`, `energy` indicating which likelihoods are to be used.
        """

        self.which = which
        self.source_coords = source_coords
        #files should be dict of files, with period string as the key
        self.event_files = event_files
        self.periods = periods
        #self.period_dict = period_dict
        """
        self.period_dict = {str(p):
            {str(index): path to data file for index in index_list} for p in periods
        }

        """
        #self.periods = list(period_dict.keys())
        #self.index_list = 
        self.index_list = index_list
        assert len(event_files) == len(periods)

        #TODO change this to named tuples?
        self.likelihoods = {}
        # Can use one spatial llh for all periods, 'tis but a Gaussian
        spatial_llh = EventDependentSpatialGaussianLikelihood()

        for p, data in zip(self.periods, self.event_files):
            print(p)
            # Open event files
            with h5py.File(data, "r") as f:
                reco_energy = f["reco_energy"][()]
                ra = f["ra"][()]
                dec = f["dec"][()]
                ang_err = f["ang_err"][()]
            if energy_likelihood == MarginalisedEnergyLikelihood2021:
                energy_llh = MarginalisedEnergyLikelihood2021(
                    index_list, path, f"p_{p}", self.source_coords[1]
                )
            elif energy_likelihood == MarginalisedIntegratedEnergyLikelihood:
                energy_llh = MarginalisedIntegratedEnergyLikelihood(
                    R2021IRF.from_period(p),
                    EffectiveArea.from_dataset("20210126", p),
                    np.linspace(2, 9, num=20)
                )

            #create likelihood objects
            self.likelihoods[p] = PointSourceLikelihood(
                spatial_llh, energy_llh, ra, dec, reco_energy,
                self.source_coords, ang_err, which=self.which
            )


    def __call__(self, *ns, index):
        """
        Calculate negative log-like ratio as function of ns and index.
        ns is now vector with an entry for each period.
        :param ns: List of numbers of source events.
        :param index: Spectral index of source spectrum.
        """
        assert len(ns) == len(self.likelihoods.keys())
        return self._func_to_minimize(np.hstack((ns, index)))


    def _func_to_minimize(self, *arg):
        """
        According to https://github.com/icecube/skyllh/blob/master/doc/user_manual.pdf,
        Eq. (59), the returned values of each period's llh._func_to_minimize() can be added.
        :param arg: numpy.ndarray, last entry is index, all before are number of source events.
        """
        neg_log_like = 0
        for (n, llh) in zip(arg[:-1], self.likelihoods.values()):
            val = llh(n, arg[-1])
            #print(val)
            neg_log_like += val
        return neg_log_like


    def _func_to_minimize_sp(self, *arg):
        """
        According to https://github.com/icecube/skyllh/blob/master/doc/user_manual.pdf,
        Eq. (59), the returned values of each period's llh._func_to_minimize() can be added.
        :param arg: numpy.ndarray, last entry is index, all before are number of source events.
        """
        neg_log_like = 0
        for (n, llh) in zip(arg, self.likelihoods.values()):
            val = llh._func_to_minimize_sp(n)
            #print(val)
            neg_log_like += val
        return neg_log_like


    def _minimize(self):
        """
        Minimize -log(likelihood_ratio) for the source hypothesis,
        returning the best fit ns and index.

        Uses the iMinuint wrapper.
        """

        error_index = 0.1
        some_llh = self.likelihoods[list(self.likelihoods.keys())[0]]
        init_index = some_llh._min_index + (some_llh._max_index - some_llh._min_index) / 2
        limit_index = (some_llh._energy_likelihood._min_index,
            some_llh._energy_likelihood._max_index)
        # Get init_ns and limit_ns for each period
        # could be nicer with some generator method
        init = []
        limits = []
        for llh in self.likelihoods.values():
            init.append(llh._ns_min + (llh._ns_max - llh._ns_min) / 2)
            limits.append((llh._ns_min, llh._ns_max))

        # Get errors to start with
        errors = [1 for _ in init]  
        name = tuple(f"n{i}" for i in range(len(init)))  

        if self.which == 'spatial':
            #Only spatial-only likelihood needs special function, because no spectral index is used
            func_to_minimize = self._func_to_minimize_sp
        else:
            func_to_minimize = self._func_to_minimize
            #add errors, limits, start value and name of index
            errors += [error_index]
            limits += [limit_index]
            init += [init_index]
            name += ("index",)

        m = Minuit(func_to_minimize, *init, name=name)
        m.errordef = 0.5
        m.errors = errors
        m.limits = limits        
        m.migrad()

        if self.which != 'spatial':
            if not m.fmin.is_valid or not m.fmin.has_covariance:

                # Fix the index as can be uninformative
                m.fixed["index"] = True
                m.migrad()
        else:
            if not m.fmin.is_valid or not m.fmin.has_covariance:
                logger.warning("Fit has not converged, proceed with caution.")

        return m


    def get_test_statistic(self):
        raise NotImplementedError("Not yet implemented.")



class EnergyDependentSpatialPointSourceLikelihood:
    """
    Calculate the point source likelihood for a given
    neutrino dataset - in terms of reconstructed
    arrival directions.

    This class is exactly as in PointSourceLikelihood,
    but without the energy depedence.
    """

    def __init__(
        self,
        direction_likelihood,
        ras,
        decs,
        energies,
        source_coord,
        band_width_factor=3.0,
    ):
        """
        Calculate the point source likelihood for a given
        neutrino dataset - in terms of reconstructed
        energies and arrival directions.

        :param direction_likelihood: An instance of SpatialGaussianLikelihood.
        :param ras: Array of right acsensions in [rad]
        :param decs: Array of declinations in [dec]
        :param source_coord: (ra, dec) pf the point to test.
        """

        self._direction_likelihood = direction_likelihood

        self._band_width = band_width_factor * self._direction_likelihood.get_low_res()

        self._dec_low = source_coord[1] - np.deg2rad(self._band_width)

        self._dec_high = source_coord[1] + np.deg2rad(self._band_width)

        if self._dec_low < np.arcsin(-0.1) or np.isnan(self._dec_low):
            self._dec_low = np.arcsin(-0.1)

        if self._dec_high > np.arcsin(1.0) or np.isnan(self._dec_high):
            self._dec_high = np.arcsin(1.0)

        self._band_solid_angle = (
            2 * np.pi * (np.sin(self._dec_high) - np.sin(self._dec_low))
        )

        self._ra_low = source_coord[0] - np.deg2rad(self._band_width)

        self._ra_high = source_coord[0] + np.deg2rad(self._band_width)

        self._ras = ras

        self._decs = decs

        self._source_coord = source_coord

        self._energies = energies

        self._ns_min = 0.0
        self._ns_max = 100

        self._select_nearby_events()

        self.Ntot = len(self._ras)

    def _select_nearby_events(self):

        source_ra, source_dec = self._source_coord

        selected = list(
            set(
                np.where(
                    (self._decs >= self._dec_low)
                    & (self._decs <= self._dec_high)
                    & (self._ras >= self._ra_low)
                    & (self._ras <= self._ra_high)
                )[0]
            )
        )

        selected_dec_band = np.where(
            (self._decs >= self._dec_low) & (self._decs <= self._dec_high)
        )[0]

        self._selected = selected

        self._selected_ras = self._ras[selected]

        self._selected_decs = self._decs[selected]

        self._selected_energies = self._energies[selected]

        self.Nprime = len(selected)

        self.N = len(selected_dec_band)

    def _signal_likelihood(self, ra, dec, source_coord, energy):

        return self._direction_likelihood((ra, dec), source_coord, energy)

    def _background_likelihood(self):

        return 1.0 / self._band_solid_angle

    def _get_neg_log_likelihood_ratio(self, ns):
        """
        Calculate the -log(likelihood_ratio) for minimization.

        Uses calculation described in:
        https://github.com/IceCubeOpenSource/SkyLLH/blob/master/doc/user_manual.pdf

        :param ns: Number of source counts.
        """

        one_plus_alpha = 1e-10
        alpha = one_plus_alpha - 1

        log_likelihood_ratio = 0.0

        for i in range(self.Nprime):

            signal = self._signal_likelihood(
                self._selected_ras[i],
                self._selected_decs[i],
                self._source_coord,
                self._energies[i],
            )

            bg = self._background_likelihood()

            chi = (1 / self.N) * (signal / bg - 1)

            alpha_i = ns * chi

            if (1 + alpha_i) < one_plus_alpha:

                alpha_tilde = (alpha_i - alpha) / one_plus_alpha
                log_likelihood_ratio += (
                    np.log1p(alpha) + alpha_tilde - (0.5 * alpha_tilde ** 2)
                )

            else:

                log_likelihood_ratio += np.log1p(alpha_i)

        log_likelihood_ratio += (self.N - self.Nprime) * np.log1p(-ns / self.N)

        return -log_likelihood_ratio

    def __call__(self, ns):
        """
        Wrapper function for convenience.
        """

        return self._get_neg_log_likelihood_ratio(ns)

    def _minimize(self):
        """
        Minimize -log(likelihood_ratio) for the source hypothesis,
        returning the best fit ns and index.

        Uses the iMiuint wrapper.
        """

        init_ns = self._ns_min + (self._ns_max - self._ns_min) / 2
        init_index = 2.19
        
        m = Minuit(
            self._func_to_minimize,
            ns=init_ns,
            index=init_index,
        )
        m.limits["ns"] = (self._ns_min, self._ns_max)
        m.errors["ns"] = 1
        m.errordef = 0.5
        m.migrad()

        self._best_fit_ns = m.values["ns"]

        return m


    def _minimize_grid(self):
        """
        Minimize -log(likelihood_ratio) for the source hypothesis,
        returning the best fit ns and index.

        This simple grid method takes roughly the same time as minuit
        and is more accurate...
        """

        ns_grid = np.linspace(self._ns_min, self._ns_max, 10)

        out = np.zeros(len(ns_grid))
        for i, ns in enumerate(ns_grid):
            out[i] = self._get_neg_log_likelihood_ratio(ns)

        sel = np.where(out == np.min(out))

        if len(sel[0]) > 1:

            self._best_fit_ns = 0.0

        else:

            self._best_fit_ns = ns_grid[sel[0]]

    def get_test_statistic(self):
        """
        Calculate the test statistic for the best fit ns
        """

        self._minimize()

        neg_log_lik = self._get_neg_log_likelihood_ratio(self._best_fit_ns)

        self.likelihood_ratio = np.exp(neg_log_lik)

        self.test_statistic = -2 * neg_log_lik

        return self.test_statistic


class SimplePointSourceLikelihood:
    def __init__(self, direction_likelihood, event_coords, source_coord):
        """
        Point source likelihood with only spatial information.
        Testing out simple algorithms for evaluating TS.
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

        return 1 / (np.deg2rad(self._band_width * 2) * 2 * np.pi)

    def _select_declination_band(self):

        decs = np.array([_[1] for _ in self._event_coords])

        _, source_dec = self._source_coord

        dec_fac = np.deg2rad(self._band_width)

        selected = np.where(
            (decs >= source_dec - dec_fac) & (decs <= source_dec + dec_fac)
        )[0]

        self._selected = selected

        self._selected_event_coords = [
            (ec[0], ec[1])
            for ec in self._event_coords
            if (ec[1] >= source_dec - dec_fac) & (ec[1] <= source_dec + dec_fac)
        ]

        self.N = len(selected)

    def __call__(self, ns):

        log_likelihood = 0.0

        for i in range(self.N):

            signal = (ns / self.N) * self._signal_likelihood(
                self._selected_event_coords[i], self._source_coord
            )

            bg = (1 - (ns / self.N)) * self._background_likelihood()

            log_likelihood += np.log(signal + bg)

        return -log_likelihood


class SimpleWithEnergyPointSourceLikelihood:
    def __init__(
        self, direction_likelihood, energy_likelihood, event_coords, source_coord
    ):
        """
        Simple version of point source likelihood.
        Also including the energy dependence.
        Testing out simple algorithms for evaluating TS.
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

        return self._direction_likelihood(
            event_coord, source_coord
        ) * self._energy_likelihood(energy, index)

    def _background_likelihood(self, energy):

        return (
            1
            / (np.deg2rad(self._band_width * 2) * 2 * np.pi)
            * self._energy_likelihood(energy, self._bg_index)
        )

    def _select_declination_band(self):

        decs = np.array([_[1] for _ in self._event_coords])

        _, source_dec = self._source_coord

        dec_fac = np.deg2rad(self._band_width)

        selected = np.where(
            (decs >= source_dec - dec_fac) & (decs <= source_dec + dec_fac)
        )[0]

        self._selected = selected

        self._selected_event_coords = [
            (ec[0], ec[1])
            for ec in self._event_coords
            if (ec[1] >= source_dec - dec_fac) & (ec[1] <= source_dec + dec_fac)
        ]

        self.N = len(selected)

    def __call__(self, ns):

        log_likelihood = 0.0

        for i in range(self.N):

            signal = (ns / self.N) * self._signal_likelihood(
                self._selected_event_coords[i], self._source_coord
            )

            bg = (1 - (ns / self.N)) * self._background_likelihood()

            log_likelihood += np.log(signal + bg)

        return -log_likelihood
