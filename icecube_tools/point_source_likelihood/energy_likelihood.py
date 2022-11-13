import numpy as np
from abc import ABC, abstractmethod
import h5py
from os.path import join

from icecube_tools.detector.r2021 import R2021IRF
from icecube_tools.detector.effective_area import EffectiveArea

"""
Module to compute the IceCube energy likelihood
using publicly available information.

Based on the method described in:
Braun, J. et al., 2008. Methods for point source analysis 
in high energy neutrino telescopes. Astroparticle Physics, 
29(4), pp.299–305.

Currently well-defined for searches with
Northern sky muon neutrinos.
"""


class MarginalisedEnergyLikelihood(ABC):
    """
    Abstract base class for the marginalised energy likelihood.

    L = \int d Etrue P(Ereco | Etrue) P(Etrue | index) = P(Ereco | index).
    """

    @abstractmethod
    def __call__(self):
        """
        Return the value of the likelihood for a given Ereco and index.
        """

        pass


class MarginalisedIntegratedEnergyLikelihood(MarginalisedEnergyLikelihood):
    """
    Calculates energy likelihood by integration rather than simulation.
    """
    #@profile
    def __init__(
        self,
        irf: R2021IRF,
        aeff: EffectiveArea, 
        reco_bins: np.ndarray,
        min_index: float=1.5,
        max_index: float=4.0,
        ):

        # TODO change reco_bins to cover the range provided by all the pdfs
        # and have the coarsest binning of all pdfs
        self._irf = irf
        self._aeff = aeff
        self.reco_bins = reco_bins
        self.true_bins_irf = irf.true_energy_bins
        self.true_bins_aeff = np.log10(aeff.true_energy_bins)
        self.true_energy_bins = np.array(sorted(list(set(self.true_bins_irf).union(self.true_bins_aeff))))
        idx = np.nonzero(
            (self.true_energy_bins <= self.true_bins_irf.max()) & (self.true_energy_bins <= self.true_bins_aeff.max()) & \
            (self.true_energy_bins >= self.true_bins_irf.min()) & (self.true_energy_bins >= self.true_bins_aeff.min())
        )
        self.true_energy_bins = self.true_energy_bins[idx]
        self.declination_bins_irf = irf.declination_bins
        self.cos_z_bins = aeff.cos_zenith_bins
        self.declination_bins_aeff = np.arcsin(-self.cos_z_bins)
        self._min_index = min_index
        self._max_index = max_index
        self.true_bins_c = self.true_energy_bins[:-1] + 0.5 * np.diff(self.true_energy_bins)
        self._previous_index = None
        self._values = {} 

        #pre-calculate cdf values
        self._cdf = np.zeros((self.true_energy_bins.size - 1, 3, self.reco_bins.size - 1))
        for c_true, e_true  in enumerate(self.true_energy_bins[:-1]):
            c_irf_true = np.digitize(e_true, self.true_bins_irf) - 1
            for c_dec, _ in enumerate(self.declination_bins_irf[:-1]):
                for c, (erecol, erecoh) in enumerate(zip(self.reco_bins[:-1], self.reco_bins[1:])):
                    pdf = self._irf.reco_energy[c_irf_true, c_dec]
                    self._cdf[c_true, c_dec, c] = pdf.cdf(erecoh) - pdf.cdf(erecol)

    @profile
    def __call__(self, ereco, index, dec):
        """
        Wrapper on _calc_likelihood to retrieve only the likelihood for a specific Ereco value.
        Saves time by storing data and checking if data of the same index is requested
        over and over again, as is done in point_source_likelihood.py for each event.
        :param ereco: Reconstructed energy in GeV, float or np.ndarray
        :param index: Spectral index > 0
        :param dec: Declination, rad
        :return: Likelihood of reconstructed energy index at declination.
        """

        if index > self._max_index:
            raise ValueError("Index too high")
        elif index < self._min_index:
            raise ValueError("Index too low")

        log_ereco = np.log10(ereco)
        reco_ind = np.digitize(log_ereco, self.reco_bins) - 1    # is np.ndarray
        dec_ind = np.digitize(dec, self.declination_bins_aeff) - 1 # is np.ndarray

        dec_ind_set = set(dec_ind)
        output = np.zeros_like(log_ereco)
        # loop over set(sec_ind):
        for dec_idx in dec_ind_set:
            single_dec = self.declination_bins_aeff[dec_idx]
            self._values[dec_idx] = self._calc_likelihood(index, single_dec)
            needed = np.nonzero((dec_ind == dec_idx))
            output[needed] = self._values[dec_idx][reco_ind[needed]]

        return output
        
        #if there was a previous index, look it up,
        # edit: can be deleted, was to save time in external for-loop!
        if self._previous_index is not None:
            #if asked for index is close to previous, look up declination
            if np.isclose(self._previous_index, index):
                try:
                    return self._values[dec_ind][reco_ind]
                except KeyError:
                    self._values[dec_ind] = self._calc_likelihood(index, dec)
            #else calculate from scratch and overwrite all previous values
            else:
                self._previous_index = index
                self._values = {}
                self._values[dec_ind] = self._calc_likelihood(index, dec)
        else:
            self._values[dec_ind] = self._calc_likelihood(index, dec)
        return self._values[dec_ind][reco_ind]


    #@profile
    def _calc_likelihood(self, index, dec):
        """
        Calculates likelihood for new reco energy binning for given index at given declination.
        """

        irf_dec_ind = np.digitize(dec, self.declination_bins_irf) - 1        

        #pre-calculate power law and aeff part, is not dependent on reco energy
        pl = np.zeros(self.true_energy_bins.size - 1)
        for c, (etruel, etrueh) in enumerate(zip(
                self.true_energy_bins[:-1], self.true_energy_bins[1:])
            ):
            
            pl[c] = self.integrated_power_law(etrueh, etruel, index)

        aeff = self._aeff.detection_probability(
            np.power(10, self.true_bins_c), -np.sin(dec), 1e8
        )
        
        values = np.zeros(self.reco_bins.size - 1)
        for c_reco, (erecol, erecoh) in enumerate(
            zip(self.reco_bins[:-1], self.reco_bins[1:])
        ):

            # Can this be done in without the loop?
            sum_this = pl * self._cdf[:, irf_dec_ind, c_reco]
            values[c_reco] = np.dot(sum_this, aeff)
            
        values = values / np.sum(values * np.diff(self.reco_bins))
        return values


    @staticmethod
    def integrated_power_law(loge_high, loge_low, index):
        return 1. / (1 - index) * \
            (np.power(10, -loge_high * (index - 1)) - np.power(10, -loge_low * (index - 1)))


    @staticmethod
    def power_law_loge(loge, index):
        return np.power(np.power(10, loge), -index + 1)



class MarginalisedEnergyLikelihood2021(MarginalisedEnergyLikelihood):
    """
    Compute the marginalised energy likelihood by reading in the provided IRF data of 2021.
    Creates instances of MarginalisedEnergyLikelihoodFromSimFixedIndex (slightly copied from
    MarginalisedEnergyLikelihoodFromSim but with different interpolating) for each given index.
    If the likelihood is requested for an index not provided with a dataset,
    the likelihood will be interpolated (linearly) between provided indices.
    """

    def __init__(self,
                 index_list,
                 path,
                 fname,
                 src_dec,
                 ftype='h5',
                 min_index=1.5,
                 max_index=4.0,
                 min_E=1e2,
                 max_E=1e9,
                 min_sind=-0.1,
                 max_sind=1.,
                 Ebins=50,
    ):
        """
        Initialise all datasets
        User needs to make sure that data sets cover the entire declination needed.
        
        :param index_list: List of indices provided with datasets
        :param path: Path where datasets are located
        :param fname: Filename, bar ending of `_{index:.1f}.txt`
        :param src_dec: Source declination in radians
        """
        #TODO change path thing and loading of data? maybe option to pass data directly
        # for each index, load a different MarginalisedEnergyLikelihoodFromSim
        #distinguish between used data set/likelihood for different indices
        self.index_list = sorted(index_list)
        self.likelihood = {}
        
        for c, i in enumerate(self.index_list):
            filename = join(path, f"{fname}_index_{i:.1f}.h5")
            print(filename)
            with h5py.File(filename, "r") as f:
                reco_energy = f["reco_energy"][()]
                dec = f["dec"][()]
                #ang_err not needed
                #ang_err = f["ang_err"][()]
            self.likelihood[f"{float(i):1.1f}"] = MarginalisedEnergyLikelihoodFromSimFixedIndex(
                reco_energy,
                dec,
                i,
                src_dec,
                min_E,
                max_E,
                min_sind,
                max_sind,
                Ebins
            )
        self.lls = np.zeros((len(index_list), self.likelihood[f"{float(i):1.1f}"]._energy_bins.shape[0]-1))
        for c, i in enumerate(self.index_list):
            self.lls[c, :] = self.likelihood[f"{i:.1f}"].likelihood
            self._energy_bins = self.likelihood[f"{i:.1f}"]._energy_bins

        #decide on max/min index based on provided simulations
        #if range of simulations is smaller, use these values
        #else use user-provided values
        self._delta_index = 0.05

        if max_index > max(index_list) - self._delta_index:
            self._max_index = max(index_list) - self._delta_index
        else:
            self._max_index = max_index

        if min_index < min(index_list) + self._delta_index:
            self._min_index = min(index_list) + self._delta_index
        else:
            self._min_index = min_index

        self._min_E = min_E
        self._max_E = max_E
        self._min_sind = min_sind
        self._max_sind = max_sind
        self._Ebins = Ebins
       

    def __call__(self, E, index, dec=0):
        """
        Returns likelihood of reconstructed energy for specified spectral index.
        :param E: Reconstructed energy in GeV, may be float or np.ndarray
        :param index: spectral index
        :param dec: dummy argument
        :return: Likelihood
        :raise ValueError: if the requested index is out of range.
        :raise ValueError: if any other interpolation than `log` or `lin` is requested. 
        """

        if index < min(self.index_list) or index > max(self.index_list):
            raise ValueError(f"Index {index} outside of range of index list.")

        if index not in self.index_list:
            raise ValueError("Only indices with simulation are allowed.")
        idx = np.digitize(np.log10(E), self._energy_bins) - 1
        

        index_index = np.digitize(index, self.index_list) - 1
        if index == max(self.index_list):
            index_index -= 1
        return self.lls[index_index, idx]


    def calc_loglike(self, energies, index):
        """
        Function intended for testing only.
        """

        loglike = 0
        self.faulty = []
        for e in energies:
            temp = self.__call__(e, index)
            if temp == 0.0:
                self.faulty.append(e)
                temp = 1e-10
            loglike += np.log10(temp)

        return -loglike


class MarginalisedEnergyLikelihoodFromSimFixedIndex(MarginalisedEnergyLikelihood):
    """
    Copied from MarginalisedEnergyLikelihoodFromSim but without the interpolating
    """

    def __init__(self,
                 energy,
                 dec,
                 sim_index,
                 src_dec=0.,
                 min_E=1e2,
                 max_E=1e9,
                 min_sind=-0.1,
                 max_sind=1.,
                 Ebins=50,
                 
    ):
        """
        :param energy: List of reconstructed energies from simulatedevents
        :param dec: List of declinations from simulated events
        :param sim_index: Spectral index used for the simulated events
        :param src_dec: Declination of source to be analised, in radians
        """

        self._energy = energy
        self._dec = dec
        self._sim_index = sim_index
        self._min_E = min_E
        self._max_E = max_E
        self._min_sind=min_sind
        self._max_sind=max_sind
        self._energy_bins = np.linspace(np.log10(min_E), np.log10(max_E), Ebins)  # GeV
        self._sin_dec_bins = np.linspace(min_sind, max_sind, 20)
        self.src_dec = src_dec


    def __call__(self, E):
        """
        Return likelihood of some reconstructed energy.
        :param E: Reconstructed energy in GeV, may be float or np.ndarray
        :return: Likelihood
        """

        idx = np.digitize(np.log10(E), self._energy_bins) - 1
        return self.likelihood[idx]


    @property
    def src_dec(self):
        return self._src_dec


    @src_dec.setter
    def src_dec(self, val):
        self._src_dec = val
        self._precompute_histograms()


    def _precompute_histograms(self):
        """
        Computes histograms of reconstructed energy from data set.
        Only uses data close in declination to specified source to account for declination dependence.
        """

        sind_idx = np.digitize(np.sin(self._src_dec), self._sin_dec_bins) - 1
        idx = (np.sin(self._dec) >= self._sin_dec_bins[sind_idx]) & (
            np.sin(self._dec) < self._sin_dec_bins[sind_idx + 1]
        )
        self._selected_energy = self._energy[idx]
        self.likelihood, _ = np.histogram(
                np.log10(self._selected_energy),
                bins=self._energy_bins,
                density=True
        )



class MarginalisedEnergyLikelihoodFromSim(MarginalisedEnergyLikelihood):
    """
    Compute the marginalised energy likelihood by using a 
    simulation of a large number of reconstructed muon 
    neutrino tracks. 
    """

    def __init__(
        self,
        energy,
        dec,
        sim_index=1.5,
        min_index=1.5,
        max_index=4.0,
        min_E=1e2,
        max_E=1e9,
        min_sind=-0.1,
        max_sind=1.0,
        Ebins=50,
    ):
        """
        Compute the marginalised energy likelihood by using a 
        simulation of a large number of reconstructed muon 
        neutrino tracks. 
        
        :param energy: Reconstructed muon energies (preferably many).
        :param dec: Reconstrcuted decs corresponding to these events.
        :param sim_index: Spectral index of source spectrum in sim.
        """

        self._energy = energy

        self._dec = dec

        self._sim_index = sim_index

        self._min_index = min_index
        self._max_index = max_index

        self._min_E = min_E
        self._max_E = max_E

        self._index_bins = np.linspace(min_index, max_index)

        self._energy_bins = np.linspace(np.log10(min_E), np.log10(max_E), Ebins)  # GeV

        self._sin_dec_bins = np.linspace(min_sind, max_sind, 20)

        self._src_dec = None


    def set_src_dec(self, src_dec):
        """
        Set the source declination in private variable
        Precompute likelihood distributions
        """

        self._src_dec = src_dec

        self._precompute_histograms()

    def _calc_weights(self, new_index):
        """
        Only compute one simulation with some given spectral index, num=index_bins.
        In order to test against multiple indices, re-use existing data
        but give new weights according to new spectral index.
        If |gamma| > |simulated gamma|, less weight needs to be at higher energies
        Convetion: index is positive, minus is explicitely stated in equations,
        assume flat spectrum (gamma=0) simulated:
        for gamma=2, shift to lower energies -> self._sim_index - new_index = 0 - 2 = -2
        -> the higher the energy, the lower the weight!
        """

        return np.power(self._selected_energy, self._sim_index - new_index)

    def _precompute_histograms(self):
        """
        Creates histograms of for each tested spectral index:
        self._likelihood empty list, one entry for each spectral index
        index of sin(dec) of source in list of sin(dec) sourrounding source
        energy is list of all Ereco from simulated events, index (idx) those who belong to correct declinations
        _selected_energy then contains all Ereco belonging to the selected events
        get index bin center
        create histogram (i.e. probability of finding some Ereco for given spectral index) for each spectral index
        """
        #TODO maybe change the sin(dec) bins to something more like +/- specified range?
        #what if src dec is right at a bin edge? too many events discarded!
        self._likelihood = np.zeros(
            (len(self._index_bins[:-1]), len(self._energy_bins[:-1]))
        )

        sind_idx = np.digitize(np.sin(self._src_dec), self._sin_dec_bins) - 1

        #only use events within the declination band hosting the source
        idx = (np.sin(self._dec) >= self._sin_dec_bins[sind_idx]) & (
            np.sin(self._dec) < self._sin_dec_bins[sind_idx + 1]
        )

        self._selected_energy = self._energy[idx]

        for i, index in enumerate(self._index_bins[:-1]):

            index_cen = index + (self._index_bins[i + 1] - index) / 2

            weights = self._calc_weights(index_cen)

            hist, _ = np.histogram(
                np.log10(self._selected_energy),
                bins=self._energy_bins,
                weights=weights,
                density=True,
            )

            self._likelihood[i] = hist

    def __call__(self, E, new_index, dec):
        """
        P(Ereco | index) = \int dEtrue P(Ereco | Etrue) P(Etrue | index)
        """

        #check for E out of bounds
        if E < self._min_E or E > self._max_E:

            raise ValueError(
                "Energy "
                + str(E)
                + "is not in the accepted range between "
                + str(self._min_E)
                + " and "
                + str(self._max_E)
            )

        #check for index out of bounds
        if new_index < self._min_index or new_index > self._max_index:

            raise ValueError(
                "Sepctral index "
                + str(new_index)
                + " is not in the accepted range between "
                + str(self._min_index)
                + " and "
                + str(self._max_index)
            )

        i_index = np.digitize(new_index, self._index_bins) - 1

        E_index = np.digitize(np.log10(E), self._energy_bins) - 1

        return self._likelihood[i_index][E_index]


class MarginalisedEnergyLikelihoodFixed(MarginalisedEnergyLikelihood):
    """
    Compute the marginalised energy likelihood for a fixed case based on a simulation.
    Eg. P(E | atmos + diffuse astro).
    """

    def __init__(
        self,
        energy,
        min_index=1.5,
        max_index=4.0,
        min_E=1e2,
        max_E=1e9,
        min_sind=-0.1,
        max_sind=1.0,
        Ebins=50,
    ):
        """
        Compute the marginalised energy likelihood for a fixed case based on a simulation.
        Eg. P(E | atmos + diffuse astro).
            
        :param energy: Reconstructed muon energies (preferably many) [GeV].
        """

        self._energy = energy

        self._min_index = min_index
        self._max_index = max_index

        self._min_E = min_E
        self._max_E = max_E

        self._energy_bins = np.linspace(np.log10(min_E), np.log10(max_E), Ebins)  # GeV

        self._precompute_histogram()

    def _precompute_histogram(self):

        hist, _ = np.histogram(
            np.log10(self._energy), bins=self._energy_bins, density=True
        )

        self._likelihood = hist

    def __call__(self, E):

        E_index = np.digitize(np.log10(E), self._energy_bins) - 1

        return self._likelihood[E_index]


class MarginalisedEnergyLikelihoodBraun2008(MarginalisedEnergyLikelihood):
    """
    Compute the marginalised enegry likelihood using 
    Figure 4 in Braun+2008. 
    """

    def __init__(self, energy_list, pdf_list, index_list):
        """
        Compute the marginalised enegry likelihood using 
        Figure 4 in Braun+2008. 

        :param energy_list: list of Ereco values (x-axis)
        :param pdf_list: list of P(Ereco | index) values (y-axis)
        :param index_list: list of spectral index inputs (diff lines)
        """

        self._energy_list = energy_list

        self._pdf_list = pdf_list

        self._index_list = index_list

        self._min_index = min(index_list)

        self._max_index = max(index_list)

    def __call__(self, energy, index, dec=0):
        """
        Return P(Ereco | index)
        """

        pdf_vals_at_E = [
            np.interp(energy, e, p) for e, p in zip(self._energy_list, self._pdf_list)
        ]

        pdf_val_at_index = np.interp(index, self._index_list, pdf_vals_at_E)

        return pdf_val_at_index


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

    weights = np.array([np.power(_, sim_index - new_index) for _ in energies])

    hist, bins = np.histogram(
        np.log10(energies), bins=bins, weights=weights, density=True
    )

    bin_midpoints = bins[:-1] + np.diff(bins) / 2

    cdf = np.cumsum(hist)
    cdf = cdf / float(cdf[-1])

    values = np.random.rand(len(energies))

    value_bins = np.searchsorted(cdf, values)

    random_from_cdf = bin_midpoints[value_bins]

    energies = np.power(10, random_from_cdf)

    return energies


def read_input_from_file(filename):
    """
    Helper function to read in data digitized from plots.
    """

    import h5py

    keys = ["E-2_spectrum", "E-2.5_spectrum", "E-3_spectrum", "atmospheric"]

    index_list = []
    energy_list = []
    pdf_list = []

    with h5py.File(filename, "r") as f:

        for key in keys:

            index_list.append(f[key]["index"][()])

            energy_list.append(f[key]["reco_energy"][()])

            pdf_list.append(f[key]["pdf"][()])

    return energy_list, pdf_list, index_list
