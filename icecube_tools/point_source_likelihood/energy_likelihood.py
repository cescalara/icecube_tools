import numpy as np
from abc import ABC, abstractmethod

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
    

    
class MarginalisedEnergyLikelihoodFromSim(MarginalisedEnergyLikelihood):
    """
    Compute the marginalised energy likelihood by using a 
    simulation of a large number of reconstructed muon 
    neutrino tracks. 
    """
    
    
    def __init__(self, energy, dec, sim_index=1.5, min_index=1.5, max_index=4.0,
                 min_E=1e2, max_E=1e9, min_sind=-0.1, max_sind=1.0):
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

        self._energy_bins = np.linspace(np.log10(min_E), np.log10(max_E)) # GeV

        self._sin_dec_bins = np.linspace(min_sind, max_sind, 20)

        self._src_dec = None

        
    def set_src_dec(self, src_dec):

        self._src_dec = src_dec
        
        self._precompute_histograms()

        
    def _calc_weights(self, new_index):

        return np.power(self._selected_energy, self._sim_index - new_index)

    
    def _precompute_histograms(self):

        self._likelihood = np.zeros( (len(self._energy_bins[:-1]), len(self._energy_bins[:-1])) )

        sind_idx = np.digitize(np.sin(self._src_dec), self._sin_dec_bins)-1

        idx = (np.sin(self._dec) >= self._sin_dec_bins[sind_idx]) & (np.sin(self._dec) < self._sin_dec_bins[sind_idx+1])

        self._selected_energy = self._energy[idx]
        
        for i, index in enumerate(self._index_bins[:-1]):

            index_cen = index + (self._index_bins[i+1] - index)/2 
            
            weights = self._calc_weights(index_cen)

            hist, _ = np.histogram(np.log10(self._selected_energy), bins=self._energy_bins,
                                   weights=weights, density=True)

            self._likelihood[i] = hist
        

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

    

class MarginalisedEnergyLikelihoodFixed(MarginalisedEnergyLikelihood):
    """
    Compute the marginalised energy likelihood for a fixed case based on a simulation.
    Eg. P(E | atmos + diffuse astro).
    """
    

    def __init__(self, energy, min_index=1.5, max_index=4.0,
                 min_E=1e2, max_E=1e9, min_sind=-0.1, max_sind=1.0):
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

        self._energy_bins = np.linspace(np.log10(min_E), np.log10(max_E)) # GeV

        self._precompute_histogram()

        
    def _precompute_histogram(self):
        

        hist, _ = np.histogram(np.log10(self._energy), bins=self._energy_bins, density=True)

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
        

    def __call__(self, energy, index):
        """
        Return P(Ereco | index)
        """

        pdf_vals_at_E = [np.interp(energy, e, p) for e, p in zip(self._energy_list, self._pdf_list)]
        
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


       
def read_input_from_file(filename):
    """
    Helper function to read in data digitized from plots.
    """

    import h5py

    keys = ['E-2_spectrum', 'E-2.5_spectrum', 'E-3_spectrum', 'atmospheric']    

    index_list = []
    energy_list = []
    pdf_list = []
    
    with h5py.File(filename, 'r') as f:

        for key in keys:

            index_list.append(f[key]['index'][()])

            energy_list.append(f[key]['reco_energy'][()])

            pdf_list.append(f[key]['pdf'][()])

    return energy_list, pdf_list, index_list
