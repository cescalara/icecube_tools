import numpy as np

"""
Module to compute the IceCube point source likelihood
using publicly available information.

Currently well-defined for searches with
Northern sky muon neutrinos.
"""


class PointSourceLikelihood():
    """
    Calculate the point source likelihood for a given 
    neutrino dataset - in terms of reconstructed 
    energies and arrival directions.
    """    
    
    def __init__(self, direction_likelihood, energy_likelihood):
        """
        Calculate the point source likelihood for a given 
        neutrino dataset - in terms of reconstructed 
        energies and arrival directions.
        
        :param direction_likelihood: An instance of vMFLikelihood.
        :param energy_likelihood: An instance of MarginalisedEnergyLikelihood.
        """

        self._direction_likihood = direction_likelihood 

        self._energy_likelihood = energy_likelihood


    def __call__(self, directions, energies):
        """
        Evaluate the PointSourceLikelihood for the given
        neutrino dataset.

        :param directions: The reconstructed nu arrival directions.
        :param energies: The reconstructed nu enrgies.
        """


                
class MarginalisedEnergyLikelihood():
    """
    Compute the marginalised energy likelihood by using a 
    simulation of a large number of reconstructed muon 
    neutrino tracks. 
    """
    
    
    def __init__(self, energy, sim_index=1.5):
        """
        Compute the marginalised energy likelihood by using a 
        simulation of a large number of reconstructed muon 
        neutrino tracks. 
        
        :param energy: Reconstructed muon energies (preferably many).
        :param sim_index: Spectral index of source spectrum in sim.
        """

        self._energy = energy

        self._sim_index = sim_index

        
    def _calc_weights(self, new_index):

        return  np.power(self._energy, self._sim_index - self._new_index)


    def __call__(self, E, new_index, min_E=1e2, max_E=1e7):
        """
        P(Ereco | index) = \int dEtrue P(Ereco | Etrue) P(Etrue | index)
        """

        self._new_index = new_index

        self._weights = self._calc_weights(new_index)

        bins = np.linspace(np.log10(min_E), np.log10(max_E)) # GeV
        
        self._hist, _ = np.histogram(np.log10(self._energy), bins=bins, weights=self._weights, density=True)
        
        E_index = np.digitize(np.log10(E), bins) - 1

        return self._hist[E_index]
        
