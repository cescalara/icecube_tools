import numpy as np

"""
Wrapper for MarginalisedEnergyLikelihoodBraun2008
to easily sampler Ereco directly form P(Ereco | index).

Used in Braun2008Simulator.
"""


class RecoEnergySampler():


    def __init__(self, marginalised_energy_likelihood):
        """
        :param marginalised_energy_likelihood: Instance of MarginalisedEnergyLikelihoodBraun2008
        """

        self._likelihood = marginalised_energy_likelihood


    def set_index(self, index):

        self._index = index

        E_list = 10**np.linspace(1, 7)

        pdf_vals = [self._likelihood(_, index) for _ in E_list]
        
        self._max_pdf = max(pdf_vals)

        self._min_pdf = min(pdf_vals)
        
        
    def __call__(self):
        """
        Sample a Ereco for a given index.
        Uses rejection sampling.

        :param index: Spectral index of source
        """

        accepted = False

        while not accepted:

            test_log10E = np.random.uniform(1, 7)

            test_pdf = np.random.uniform(self._min_pdf, self._max_pdf) 
        
            if test_pdf < self._likelihood(10**test_log10E, self._index):

                accepted = True

        return 10**test_log10E

 
    
        
        

    
