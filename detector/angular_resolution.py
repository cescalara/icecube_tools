import numpy as np
from abc import ABC, abstractmethod

"""
Module for handling the angular resolution
of IceCube based on public data.
"""

R2018_ANG_RES_FILENAME = "AngRes.txt"


class IceCubeAngResReader(ABC):
    """
    Abstract base class for different input files
    from the IceCube website.
    """

    def __init__(self, filename):
        """
        Abstract base class for different input files
        from the IceCube website.
        
        :param filename: Name of file to read.
        """

        self._filename = filename

        self.ang_res_values = None

        self.true_energy_bins = None
        
        self.read()
        
        super().__init__()
        

    @abstractmethod
    def read(self):

        pass


class R2018AngResReader(IceCubeAngResReader):
    """
    Reader for the 2018 Oct 18 release.
    Link: https://icecube.wisc.edu/science/data/PS-3years.
    """

    def read(self):

        import pandas as pd
        
        self.year = int(self._filename[-15:-11])
        self.nu_type = 'nu_mu'
         
        filelayout = ['E_min [GeV]', 'E_max [GeV]', 'Med. Resolution[deg]']
        output = pd.read_csv(self._filename, comment='#',
                             delim_whitespace=True,
                             names=filelayout).to_dict()
        
        true_energy_lower = set(output['E_min [GeV]'].values())
        true_energy_upper = set(output['E_max [GeV]'].values())
    
        self.true_energy_bins = np.array( list(true_energy_upper.union(true_energy_lower)) )
        self.true_energy_bins.sort()
                
        self.ang_res_values = np.array( list(output['Med. Resolution[deg]'].values()) )
        

class AngularResolution():

    def __init__(self, filename):

        self._filename = filename
        
        self._reader = self.get_reader()

        self.values = self._reader.ang_res_values

        self.true_energy_bins = self._reader.true_energy_bins
        

    def get_reader(self):
        """
        Define an IceCubeAeffReader based on the filename.
        """      

        if R2018_ANG_RES_FILENAME in self._filename:

            return R2018AngResReader(self._filename)

        else:

            raise ValueError(self._filename + ' is not recognised as one of the known angular resolution files.')
        
