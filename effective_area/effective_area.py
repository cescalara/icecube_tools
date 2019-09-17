import numpy as np
from abc import ABC, abstractmethod

"""
Module for working with the public IceCube
effective area information.
"""



class IceCubeAeffReader(ABC):
    """
    Abstract base class for a file reader to handle
    the different types of data files provided 
    on the IceCube website.
    """

    def __init__(self, filename):
        """
        Abstract base class for a file reader to handle
        the different types of data files provided 
        on the IceCube website.

        :param filename: name of the file to be read from (string).
        """

        self._filename = filename

        self.effective_area_values = None

        self.true_energy_bins = None

        self.cos_zenith_bins = None 

        self._label_order = {'true_energy' : 0, 'cos_zenith' : 1}

        self._units = {'effective_area' : 'm^2', 'true_energy' : 'GeV', 'cos_zenith' : ''}

        self.read()
        
        super().__init__()


    @abstractmethod
    def read(self):
        """
        To be defined.
        """

        pass
    

    
class R2015AeffReader(IceCubeAeffReader):
    """
    Reader for the 2015 Aug 20 release.
    Link: https://icecube.wisc.edu/science/data/HE_NuMu_diffuse.
    """

    def __init__(self, filename):
        
        super().__init__(filename)

        self._label_order['reconstructed_energy'] = 2

        self._units['reconstructed_energy'] = 'GeV'

        
    def read(self, year=2011, nu_type='nu_mu'):
        """
        Read input from the provided HDF5 file.
        """

        import h5py

        self.year = year
        self.nu_type = nu_type
        
        with h5py.File(self._filename, 'r') as f:

            directory = f[str(year) + '/' + nu_type + '/']
            
            self.effective_area_values = directory['area'][()]

            self.true_energy_bins = directory['bin_edges_0'][()]

            self.cos_zenith_bins = directory['bin_edges_1'][()]
            
            self.reconstructed_energy_bins = directory['bin_edges_2'][()]


            
class R2018AeffReader(IceCubeAeffReader):
    """
    Reader for the 2018 Oct 18 release.
    Link: https://icecube.wisc.edu/science/data/PS-3years.
    """

    def read(self):

        import pandas as pd
        
        self.year = int(self._filename[-22:-18])
        self.nu_type = 'nu_mu'
         
        filelayout = ['Emin', 'Emax', 'cos(z)min', 'cos(z)max', 'Aeff']
        output = pd.read_csv(self._filename, comment = '#',
                             delim_whitespace = True,
                             names = filelayout).to_dict()
        
        true_energy_lower = set(output['Emin'].values())
        true_energy_upper = set(output['Emax'].values())

        cos_zenith_lower = set(output['cos(z)min'].values())
        cos_zenith_upper = set(output['cos(z)max'].values())
        
        self.true_energy_bins = np.array( list(true_energy_upper.union(true_energy_lower)) )

        self.cos_zenith_bins = np.array( list(cos_zenith_upper.union(cos_zenith_lower)) )

        self.effective_area_values = np.reshape(list(output['Aeff'].values()),
                                                (len(true_energy_lower),
                                                 len(cos_zenith_lower)))
        

        
class IceCubeEffectiveArea(ABC):
    """
    Abstract base class for the IceCube effective area.
    """
    
