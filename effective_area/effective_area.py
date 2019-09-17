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



        
