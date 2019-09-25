import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
import h5py

from tqdm.autonotebook import tqdm as progress_bar

from detector import Detector
from source_model import Source, DIFFUSE, POINT
from neutrino_calculator import NeutrinoCalculator

"""
Module for running neutrino production 
and detection simulations.
"""


class Simulator():

    def __init__(self, source, detector):

        self._source = source

        self._detector = detector

        self.max_cosz = 1 

        self.time = 1 # year

        
    @property
    def source(self):

        return self._source

    @source.setter
    def source(self, value):

        if not isinstance(value, Source):

            raise ValueError(str(value) + ' is not an instance of Source.')

        else:

            self._source = source

    @property
    def detector(self):

        return self._detector

    @detector.setter
    def detector(self):

        if not isinstance(value, Detector):

            raise ValueError(str(value) + ' is not an instance of Detector')

        
    def _get_expected_number(self):
        """
        Find the expected number of neutrinos.
        """
        
        nu_calc = NeutrinoCalculator(self.source, self.detector.effective_area)

        self._Nex = nu_calc(time=self.time, min_energy=self.source.flux_model._lower_energy, max_cosz=self.max_cosz)
        
        
    def run(self, N=None):
        """
        Run a simulation for the given source 
        and detector configuration.
        """

        if not N:

            self._get_expected_number()

            self.N = np.random.poisson(self._Nex)           
            
        else:

            self.N = int(N)
            
        self.true_energy = []
        self.reco_energy = []
        self.coordinate = []
        self.ra = []
        self.dec = []
        
        max_energy = self.source.flux_model._upper_energy
        
        for i in progress_bar(range(self.N), desc='Sampling'):

            accepted = False
            
            while not accepted:

                Etrue = self.source.flux_model.sample(1)[0]
                
                ra, dec = sphere_sample()
                cosz = -np.sin(dec)

                if cosz > self.max_cosz:

                    detection_prob = 0

                else:
                
                    detection_prob = float(self.detector.effective_area.detection_probability(Etrue, cosz, max_energy))

                accepted = np.random.choice([True, False], p=[detection_prob, 1-detection_prob])
                
            self.true_energy.append(Etrue)

            Ereco = self.detector.energy_resolution.sample(Etrue)
            self.reco_energy.append(Ereco)
            
            if self.source.source_type == DIFFUSE:

                self.coordinate.append(SkyCoord(ra*u.rad, dec*u.rad, frame='icrs'))
                self.ra.append(ra)
                self.dec.append(dec)
                
            else:

                reco_ra, reco_dec = self.detector.angular_resolution.sample(Etrue, ra, dec)
                self.coordinate.append(SkyCoord(reco_ra*u.rad, reco_dec*u.rad, frame='icrs'))
                self.ra.append(ra)
                self.dec.append(dec)
                

    def save(self, filename):
        """
        Save the output to filename.
        """

        self._filename = filename

        with h5py.File(filename, 'w') as f:

            f.create_dataset('true_energy', data=self.true_energy)

            f.create_dataset('reco_energy', data=self.reco_energy)

            f.create_dataset('ra', data=self.ra)

            f.create_dataset('dec', data=self.dec)

            f.create_dataset('index', data=self.source.flux_model._index)

            f.create_dataset('source_type', data=self.source.source_type)

                
def sphere_sample(N=1, radius=1):
    """
    Sample points uniformly on a sphere.
    """

    u = np.random.uniform(0, 1, N)
    v = np.random.uniform(0, 1, N)
            
    phi = 2 * np.pi * u
    theta = np.arccos(2 * v - 1)

    ra, dec = spherical_to_icrs(theta, phi)
    
    return ra, dec


def spherical_to_icrs(theta, phi):
    """
    convert spherical coordinates to ICRS
    ra, dec.
    """

    ra = phi

    dec = np.pi/2 - theta

    return ra, dec


