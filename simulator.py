import numpy as np

from detector import Detector
from source_model import Source, DIFFUSE, POINT
from neutrino calculator import NeutrinoCalulator

"""
Module for running neutrino production 
and detection simulations.
"""


class Simulator():

    def __init__(self, source, detector):

        self._source = source

        self._detector = detector

        self.min_energy = 1e2 # GeV

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


    def run(self):
        """
        Run a simulation for the given source 
        and detector configuration.
        """

        self.kwargs = kwargs
        
        # Sample number of events
        nu_calc = NeutrinoCalculator(self.source, self.detector.effective_area)

        self._Nex = nu_calc(time=self.time, min_energy=self.min_energy, max_cosz=self.max_cosz)

        self.N = np.random.poisson(self._Nex)

        true_energy = []
        reco_energy = []
        coordinate = []
        for i in range(self.N):

            accepted = False

            while not accepted:

                Etrue = self.source.flux.model.sample(self.min_energy)

                ra, dec = sphere_sample()
                cosz = -np.sin(dec)
                
                detection_prob = detector.effective_area.detection_probability(Etrue, cosz)

                accepted = np.random.choice([True, False], p=[detection_prob, 1-detection_prob])
                
            true_energy.append(Etrue)

            Ereco = self.detector.energy_resolution.sample(Etrue)
            reco_energy.append(Ereco)
            
            if self.source.source_type == DIFFUSE:

                coordinate.append(SkyCoord(ra*u.rad, dec*u.rad, frame='icrs'))

            else:

                reco_ra, reco_dec = self.detector.angular_resolution.sample(Etrue, ra, dec)
                coordinate.append(SkyCoord(reco_ra*u.rad, reco_dec*u.rad, frame='icrs'))
            

def sphere_sample(N=1, radius=1):
    """
    Sample points uniformly on a sphere.
    """

    u = np.random.uniform(0, 1, N)
    v = np.random.uniform(0, 1, N)
            
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)

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

