import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
import h5py
from scipy.stats import uniform
import logging
import sys
from os.path import join
from typing import List, Dict

# from memory_profiler import profile
from tqdm import tqdm as progress_bar
#maybe add from tqdm.contrib.logging import logging_redirect_tqdm
#to re-route logging to tqdm for nicer printing



from .detector.detector import Detector, TimeDependentIceCube
from .source.source_model import Source, DIFFUSE, POINT
from .source.flux_model import PowerLawFlux, BrokenPowerLawFlux
from .neutrino_calculator import NeutrinoCalculator
from .detector.angular_resolution import FixedAngularResolution, AngularResolution
from .detector.r2021 import R2021IRF
from .utils.data import Uptime, data_directory, SimEvents

"""
Module for running neutrino production 
and detection simulations.
"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Simulator(SimEvents):
    def __init__(self, sources, detector, period):
        """
        Class for handling simple neutrino production
        and detection simulations.

        :param sources: List of/single Source object.
        """
        super().__init__()
        logger.info("Instantiating simulation.")
        if not isinstance(sources, list):
            sources = [sources]

        self._sources = sources

        self._detector = detector

        self.max_cosz = 1
        self.min_cosz = -1

        self.time = 1  # year

        self._periods = [period]
        self._period = period


    @property
    def sources(self):

        return self._sources

    @sources.setter
    def sources(self, value):

        if not isinstance(value, Source):

            raise ValueError(str(value) + " is not an instance of Source.")

        else:

            self._sources.append(value)

    @property
    def detector(self):

        return self._detector

    @detector.setter
    def detector(self, value):

        if not isinstance(value, Detector):

            raise ValueError(str(value) + " is not an instance of Detector")

        self._detector = value

    def _get_expected_number(self):
        """
        Find the expected number of neutrinos.
        """

        nu_calc = NeutrinoCalculator(self.sources, self.detector.effective_area)

        self._Nex = nu_calc(time=self.time, min_cosz=self.min_cosz, max_cosz=self.max_cosz)

        self._source_weights = np.array(self._Nex) / sum(self._Nex)

    #@profile
    def run_energy(self, N=None, seed=1234):
        """
        Run a simulation of energy reconstruction for the given set of sources
        and detector configuration.
        The expected number of neutrinos will be
        calculated for each source. If total N is forced,
        then the number from each source will be weighted
        accordingly.
        :param N: Set expected number of neutrinos manually.
        :return: Arrival energy, reconstructed energy in GeV.
        """

        np.random.seed(seed)

        self._get_expected_number()

        if not N:

            self.N = np.random.poisson(sum(self._Nex))

        else:

            self.N = int(N)

        v_min = - self.max_cosz
        v_max = - self.min_cosz



        self._true_energy = []
        self._arrival_energy = []
        self._reco_energy = []
        self.coordinate = []
        self._ra = []
        self._dec = []
        self._source_label = np.zeros(self.N, dtype=int)
        self._ang_err = []        

        #During detector simulation, many events are discarded due to 
        #the detection probability (scaled version of the effective area).
        #To increase speed, take many more events at once and only keep
        #the accepted ones.

        #TODO maybe change the factor to something spectral index dependent
        num = self.N * 1000
        label = np.random.choice(range(len(self.sources)), self.N, p=self._source_weights)
        l_set = set(label)
        l_num = {i: np.argwhere(i == label).shape[0] for i in l_set}
        max_energy = {}
        #create dicts of empty arrays of matching length (i.e. expected events) for each surce
        ra_d = {i: np.zeros(l_num[i]) for i in l_set}
        dec_d = {i: np.zeros(l_num[i]) for i in l_set}
        Etrue_d = {i: np.zeros(l_num[i]) for i in l_set}
        Earr_d = {i: np.zeros(l_num[i]) for i in l_set}
        
        accepted = np.zeros(self.N, dtype=bool)
 
        #go over each source
        for i in l_set:
            #simulate until appropriate number of events is accepted

            max_energy[i] = self.sources[i].flux_model._upper_energy

            logger.info(f"source: {i}")

            while True:
                #check if data is needed, else break loop
                where_zero = np.argwhere(Etrue_d[i] == 0.)
                if where_zero.size == 0:
                    logger.debug("no more empty slots, done")
                    break

                Etrue_ = self.sources[i].flux_model.sample(num)
                if self.sources[i].source_type == DIFFUSE:

                    ra_, dec_ = sphere_sample(v_min=v_min, v_max=v_max, N=num)

                else:

                    ra_, dec_ = np.full(num, self.sources[i].coord[0]), np.full(num, self.sources[i].coord[1])

                cosz = -np.sin(dec_)


                Earr_ = Etrue_ / (1 + self.sources[i].z)
                detection_prob = self.detector.effective_area.detection_probability(
                        Earr_, cosz, max_energy[i]
                ).astype(float)
                

                samples = uniform.rvs(size=num)
                accepted_ = samples <= detection_prob
                idx = np.nonzero(accepted_)
                if idx[0].size == 0:
                    continue
                else:
                    start = np.min(where_zero)
                    end = start + idx[0].size
                    try:
                        Etrue_d[i][start:end] = Etrue_[idx]
                        Earr_d[i][start:end] = Earr_[idx]
                        ra_d[i][start:end] = ra_[idx]
                        dec_d[i][start:end] = dec_[idx]
                        logger.debug("All data placed.")
                    except (IndexError, ValueError):
                        logger.debug("Not enough slots, cutting short.")
                        remaining = np.argwhere(Etrue_d[i] == 0.).size
                        Etrue_d[i][start:] = Etrue_[idx][0:remaining]
                        Earr_d[i][start:] = Earr_[idx][0:remaining]
                        ra_d[i][start:] = ra_[idx][0:remaining]
                        dec_d[i][start:] = dec_[idx][0:remaining]
                        break
            
            
            if not isinstance(self.detector.energy_resolution, R2021IRF):
                Ereco = self.detector.energy_resolution.sample(Earr_d[i])
                try:
                    self._reco_energy += list(Ereco)
                except TypeError:
                    self._reco_energy += [Ereco]
            else:
                Ereco = self.detector.energy_resolution.sample_energy(
                    (ra_d[i], dec_d[i]), np.log10(Earr_d[i], seed=seed)
                )
                try:
                    self._reco_energy += list(np.power(10,Ereco))
                except TypeError:
                    self._reco_energy += [np.power(10, Ereco)]

        self._arrival_energy = np.concatenate(tuple(Earr_d[k] for k in Earr_d.keys()))

        return self._arrival_energy, self._reco_energy

    # @profile
    def run(self, N=None, seed=1234, show_progress=True):
        """
        Run a simulation for the given set of sources
        and detector configuration.
        The expected number of neutrinos will be
        calculated for each source. If total N is forced,
        then the number from each source will be weighted
        accordingly.
        :param N: Set expected number of neutrinos manually.
        :param seed: Set random seed.
        """

        #if show_progress:
        #    logger.basicConfig(level=logging.INFO)
        #
        #else:
        #    logger.basicConfig(level=logging.CRITICAL)

        np.random.seed(seed)

        self._get_expected_number()

        if not N:

            self.N = np.random.poisson(sum(self._Nex))
            logger.info("Random N.")

        else:

            self.N = int(N)
            logger.info("N provided.")

        v_min = - self.max_cosz
        v_max = - self.min_cosz

        self._true_energy = []
        self._arrival_energy = []
        self._reco_energy = []
        self.coordinate = []
        self._ra = []
        self._dec = []
        self._ang_err = []
        self._source_label = []
        #During detector simulation, many events are discarded due to 
        #the detection probability (scaled version of the effective area).
        #To increase speed, take many more events at once and only keep
        #the accepted ones.

        #TODO maybe change the factor to something spectral index dependent
        num = self.N * 5000
        label = np.random.choice(range(len(self.sources)), self.N, p=self._source_weights)
        l_set = set(label)
        l_num = {i: np.argwhere(i == label).shape[0] for i in l_set}
        max_energy = {}
        #create dicts of empty arrays of matching length (i.e. expected events) for each surce
        ra_d = {i: np.zeros(l_num[i]) for i in l_set}
        dec_d = {i: np.zeros(l_num[i]) for i in l_set}
        Etrue_d = {i: np.zeros(l_num[i]) for i in l_set}
        Earr_d = {i: np.zeros(l_num[i]) for i in l_set}
        
        accepted = np.zeros(self.N, dtype=bool)
        if show_progress:
            progress = progress_bar(range(self.N), desc="Sampling", position=0, leave=True)
        #go over each source
        for i in l_set:
            #simulate until appropriate number of events is accepted
            
            max_energy[i] = self.sources[i].flux_model._upper_energy
            
            logger.info(f"Simulating source {i}")
            while True:
                #check if data is needed, else break loop
                where_zero = np.argwhere(Etrue_d[i] == 0.)
                if where_zero.size == 0:
                    logger.debug("no more empty slots, done")
                    break
                
                Etrue_ = self.sources[i].flux_model.sample(num)
                if self.sources[i].source_type == DIFFUSE:

                    ra_, dec_ = sphere_sample(v_min=v_min, v_max=v_max, N=num)

                else:

                    ra_, dec_ = np.full(num, self.sources[i].coord[0]), np.full(num, self.sources[i].coord[1])

                cosz = -np.sin(dec_)


                Earr_ = Etrue_ / (1 + self.sources[i].z)
                detection_prob = self.detector.effective_area.detection_probability(
                        Earr_, cosz, max_energy[i]
                ).astype(float)
                #detection_prob = 1.0

                samples = uniform.rvs(size=num, random_state=seed)
                accepted_ = samples <= detection_prob
                idx = np.nonzero(accepted_)
                
                if idx[0].size == 0:
                    continue
                else:
                    start = np.min(where_zero)
                    end = start + idx[0].size
                    try:
                        Etrue_d[i][start:end] = Etrue_[idx]
                        Earr_d[i][start:end] = Earr_[idx]
                        ra_d[i][start:end] = ra_[idx]
                        dec_d[i][start:end] = dec_[idx]
                        logger.debug("All data placed.")
                        if show_progress:
                            progress.update(idx[0].size)
                    except (IndexError, ValueError):
                        logger.debug("Not enough slots, cutting short.")
                        remaining = np.argwhere(Etrue_d[i] == 0.).size
                        Etrue_d[i][start:] = Etrue_[idx][0:remaining]
                        Earr_d[i][start:] = Earr_[idx][0:remaining]
                        ra_d[i][start:] = ra_[idx][0:remaining]
                        dec_d[i][start:] = dec_[idx][0:remaining]
                        if show_progress:
                            progress.update(remaining)
                        break
            # print("Sampling spectrum is done")
            logger.info("Done sampling the spectrum") 
            if not isinstance(self.detector.energy_resolution, R2021IRF):
                logger.info("Sampled reco energy")
                Ereco = self.detector.energy_resolution.sample(Earr_d[i])
                #this and all following try-except TypeError blocks
                #are needed if only a single event is sampled
                #then list() will not work bc float is not an iterable
                try:
                    self._reco_energy += list(Ereco)           
                except TypeError:
                    self._reco_energy += [Ereco]

            #do source type specific things here
            if self.sources[i].source_type == DIFFUSE:

                logger.info("Sampling angular uncertainty for diffuse source")
                if isinstance(self.detector.angular_resolution, R2021IRF):
                    _, _, reco_ang_err, Ereco = self.detector.angular_resolution.sample(
                        (ra_d[i], dec_d[i]), np.log10(Earr_d[i]), seed=seed
                    )
                    try:
                        self._ang_err += list(reco_ang_err)
                        self._reco_energy += list(Ereco)
                    except TypeError as e:
                        print(e)
                        self._ang_err += [reco_ang_err]
                        self._reco_energy += [Ereco]
                        
                elif isinstance(self.detector.angular_resolution, AngularResolution):
                    reco_ang_err = self.detector.angular_resolution.get_ret_ang_err(
                        Earr_d[i]
                    )
                    try:
                        self._ang_err += list(reco_ang_err)
                    except TypeError:
                        self._ang_err += [reco_ang_err]
                elif isinstance(
                    self.detector.angular_resolution, FixedAngularResolution
                ):
                    reco_ang_err = self.detector.angular_resolution.ret_ang_err
                    #TODO fix
                    temp = [reco_ang_err] * l_num[i]

                try:
                    self._dec += list(dec_d[i])
                    self._ra += list(ra_d[i])
                except TypeError:
                    self._dec += [dec_d[i]]
                    self._ra += [ra_d[i]]
                logger.info("Sampled angular uncertainty for diffuse source")

            else:

                logger.info("Sampling angular uncertainty for point source")
                if isinstance(self.detector.angular_resolution, R2021IRF):
                    #loop over events handled inside R2021IRF
                    reco_ra, reco_dec, reco_ang_err, Ereco  = self.detector.angular_resolution.sample(
                        (ra_d[i], dec_d[i]), np.log10(Earr_d[i]), seed=seed)
                    self._reco_energy += list(Ereco)

                elif isinstance(self.detector.angular_resolution, AngularResolution):
                    reco_ra, reco_dec = self.detector.angular_resolution.sample(
                        Earr_d[i], (ra_d[i], dec_d[i])
                    )
                    reco_ang_err = self.detector.angular_resolution.ret_ang_err

                elif isinstance(
                    self.detector.angular_resolution, FixedAngularResolution
                ):
                    reco_ra, reco_dec = self.detector.angular_resolution.sample(
                        (ra_d[i], dec_d[i])
                    )
                    reco_ang_err = self.detector.angular_resolution.ret_ang_err
                try:
                    self._ang_err +=list(reco_ang_err)
                    self._dec += list(reco_dec)
                    self._ra += list(reco_ra)
                except TypeError as e:
                    print(e)
                    self._ang_err += [reco_ang_err]
                    self._dec += [reco_dec]
                    self._ra += [reco_ra]
                logger.info("Sampled angular uncertainty for point source")

        logger.info("Creating array of simulation data")  
        self._true_energy = np.concatenate(tuple(Etrue_d[k] for k in Earr_d.keys()))
        self._arrival_energy = np.concatenate(tuple(Earr_d[k] for k in Earr_d.keys()))
        self._source_label = np.concatenate(tuple(np.full(l_num[l], l, dtype=int) for l in Earr_d.keys()))
        if show_progress:
            progress.close()
        logger.info("Created array of simulation data")

        self._ra = {self._period: self._ra}
        self._dec = {self._period: self._dec}
        self._true_energy = {self._period: self._true_energy}
        self._reco_energy = {self._period: self._reco_energy}
        self._arrival_energy = {self._period: self._arrival_energy}
        self._source_label = {self._period: self._source_label}
        self._ang_err = {self._period: self._ang_err}
        
 

    def save(self, filename):
        """
        Save the output to filename, for all but the R2021 release.
        """

        self._filename = filename

        with h5py.File(filename, "w") as f:

            f.create_dataset("true_energy", data=self.true_energy)

            f.create_dataset("arrival_energy", data=self.arrival_energy)

            f.create_dataset("reco_energy", data=self.reco_energy)

            f.create_dataset("ra", data=self.ra)

            f.create_dataset("dec", data=self.dec)

            f.create_dataset("ang_err", data=self.ang_err)

            f.create_dataset("source_label", data=self.source_label)

            for i, source in enumerate(self.sources):

                s = f.create_group("source_" + str(i))

                if isinstance(source.flux_model, PowerLawFlux):

                    s.create_dataset("index", data=source.flux_model._index)

                    s.create_dataset(
                        "normalisation_energy",
                        data=source.flux_model._normalisation_energy,
                    )

                elif isinstance(source.flux_model, BrokenPowerLawFlux):

                    s.create_dataset("index1", data=source.flux_model._index1)

                    s.create_dataset("index2", data=source.flux_model._index2)

                    s.create_dataset(
                        "break_energy", data=source.flux_model._break_energy
                    )

                s.create_dataset("source_type", data=source.source_type)

                s.create_dataset("normalisation", data=source.flux_model._normalisation)



class Braun2008Simulator:
    """
    Simple simulator which uses the results
    given in Braun+2008.

    Takes a fixed number of neutrinos as arguments
    rather than source models to reflect the method
    in the paper.
    """

    def __init__(self, source, effective_area, reco_energy_sampler, angular_resolution):
        """
        Simple simulator which uses the results
        given in Braun+2008.

        Takes a fixed number of neutrinos as arguments
        rather than source models to reflect the method
        in the paper.

        :param source: Instance of Source.
        :param effective_area: Instance of EffectiveArea
        :param reco_energy_sampler: Instance of RecoEnergySampler
        """

        self.source = source

        self.effective_area = effective_area

        self.reco_energy_sampler = reco_energy_sampler

        # Hard code to match Braun+2008
        self.angular_resolution = angular_resolution
        self.max_cosz = 0.1
        self.min_cosz = -1.
        self.reco_energy_index = 3.8

    def run(self, N, show_progress=True):
        """
        Run the simulation.

        :param N: Number of events to simulate.
        """

        self.N = N

        self._true_energy = []
        self._reco_energy = []
        self.coordinate = []
        self._ra = []
        self._dec = []
        self._source_label = []

        self.reco_energy_sampler.set_index(self.reco_energy_index)

        v_min = - self.max_cosz
        v_max = - self.min_cosz

        max_energy = self.source.flux_model._upper_energy

        for i in progress_bar(
            range(self.N), desc="Sampling", disable=(not show_progress)
        ):

            accepted = False

            while not accepted:

                Etrue = self.source.flux_model.sample(1)[0]

                if self.source.source_type == DIFFUSE:

                    ra, dec = sphere_sample(v_min=v_min, v_max=v_max)

                else:

                    ra, dec = self.source.coord

                cosz = -np.sin(dec)

                detection_prob = float(
                    self.effective_area.detection_probability(Etrue, cosz, max_energy)
                )

                accepted = np.random.choice(
                    [True, False], p=[detection_prob, 1 - detection_prob]
                )

            self._true_energy.append(Etrue)

            Ereco = self.reco_energy_sampler()
            self._reco_energy.append(Ereco)

            if self.source.source_type == DIFFUSE:

                self.coordinate.append(SkyCoord(ra * u.rad, dec * u.rad, frame="icrs"))
                self._ra.append(ra)
                self._dec.append(dec)

            else:

                if isinstance(self.angular_resolution, AngularResolution):
                    reco_ra, reco_dec = self.angular_resolution.sample(Etrue, (ra, dec))

                elif isinstance(self.angular_resolution, FixedAngularResolution):
                    reco_ra, reco_dec = self.angular_resolution.sample((ra, dec))

                self.coordinate.append(
                    SkyCoord(reco_ra * u.rad, reco_dec * u.rad, frame="icrs")
                )
                self._ra.append(reco_ra)
                self._dec.append(reco_dec)

    '''
    def save(self, filename):
        """
        Save the output to filename.
        """

        self._filename = filename

        with h5py.File(filename, "w") as f:

            f.create_dataset("true_energy", data=self._true_energy)

            f.create_dataset("reco_energy", data=self._reco_energy)

            f.create_dataset("ra", data=self._ra)

            f.create_dataset("dec", data=self._dec)

            f.create_dataset("index", data=self.source.flux_model._index)

            f.create_dataset("source_type", data=self.source.source_type)

            f.create_dataset(
                "normalisation", data=self.source.flux_model._normalisation
            )

            f.create_dataset(
                "normalisation_energy",
                data=self.source.flux_model._normalisation_energy,
            )
    '''

class TimeDependentSimulator(SimEvents):
    """
    Simulator-class for simulations spanning multiple data taking periods.
    """

    _available_periods = ["IC40", "IC59", "IC79", "IC86_I", "IC86_II"]

    _time_limits = {}

    # need to find time limits of data taking periods
    # s.t. starting point (in years, days, whatever)
    # can be defined, endpoint as well
    # class instance then calculates the according times
    # for each period

    def __init__(self, periods, sources, **kwargs):
        """Instanciates multi-period simulator.
        :param periods: Tuple of periods to be included in simulation.
        :param sources: List of sources to be simulated.
        :param kwargs: Dict with further settings.
        """
        super().__init__()
        self.simulators = {}
        if not all(_ in self._available_periods for _ in periods):
            raise ValueError("Some periods not supported.")

        #Get time dependent detector.
        time_dependent_detector = TimeDependentIceCube.from_periods(*periods)
        self.simulators = {
            p: Simulator(sources, sim, p) for p, sim in time_dependent_detector.yield_detectors()
        }
        self.sources = sources
        self._periods = periods

        if kwargs.get("time"):
            self.time = kwargs["time"]
        else:
            logger.warning("Need to set simulation times, defaults to 1 year each.")

    def run(self, N: List=None, seed=1234, show_progress=False):
        """
        Runs simulation for each period.
        :param N: List of Ns to be set as expected number of neutrinos in sample.
        :param seed: Random seed.
        :param show_progress: Bool, True if debugging information on simulation is to be shown.
        Currently not implemented.
        """

        for p, sim in self.simulators.items():
            logger.info(f"Simulating period {p}.")
            sim.run(N=None, seed=1234, show_progress=show_progress)
            self._true_energy[sim._period] = sim.true_energy[p]
            self._arrival_energy[sim._period] = sim.arrival_energy[p]
            self._reco_energy[sim._period] = sim.reco_energy[p]
            self._ra[sim._period] = sim.ra[p]
            self._dec[sim._period] = sim.dec[p]
            self._ang_err[sim._period] = sim.ang_err[p]
            self._source_label[sim._period] = sim.source_label[p]


    def save(self, path, file_prefix):
        """
        Saves simulated data sets.
        :param path: Path to directory in which files should be saved.
        :param file_prefix: Actually suffix of filename, to be appended to all files.
        :return: Dictionary of filenames.
        """

        d = {}
        for p, sim in self.simulators.items():
            file_name = join(path, f"p_{p}_{file_prefix}.h5")
            d[p] = file_name
            sim.save(file_name)
        return d


    def get_expected_number(self):
        #loop over all periods and call _get_expected_number() with properly chosen time
        # TODO add way of extracting these numbers
        return {p: sim._get_expected_number() for p, sim in self.simulators.items()}
        #for sim in self.simulators.values():
        #    sim._get_expected_number()

    """
    def _yield_time(self):
        for p, sim in self.simulators.items():
            yield sim.time
    """

    @property
    def time(self):
        """
        Returns dictionary of simulator times.
        :return: Dictionary of simulator times.
        """

        return {p: sim.time for p, sim in self.simulators.items()}


    @time.setter
    def time(self, times: Dict):
        """
        Sets simulator times.
        :param times: Dict returned by Uptime.find_time_obs().
        """
        #TODO rewrite method
        for p, t in times.items():
            self.simulators[p].time = t
        logger.info("Set simulation times")
            

    @property
    def sources(self):
        return self._sources
    

    @sources.setter
    def sources(self, source_list):
        self._sources = source_list
        for sim in self.simulators.values():
            sim._sources = source_list


    
def sphere_sample(radius=1, v_min=-1, v_max=1, N=1):
    """
    Sample points uniformly on a sphere.
    """

    u = np.random.uniform(0, 1, size=N)
    v = np.random.uniform(v_min, v_max, size=N)

    phi = 2 * np.pi * u
    theta = np.arccos(v)

    ra, dec = spherical_to_icrs(theta, phi)

    return ra, dec


def spherical_to_icrs(theta, phi):
    """
    convert spherical coordinates to ICRS
    ra, dec.
    """

    ra = phi

    dec = np.pi / 2 - theta

    return ra, dec


def lists_to_tuple(list1, list2):

    return [(list1[i], list2[i]) for i in range(0, len(list1))]
