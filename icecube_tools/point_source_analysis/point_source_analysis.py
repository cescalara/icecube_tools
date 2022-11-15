from ..point_source_likelihood.point_source_likelihood import (
    PointSourceLikelihood, TimeDependentPointSourceLikelihood
)
from ..point_source_likelihood.energy_likelihood import (
    MarginalisedIntegratedEnergyLikelihood, MarginalisedEnergyLikelihood
)

from ..detector.r2021 import R2021IRF
from ..detector.effective_area import EffectiveArea
from ..utils.data import data_directory, available_periods, ddict, Events, Uptime
from ..utils.coordinate_transforms import *

import yaml
import h5py
import healpy as hp
import numpy as np
from tqdm import tqdm as progress_bar

from abc import ABC, abstractmethod

from os.path import join
import os.path
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


"""
Classes to perform reproducible point source analyses
"""

class PointSourceAnalysis(ABC):
    """Meta class"""

    def __init__(self):
        pass


    @abstractmethod
    def load_config(self):
        pass


    @abstractmethod
    def write_config(self):
        pass


    @abstractmethod
    def write_output(self):
        pass


    @abstractmethod
    def apply_cuts(self):
        """Make cuts on loaded data"""
        pass


    @property
    def which(self):
        return self._which



class MapScan(PointSourceAnalysis):

    #Config structure for yaml files
    config_structure = {
        "sources":
            {"nside": int, "npix": int, "ras": list, "decs": list}, 
        "data": {
            "periods": list,
            "cuts": {"northern": {"emin": float}, "equator": {"emin": float}, "southern": {"emin": float},
            "min_dec": float, "max_dec": float
            },
            "likelihood": str
        },
    }


    def __init__(self, path: str, events: Events):
        """
        Instantiate analysis object.
        :param path: Path to config
        :param events: object inheriting from :class:`icecube_tools.utils.data.Events`
        :param energy_likelihood: Dict of energy_likelihoods,
            key: period, value: :class:`icecube_tools.point_source_likelihood.energy_likelihood.MarginalisedEnergyLikelihood`
        """

        self.events = events
        self.num_of_irf_periods = 0
        is_86 = False
        for p in events.periods:
            #if "86" in p and and not is_86:
            if p in ["IC86_II", "IC86_III", "IC86_IV", "IC86_V", "IC86_VI", "IC86_VII"] and not is_86:
                self.num_of_irf_periods += 1
                is_86 = True
            else:
                self.num_of_irf_periods += 1
        self.uptime = Uptime()
        self.times = {p: self.uptime.time_obs(p).value for p in self.events.periods}
       

        self.load_config(path)
        self.apply_cuts()
        self.energy_likelihood = {}
        for p in self.events.periods:
            aeff = EffectiveArea.from_dataset("20210126", p)
            irf = R2021IRF.from_period(p)
            self.energy_likelihood[p] = MarginalisedIntegratedEnergyLikelihood(
                irf,
                aeff,
                np.linspace(2, 9, num=25)
            )
        #self._make_output_arrays()


    def perform_scan(self, show_progress=False, minos=False):
        logger.info("Performing scan for periods: {}".format(self.events.periods))
        ra = self.events.ra
        dec = self.events.dec
        ang_err = self.events.ang_err
        reco_energy = self.events.reco_energy
        if show_progress:
            for c in progress_bar(range(len(self.ra_test))):
                self._test_source((self.ra_test[c], self.dec_test[c]), c, ra, dec, reco_energy, ang_err)
        else:
            for c, (ra_t, dec_t) in enumerate(zip(self.ra_test, self.dec_test)):
                self._test_source((ra_t, dec_t), c, ra, dec, reco_energy, ang_err, minos)



    def _test_source(self, source_coord, num, ra, dec, reco_energy, ang_err, minos=False):
        if source_coord[1] <= np.deg2rad(90):    #delete this...
            likelihood = TimeDependentPointSourceLikelihood(
                source_coord,
                self.events.periods,
                ra,
                dec,
                reco_energy,
                ang_err,
                self.energy_likelihood,
                which=self.which,
                times=self.times
            )
            if likelihood.Nprime > 0:    # else somewhere division by zero
                logging.info("Nearby events: {}".format(likelihood.Nprime))
                self.ts[num] = likelihood.get_test_statistic()
                self.index[num] = likelihood._best_fit_index
                self.ns[num] = likelihood._best_fit_ns
                self.index_error[num] = likelihood.m.errors["index"]
                self.ns_error[num] = np.array(
                    [likelihood.m.errors[n] for n in likelihood.m.parameters if n != "index"]
                )
                self.fit_ok[num] = likelihood.m.fmin.is_valid
                
                # is computationally too expensive for the entire grid, only use at certain points!
                if self.fit_ok[num] and minos:
                    minos = likelihood.m.minos()
                    if minos.valid:
                        self.index_merror[num, 0] = minos.merrors["index"].lower
                        self.index_merror[num, 1] = minos.merrors["index"].upper
                        self.ns_merror[num, 0] = minos.merrors["ns"].lower
                        self.ns_merror[num, 1] = minos.merrors["ns"].upper
                


    def load_config(self, path):
        """
        Load analysis config from file
        """

        with open(path, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        logger.debug("{}".format(str(config)))    # ?!
        self.config = config
        source_config = config.get("sources", False)
        if source_config:
            self.nside = source_config.get("nside")
            self.npix = source_config.get("npix")
            self.ra_test = source_config.get("ras")
            self.dec_test = source_config.get("decs")
        data_config = config.get("data")
        self.periods = data_config.get("periods")
        cuts = data_config.get("cuts", False)
        if cuts:
            self.northern_emin = float(data_config.get("cuts").get("northern").get("emin", 1e1))
            self.equator_emin = float(data_config.get("cuts").get("equator").get("emin", 1e1))
            self.southern_emin = float(data_config.get("cuts").get("southern").get("emin", 1e1))
            self.min_dec = data_config.get("cuts").get("min_dec", -90)
            self.max_dec = data_config.get("cuts").get("max_dec", 90)
        else:
            self.northern_emin = 1e1
            self.equator_emin = 1e1
            self.southern_emin = 1e1
            self.min_dec = -90.
            self.max_dec = 90.
        self._which = data_config.get("likelihood", "both")


    def write_config(self, path, source_list=False):
        """
        Write config used in analysis to file
        """

        config = ddict()
        try:
            config.add(self.nside, "sources", "nside")
        except AttributeError:
            pass
        try:
            config.add(self.npix, "sources", "npix")
        except AttributeError:
            pass
        if source_list:
            config.add(self.ra_test, "sources", "ra")
            config.add(self.dec_test, "sources", "dec")
        config.add(self.periods, "data", "periods")
        for emin, region in zip([self.northern_emin, self.equator_emin, self.southern_emin],
            ["northern", "equator", "southern"]
        ):
            try:
                config.add(emin, "data", "cuts", region, "emin")
            except AttributeError:
                pass
        try:
            config.add(self.min_dec, "data", "cuts", "min_dec")
        except AttributeError:
            config.add(-90, "data", "cuts", "min_dec")
        try:
            config.add(self.max_dec, "data", "cuts", "max_dec")
        except AttributeError:
            config.add(90, "data", "cuts", "min_dec")
        config.add(self.which, "data", "likelihood")

        with open(path, "w") as f:
            yaml.dump(config, f)


    def write_output(self, path: str):
        """
        Save analysis results to hdf5 and additionally the used config, path is saved in results hdf5.
        """

        try:
            self.ts
            assert np.any(self.ts)
        except (AttributeError, AssertionError):
            logging.error("Call perform_scan() first")
            return

        self.write_config(os.path.splitext(path)[0]+".yaml")
            
        with h5py.File(path, "w") as f:
            meta = f.create_group("meta")
            meta.create_dataset("ra", shape=self.ra_test.shape, data=self.ra_test)
            meta.create_dataset("dec", shape=self.dec_test.shape, data=self.dec_test)
            meta.create_dataset("periods", data=self.periods)
            meta.attrs["config_path"] = os.path.splitext(path)[0]+".yaml"
        
            data = f.create_group("output")
            data.create_dataset("ts", shape=self.ts.shape, data=self.ts)
            data.create_dataset("index", shape=self.index.shape, data=self.index)
            data.create_dataset("ns", shape=self.ns.shape, data=self.ns)
            data.create_dataset("ns_error", shape=self.ns_error.shape, data=self.ns_error)
            data.create_dataset("index_error", shape=self.index_error.shape, data=self.index_error)
            data.create_dataset("ns_merror", shape=self.ns_merror.shape, data=self.ns_merror)
            data.create_dataset("index_merror", shape=self.index_merror.shape, data=self.index_merror)


    def generate_sources(self, nside=True):
        """
        Generate sources from config-specified specifics
        """

        reload = True
        if self.nside is not None and nside:
            self.npix = hp.nside2npix(self.nside)
            logger.warning("Overwriting npix with nside = {}".format(self.nside))
        elif self.npix is not None and not nside:
            logger.info("Using npix = {}".format(self.npix))
        elif self.ra_test is not None and self.dec_test is not None:
            logger.info("Using provided ra and dec")
            assert len(self.ra_test) == len(self.dec_test)
            reload = False

        if reload:
            logger.info(f"resolution in degrees: {hp.nside2resol(self.nside, arcmin=True)/60}")
            theta_test, phi_test = hp.pix2ang(self.nside, np.arange(self.npix), nest=False)
            ra_test, dec_test = spherical_to_icrs(theta_test, phi_test)
            self.ra_test = ra_test[
                np.nonzero((dec_test <= np.deg2rad(self.max_dec)) & (dec_test >= np.deg2rad(self.min_dec)))
            ]
            self.dec_test = dec_test[
                np.nonzero((dec_test <= np.deg2rad(self.max_dec)) & (dec_test >= np.deg2rad(self.min_dec)))
            ]
        self._make_output_arrays()

    
    def _make_output_arrays(self):
        if self.npix is not None:
            num = self.npix
        elif self.ra_test is not None and self.dec_test is not None:
            num = len(self.ra_test)
        else:
            raise ValueError("Can't create output arrays, no well-defined source list supplied.")

        self.ts = np.zeros(num)
        self.index = np.zeros(num)
        self.ns = np.zeros(num)
        self.ns_error = np.zeros(num)
        self.index_error = np.zeros(num)
        self.fit_ok = np.zeros(num, dtype=bool)
        self.index_merror = np.zeros((num, 2))                          #for asymmetric minos errors
        self.ns_merror = np.zeros((num, 2))    #for asymmetric minos errors


    def apply_cuts(self):
        #make cuts based on config
        #incudes right now: energy only
        mask = {}
        self.events.mask = None
        try:
            for p in self.periods:
                events = self.events.period(p)
                mask[p] = np.nonzero(
                    ((events["reco_energy"] > self.northern_emin) & (events["dec"] >= np.deg2rad(10))) |
                    ((events["reco_energy"] > self.equator_emin) & (events["dec"] < np.deg2rad(10)) & 
                        (events["dec"] > np.deg2rad(-10))) |
                    ((events["reco_energy"] > self.southern_emin) & (events["dec"] <= np.deg2rad(-10)))
                )
            self.events.mask = mask
        except AttributeError:
            pass
        