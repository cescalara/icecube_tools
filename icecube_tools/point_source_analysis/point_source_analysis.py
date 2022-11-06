from ..point_source_likelihood.point_source_likelihood import (
    PointSourceLikelihood, TimeDependentPointSourceLikelihood
)
from ..utils.data import data_directory, available_periods, ddict
from ..utils.coordinate_transforms import *

import yaml

import healpy as hp
import numpy as np

from abc import ABC, abstractmethod

from os.path import join
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


"""
Classes to perform reproducible point source analyses
"""

class PointSourceAnalysis(ABC):
    """Meta class"""

    @abstractmethod
    def load_config(self):
        """Load yaml analysis config"""
        pass


    @abstractmethod
    def write_config(self):
        """Write yaml analysis config"""
        pass


    @abstractmethod
    def apply_cuts(self):
        """Make cuts on loaded data"""
        pass


    def add_events(self, *periods):
        """Add events for multiple data seasons of a single IRF, i.e. only IC86_II and up"""
        events = []
        self.num_of_irf_periods = 0
        for p in periods:
            if p in available_periods:
                self.num_of_irf_periods += 1
            events.append(np.loadtxt(join(data_directory, f"20210126_PS-IC40-IC86_VII/icecube_10year_ps/events/{p}_exp.csv")))
        return np.concatenate(tuple(events))


    def load_events(self, *periods):
        add = []
        for p in periods:
            if p in ["IC86_II", "IC86_III", "IC86_IV", "IC86_V", "IC86_VI", "IC86_VII"]:
                continue
            else:
                yield np.loadtxt(np.join(data_directory, f"20210126_PS-IC40-IC86_VII/icecube_10year_ps/events/{p}_exp.csv"))
        if add:
            yield self.add_events(*add)


class MapScan(PointSourceAnalysis):

    #Config structure for yaml files
    config_structure = {
        "sources":
            {"nside": int, "npix": int, "ras": list, "decs": list}, 
        "data": {
            "periods": list,
            "cuts": {"northern": {"emin": float}, "equator": {"emin": float}, "southern": {"emin": float}
            }
        },
    }


    def __init__(self, path):
        self.load_config(path)
        self._make_output_arrays()
        self.perform_scan()


    def perform_scan(self):
        for c, (ra, dec) in enumerate(zip(self.test_ra, self.test_dec)):
            self.test_source((ra, dec), c)


    def test_source(self, source_coord, num):
        if source_coord[1] <= np.deg2rad(10):
            likelihood = TimeDependentPointSourceLikelihood(
                source_coord,
                self.periods,

            )
            if likelihood.N > 0:    # else somewhere division by zero
                self.tsts[num] = likelihood.get_test_statistic()
                self.index[num] = likelihood._best_fit_index
                self.ns[num] = likelihood._best_fit_ns
                self.index_err[num] = likelihood._m.errors["index"]
                self.ns_err[num] = likelihood._m.errors["ns"]




    def load_config(self, path):
        """
        Load analysis config from file
        """

        with open(path, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        print(config)
        self.config = config
        source_config = config.get("sources")
        self.nside = source_config.get("nside")
        self.npix = source_config.get("npix")
        data_config = config.get("data")
        self.periods = data_config.get("periods")
        self.northern_emin = data_config.get("cuts").get("northern").get("emin")
        self.equator_emin = data_config.get("cuts").get("equator").get("emin")
        self.southern_emin = data_config.get("cuts").get("southern").get("emin")


    def write_config(self, path):
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

        config.add(self.periods, "data", "periods")
        config.add(self.northern_emin, "data", "cuts", "northern", "emin")
        config.add(self.equator_emin, "data", "cuts", "equator", "emin")
        config.add(self.southern_emin, "data", "cuts", "southern", "emin")

        with open("new_config.yaml", "w") as f:
            yaml.dump(config, f)


    def generate_sources(self, nside=True):
        """
        Generate sources from config-specified specifics
        """

        reload = True
        if self.nside is not None and nside:
            self.npix = hp.nside2npix(self.nside)
            print("Overwriting npix with nside = {}".format(self.nside))
        elif self.npix is not None and not nside:
            print("Using npix = {}".format(self.npix))
        elif self.ra_test is not None and self.dec_test is not None:
            print("Using provided ra and dec")
            reload = False

        if reload:
            print(f"resolution in degrees: {hp.nside2resol(self.nside, arcmin=True)/60}")
            theta_test, phi_test = hp.pix2ang(self.nside, np.arange(self.npix), nest=False)
            ra_test, dec_test = spherical_to_icrs(theta_test, phi_test)
            self.ra_test = ra_test
            self.dec_test = dec_test

    
    def _make_output_arrays(self):
        self.ts = np.zeros(self.npix)
        self.index = np.zeros(self.npix)
        self.ns = np.zeros((self.npix, self.num_of_irf_periods))
        self.ns_error = np.zeros((self.npix, self.num_of_irf_periods))
        self.index_error = np.zeros(self.npix)


    def apply_cuts(self):
        pass