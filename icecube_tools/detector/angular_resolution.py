import numpy as np
from abc import ABC, abstractmethod
from vMF import sample_vMF
from astropy.coordinates import SkyCoord
from astropy import units as u

from icecube_tools.utils.data import IceCubeData, find_files

"""
Module for handling the angular resolution
of IceCube based on public data.
"""

R2018_ANG_RES_FILENAME = "AngRes.txt"
R2015_ANG_RES_FILENAME = "angres_plot"

TRUE_ENERGY = 0
RECO_ENERGY = 1

_supported_dataset_ids = ["20181018"]


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

        self.reco_energy_values = None

        self.read()

        super().__init__()

    @abstractmethod
    def read(self):

        pass


class R2015AngResReader(IceCubeAngResReader):
    """
    Reader for the 2015 release.
    """

    def read(self):

        out = np.loadtxt(self._filename, delimiter=",", comments="#")

        self.ang_res_values = out.T[1]

        self.reco_energy_values = out.T[0]


class FromPlotAngResReader(IceCubeAngResReader):
    """
    Reader for the plots from the Aartsen+2018
    point source analysis paper.
    """

    def read(self):

        out = np.loadtxt(self._filename, delimiter=",", comments="#")

        self.ang_res_values = out.T[1]

        self.true_energy_values = out.T[0]

        self.true_energy_bins = None


class R2018AngResReader(IceCubeAngResReader):
    """
    Reader for the 2018 Oct 18 release.
    Link: https://icecube.wisc.edu/science/data/PS-3years.
    """

    def read(self):

        import pandas as pd

        self.year = int(self._filename[-15:-11])
        self.nu_type = "nu_mu"

        filelayout = ["E_min [GeV]", "E_max [GeV]", "Med. Resolution[deg]"]
        output = pd.read_csv(
            self._filename, comment="#", delim_whitespace=True, names=filelayout
        ).to_dict()

        true_energy_lower = set(output["E_min [GeV]"].values())
        true_energy_upper = set(output["E_max [GeV]"].values())

        self.true_energy_bins = np.array(
            list(true_energy_upper.union(true_energy_lower))
        )
        self.true_energy_bins.sort()

        self.ang_res_values = np.array(list(output["Med. Resolution[deg]"].values()))

        self.true_energy_values = (
            self.true_energy_bins[0:-1] + np.diff(self.true_energy_bins) / 2
        )


class AngularResolution:
    def __init__(self, filename, offset=0):

        self._filename = filename

        self._reader = self.get_reader()

        self.values = self._reader.ang_res_values + offset

        if self._energy_type == TRUE_ENERGY:

            self.true_energy_bins = self._reader.true_energy_bins

            self.true_energy_values = self._reader.true_energy_values

        elif self._energy_type == RECO_ENERGY:

            self.reco_energy_values = self._reader.reco_energy_values

        self.sigma = None

    def get_reader(self):
        """
        Define an IceCubeAeffReader based on the filename.
        """

        if R2018_ANG_RES_FILENAME in self._filename:

            self._energy_type = TRUE_ENERGY

            return R2018AngResReader(self._filename)

        elif R2015_ANG_RES_FILENAME in self._filename:

            self._energy_type = RECO_ENERGY

            return R2015AngResReader(self._filename)

        elif ".csv" in self._filename:

            self._energy_type = TRUE_ENERGY

            return FromPlotAngResReader(self._filename)

        else:

            raise ValueError(
                self._filename
                + " is not recognised as one of the known angular resolution files."
            )

    def _get_angular_resolution(self, E):
        """
        Get the median angular resolution for the
        given Etrue/Ereco.
        """

        if self._energy_type == TRUE_ENERGY:

            true_energy_bin_cen = (
                self.true_energy_bins[:-1] + self.true_energy_bins[1:]
            ) / 2

            ang_res = np.interp(np.log(E), np.log(true_energy_bin_cen), self.values)

        elif self._energy_type == RECO_ENERGY:

            ang_res = np.interp(np.log(E), np.log(self.reco_energy_values), self.values)

        return ang_res

    def sample(self, Etrue, coord):
        """
        Sample new ra, dec values given a true energy
        and direction.
        """

        ra, dec = coord

        sigma = self._get_angular_resolution(Etrue)

        sky_coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs")

        sky_coord.representation_type = "cartesian"

        unit_vector = np.array([sky_coord.x, sky_coord.y, sky_coord.z])

        kappa = 5000 * np.power(sigma, -2)

        new_unit_vector = sample_vMF(unit_vector, kappa, 1)[0]

        new_sky_coord = SkyCoord(
            x=new_unit_vector[0],
            y=new_unit_vector[1],
            z=new_unit_vector[2],
            representation_type="cartesian",
        )

        new_sky_coord.representation_type = "unitspherical"

        new_ra = new_sky_coord.ra.rad

        new_dec = new_sky_coord.dec.rad

        return new_ra, new_dec

    @classmethod
    def from_dataset(cls, dataset_id):
        """
        Load angular resolution from publicly
        available data.
        """

        data_interface = IceCubeData()

        if dataset_id not in _supported_dataset_ids:

            raise NotImplementedError("This dataset is not currently supported")

        dataset = data_interface.find(dataset_id)

        data_interface.fetch(dataset)

        dataset_dir = data_interface.get_path_to(dataset[0])

        if dataset_id == "20181018":

            files = find_files(dataset_dir, R2018_ANG_RES_FILENAME)

            angres_file_name = files[2]

        return cls(angres_file_name)


class FixedAngularResolution:
    def __init__(self, sigma=1.0):
        """
        Simple fixed angular resolution.

        :param sigma: Resolution [deg].
        """

        self.sigma = sigma

    def sample(self, coord):
        """
        Sample reconstructed coord given original position.

        :coord: ra, dec in [rad].
        """

        ra, dec = coord

        sky_coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs")

        sky_coord.representation_type = "cartesian"

        unit_vector = np.array([sky_coord.x, sky_coord.y, sky_coord.z])

        kappa = 5000 * np.power(self.sigma, -2)

        new_unit_vector = sample_vMF(unit_vector, kappa, 1)[0]

        new_sky_coord = SkyCoord(
            x=new_unit_vector[0],
            y=new_unit_vector[1],
            z=new_unit_vector[2],
            representation_type="cartesian",
        )

        new_sky_coord.representation_type = "unitspherical"

        new_ra = new_sky_coord.ra.rad

        new_dec = new_sky_coord.dec.rad

        return new_ra, new_dec


def icrs_to_unit_vector(ra, dec):
    """
    Convert to unit vector.
    """

    theta = np.pi / 2 - dec
    phi = ra

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.array([x, y, z])


def unit_vector_to_icrs(unit_vector):
    """
    Convert to ra, dec.
    """

    x = unit_vector[0]
    y = unit_vector[1]
    z = unit_vector[2]

    phi = np.arctan(y / x)
    theta = np.arccos(z)

    ra = phi
    dec = np.pi / 2 - theta

    return ra, dec
