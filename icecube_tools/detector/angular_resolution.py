import numpy as np
from abc import ABC, abstractmethod
from vMF import sample_vMF
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy import stats
from scipy.spatial.transform import Rotation as R
from icecube_tools.utils.data import IceCubeData, find_files, data_directory
from icecube_tools.utils.vMF import get_kappa, get_theta_p

"""
Module for handling the angular resolution
of IceCube based on public data.
"""

R2018_ANG_RES_FILENAME = "AngRes.txt"
R2015_ANG_RES_FILENAME = "angres_plot"
R2021_ANG_RES_FILENAME = "IC86_II_smearing.csv"




TRUE_ENERGY = 0
RECO_ENERGY = 1

_supported_dataset_ids = ["20181018", "20210126"]


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

        self.prob_contained = None

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

        self.prob_contained = 0.68


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

        self.prob_contained = 0.68


class R2018AngResReader(IceCubeAngResReader):
    """
    Reader for the 2018 Oct 18 release.
    Link: https://icecube.wisc.edu/science/data/PS-3years.
    """

    def read(self):

        self.prob_contained = 0.68

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


class R2021AngResReader(IceCubeAngResReader):
    """
    Reader for the 2021 Jan 26 release.
    Link: https://icecube.wisc.edu/data-releases/2021/01/all-sky-point-source-icecube-data-years-2008-2018/
    """

    def read(self):

        self.prob_contained = 0.68

        self.year = 2012    # subject to change
        self.nu_type = "nu_mu"

        self.output = np.loadtxt(self._filename, comments="#")

        true_energy_lower = np.array(list(set(self.output[:, 0])))
        true_energy_upper = np.array(list(set(self.output[:, 1])))

        self.true_energy_bins = np.union1d(true_energy_lower, true_energy_upper)
        self.true_energy_bins.sort()

        dec_lower = np.array(list(set(self.output[:, 2])))
        dec_higher = np.array(list(set(self.output[:, 3])))

        self.declination_bins = np.radians(np.union1d(dec_lower, dec_higher))
        self.declination_bins.sort()

        self.ang_res_values = 1    # placeholder, isn't used anyway

        #is this used?
        self.true_energy_values = (
            self.true_energy_bins[0:-1] + np.diff(self.true_energy_bins) / 2
        )




class AngularResolution(object):
    """
    Generic angular resolution class.
    """

    supported_datasets = _supported_dataset_ids

    def __init__(
        self,
        filename,
        ret_ang_err_p=0.68,
        offset=0,
        scale=1,
        scatter=None,
        minimum=0.1,
        maximum=10,
    ):
        """
        Generic angular resolution class.

        :param filename: File to load from
        :param ret_ang_err: Returned angular error will conrrespond to
        the radius of a region containing this probability
        :param offset: Add an offset to the read values.
        :param scale: Add a scale factor to the read values.
        :param scatter: Add scatter around read values.
        :param minimum: Specify minimum possible resolution
        :param maxmimum: Specify maximum possible resolution
        """

        self._filename = filename

        self._reader = self.get_reader()
        #put self.values somewhere else, not needed for every child class
        self.values = (self._reader.ang_res_values + offset) * scale

        if self._energy_type == TRUE_ENERGY:

            self.true_energy_bins = self._reader.true_energy_bins

            self.true_energy_values = self._reader.true_energy_values

        elif self._energy_type == RECO_ENERGY:

            self.reco_energy_values = self._reader.reco_energy_values

        self.ang_err_p = self._reader.prob_contained

        self.ret_ang_err_p = ret_ang_err_p

        self._ret_ang_err = None

        self._scatter = scatter

        self._minimum = minimum

        self._maximum = maximum

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

        elif R2021_ANG_RES_FILENAME in self._filename:

            self._energy_type = TRUE_ENERGY

            return R2021AngResReader(self._filename)

        elif ".csv" in self._filename:

            self._energy_type = TRUE_ENERGY

            return FromPlotAngResReader(self._filename)

        else:

            raise ValueError(
                self._filename
                + " is not recognised as one of the known angular resolution files."
            )

    def _get_ang_err(self, E):
        """
        Get the median angular error for the
        given Etrue/Ereco, corresponding to prob_contained.

        If scatter, sample from a normal distribution
        centred on the median value.
        """

        # Get median value for this true energy
        if self._energy_type == TRUE_ENERGY:

            true_energy_bin_cen = (
                self.true_energy_bins[:-1] + self.true_energy_bins[1:]
            ) / 2

            ang_res = np.interp(np.log(E), np.log(true_energy_bin_cen), self.values)

        elif self._energy_type == RECO_ENERGY:

            ang_res = np.interp(np.log(E), np.log(self.reco_energy_values), self.values)

        # Add scatter if required
        if self._scatter:

            a = (self._minimum - ang_res) / self._scatter
            b = (self._maximum - ang_res) / self._scatter

            ang_res = stats.truncnorm(a, b, loc=ang_res, scale=self._scatter,).rvs(
                1
            )[0]

        # Check bounds
        if ang_res < self._minimum:

            ang_res = self._minimum

        if ang_res > self._maximum:

            ang_res = self._maximum

        return ang_res

    def get_ret_ang_err(self, E):
        """
        Get the median angualr resolution for the
        given Etrue/Ereco, corresponsing to ret_ang_err_p.
        """

        ang_err = self._get_ang_err(E)

        kappa = get_kappa(ang_err, self.ang_err_p)

        return get_theta_p(kappa, self.ret_ang_err_p)

    def sample(self, Etrue, coord):
        """
        Sample new ra, dec values given a true energy
        and direction.
        """

        ra, dec = coord

        ang_err = self._get_ang_err(Etrue)

        sky_coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs")

        sky_coord.representation_type = "cartesian"

        unit_vector = np.array([sky_coord.x, sky_coord.y, sky_coord.z])

        kappa = get_kappa(ang_err, self.ang_err_p)

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

        self._ret_ang_err = get_theta_p(kappa, self.ret_ang_err_p)

        return new_ra, new_dec

    @property
    def ret_ang_err(self):

        return self._ret_ang_err

    @classmethod
    def from_dataset(cls, dataset_id, fetch=True, **kwargs):
        """
        Load angular resolution from publicly
        available data.

        :dataset_id: ID date of the dataset e.g. "20181018"
        :param fetch: If true, download dataset if missing
        """

        if dataset_id not in _supported_dataset_ids:

            raise NotImplementedError("This dataset is not currently supported")

        if fetch:

            data_interface = IceCubeData()

            dataset = data_interface.find(dataset_id)

            data_interface.fetch(dataset)

            dataset_dir = data_interface.get_path_to(dataset[0])

        else:

            dataset_dir = data_directory

        if dataset_id == "20181018":

            files = find_files(dataset_dir, R2018_ANG_RES_FILENAME)

            angres_file_name = files[2]

        elif dataset_id == "20210126":

            files = find_files(dataset_dir, R2021_ANG_RES_FILENAME)
            angres_file_name = files[0]
            return R2021AngularResolution(angres_file_name, **kwargs)

        return AngularResolution(angres_file_name, **kwargs)


class R2021AngularResolution:
    """
    Special class to handle smearing effects given in the 2021 data release:
    1) kinematic angle, what the readme calls "PSF"
    2) misreconstruction of tracks, what the readme calls "AngErr"
    """
    #TODO: use bins in logspace? seems like the better choice than linspace
    #      definitely use bins in logspace: e.g. PSF: erroneous scattering angles produced
    #TODO:  cleanup print() calls and imports
    #       
    def __init__(self, filename, **kwargs):

        self._energy_type = TRUE_ENERGY
        self._filename = filename

        self._reader = R2021AngResReader(self._filename) 

        self.dataset = self._reader.output 

        if self._energy_type == TRUE_ENERGY:
            self.true_energy_bins = self._reader.true_energy_bins
        else:
            raise NotImplementedError("Reconstructed energy is not implemented.")

        self.true_energy_values = (
            self.true_energy_bins[0:-1] + np.diff(self.true_energy_bins) / 2
        )
        self.declination_bins = self._reader.declination_bins

        self.ang_res_values = 1    # placeholder, isn't used anyway

        self.uniform = stats.uniform(0, 2*np.pi)

        # Dictionary of dictionary of... for both PSF and AngErr, energy and dec bin to
        # store marginal pdfs once they are needed.
        # Keys are indices of self._true_energy_bins[:-1] and self._declination_bins[:-1]
        self.marginal_pdfs = {"PSF":    {c: {} for c in range(self.true_energy_bins[:-1].shape[0])}, 
                              "AngErr": {c: {} for c in range(self.true_energy_bins[:-1].shape[0])}}

        #TODO: delete after testing
        self._kinematic_angles = []
        self._angular_errors = []
        self._azimuth_1 = []
        self._azimuth_2 = []


    def _return_bins(self, energy, declination):
        """
        Returns the lower bin edges and their indices for given energy and declination.
        """
        if energy >= self.true_energy_bins[0] and energy <= self.true_energy_bins[-1]:
            c_e = np.digitize(energy, self.true_energy_bins)
            #Need to get the index of lower bin edge.
            #np.digitize returns one too much, two if energy=highest bin edge
            if c_e < self.true_energy_bins.shape[0]:
                c_e -= 1
            else:
                c_e -= 2
            e = self.true_energy_bins[c_e]
        else:
            raise ValueError("Energy out of bounds.")


        if declination >= self.declination_bins[0] and declination <= self.declination_bins[-1]:
            c_d = np.digitize(declination, self.declination_bins)
            #Same procedure
            if c_d < self.declination_bins.shape[0]:
                c_d -= 1
            else:
                c_d -= 2
            d = self.declination_bins[c_d]
        else:
            raise ValueError("Declination out of bounds.")

        

        """
        for c_e, e in enumerate(self.true_energy_bins):
            if energy >= e and energy <= self.true_energy_bins[c_e+1]:
                break
        else:
            print("Outside of energy range")

        for c_d, d in enumerate(self.declination_bins):
            if declination >= d and declination <= self.declination_bins[c_d+1]:
                break
        else:
            print("Outside of declination range")
        """
        return c_e, e, c_d, d


    def _marginalisation(self, c_e, c_d, qoi):
        """
        Function that marginalises over the smearing data provided for the 2021 release.
        Arguments:
            dataset
            energy: energy of arriving neutrino
            declination: declination of arriving neutrino
            qui: quantity of interest, everything else is marginalised over
        Sticking to readme's naming convention for now.

        Careful: Samples in log-space!
        """
        
        if qoi == "PSF":
            needed_index = 6
        elif qoi == "AngErr":
            needed_index = 8
        else:
            raise ValueError("Not other quantity of interest is available.")
        
        #do pre-selection: lowest energy and highest declination, save into new array
        reduced_data = self.dataset[np.intersect1d(np.argwhere(
            np.isclose(self.dataset[:, 0], self.true_energy_bins[c_e])),
                                np.argwhere(
            np.isclose(self.dataset[:, 2], np.rad2deg(self.declination_bins[c_d]))))]
        
        bins = np.array(sorted(list(set(reduced_data[:, needed_index]).union(
                    set(reduced_data[:, needed_index+1])))))
        
        frac_counts = np.zeros(bins.shape[0]-1)
       
        #marginalise over uninteresting quantities
        for c_b, b in enumerate(bins[:-1]):
            indices = np.nonzero(np.isclose(b, reduced_data[:, needed_index]))

            frac_counts[c_b] = np.sum(reduced_data[indices, -1])
        return frac_counts, np.log10(bins)


    def _make_distribution(self, c_e, c_d, type_):
        """
        Create and store distribution of quantity of interest.
        """
        n, bins = self._marginalisation(self.true_energy_bins[c_e], self.declination_bins[c_d], type_)
        self.marginal_pdfs[type_][c_e][c_d] = stats.rv_histogram((n, bins))

    def _get_ang_err(self, c_e, c_d, type_):
        """
        Overwrite method of parent class with appropriate 2-step error sampling.
        TO BE DONE
        """

        azimuth = self.uniform.rvs(1)[0]
        try:
            deflection = self.marginal_pdfs[type_][c_e][c_d].rvs(size=1)[0]
        except KeyError:
            n, bins = self._marginalisation(c_e, c_d, type_)
            self.marginal_pdfs[type_][c_e][c_d] = stats.rv_histogram((n, bins))
            deflection = self.marginal_pdfs[type_][c_e][c_d].rvs(size=1)[0]   # draws log(angle) values
        return np.power(10, deflection), azimuth


    def _do_rotation(self, vec, c_e, c_d, type_):
        """
        Function called to sample deflections from appropriate distributions and
        rotate a coordinate vector by that amount.
        """
        
        def make_perp(vec):
            perp = np.zeros(3)
            if not np.all(np.isclose(vec[:2], 0.)):
                perp[0] = - vec[1]
                perp[1] = vec[0]
                perp /= np.linalg.norm(perp)
            else:
                perp[1] = 1.
            # print(perp)
            return perp

        #sample kinematic angle from distribution
        deflection, azimuth = self._get_ang_err(c_e, c_d, type_)
        print(deflection)
        # azimuth = 0.
        # print(deflection, azimuth)
        if type_ == "PSF":
            self._kinematic_angles.append(deflection)
            self._azimuth_1.append(azimuth)
        elif type_ == "AngErr":
            self._angular_errors.append(deflection)
            self._azimuth_2.append(azimuth)
        #first rotation vector is perpendicular to incident direction vector
        rot_vec_1 = make_perp(vec)
        #length is given by rotation angle, sampled from dist, converted to radians
        rot_vec_1 *= np.deg2rad(deflection) 
        #create rotation object from vector
        rot_1 = R.from_rotvec(rot_vec_1)
        #second rotation is around incident direction, length again sampled from uniform dist
        #azimuth already in radians
        rot_vec_2 = vec * azimuth
        rot_2 = R.from_rotvec(rot_vec_2)

        intermediate = rot_1.apply(vec)
        final = rot_2.apply(intermediate)

        return final


    def sample(self, Etrue, coord):
        """
        Sample new ra, dec values given a true energy and direction.
        """

        def get_angle(vec1, vec2):
            return np.rad2deg(np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))))
        ra, dec = coord
        sky_coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs")
        sky_coord.representation_type = "cartesian"
        unit_vector = np.array([sky_coord.x, sky_coord.y, sky_coord.z])

        c_e, e, c_d, d = self._return_bins(Etrue, dec)

        #for testing: only use one at a time
        intermediate_vector = self._do_rotation(unit_vector, c_e, c_d, "PSF")
        new_unit_vector = self._do_rotation(intermediate_vector, c_e, c_d, "AngErr")
        #create sky coordinates from rotated/deflected vector
        new_sky_coord = SkyCoord(
            x=new_unit_vector[0],
            y=new_unit_vector[1],
            z=new_unit_vector[2],
            representation_type="cartesian",
        )

        new_sky_coord.representation_type = "unitspherical"

        new_ra = new_sky_coord.ra.rad

        new_dec = new_sky_coord.dec.rad
        print(np.rad2deg(np.pi/2 - new_dec))
        reco_ang_err = get_angle(new_unit_vector, unit_vector)
        #return signature matches simulator.py

        # return unit_vector, intermediate_vector, new_unit_vector
        return new_ra, new_dec, reco_ang_err


    def create_sample(self, N, Etrue, coords):
        """
        Testing function to quickly generate a bunch of test data
        And I just wanted to show off that I know how to use yield.
        """
        for i in range(N):
            unit_vector, intermediate, new = self.sample(Etrue, coords)
            yield intermediate, new
            # ra, dec, reco_ang_err = self.sample(Etrue, coords)
            # return ra, dec, reco_ang_err

class FixedAngularResolution:
    """
    Simple fixed angular resolution.
    """

    def __init__(self, ang_err=1.0, ang_err_p=0.68, ret_ang_err_p=0.68):
        """
        Simple fixed angular resolution.

        :param ang_err: Resolution [deg].
        :param ang_err_p: The probability contained in the passed
        angular error region
        :param ret_ang_err_p: The returned angular error will correspond
        to a region containing this probability
        """

        self.ang_err = ang_err

        self.ang_err_p = ang_err_p

        self.ret_ang_err_p

        self._kappa = get_kappa(ang_err, ang_err_p)

        self._ret_ang_err = get_theta_p(self._kappa, ret_ang_err_p)

    @property
    def ret_ang_err(self):

        return self._ret_ang_err

    @property
    def kappa(self):

        return self._kappa

    def sample(self, coord):
        """
        Sample reconstructed coord given original position.

        :coord: ra, dec in [rad].
        """

        ra, dec = coord

        sky_coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs")

        sky_coord.representation_type = "cartesian"

        unit_vector = np.array([sky_coord.x, sky_coord.y, sky_coord.z])

        new_unit_vector = sample_vMF(unit_vector, self._kappa, 1)[0]

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
