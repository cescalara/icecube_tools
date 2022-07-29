import numpy as np
from scipy.stats import rv_histogram, uniform, norm
from scipy.spatial.transform import Rotation as R
from astropy.coordinates import SkyCoord
from astropy import units as u
from vMF import sample_vMF

import logging
logging.basicConfig(encoding='utf-8', level=logging.INFO)
from numpy.random import Generator, PCG64
seed = 42
scipy_randomGen = uniform
scipy_randomGen.random_state=Generator(PCG64(seed))

from icecube_tools.detector.energy_resolution import EnergyResolution
from icecube_tools.detector.angular_resolution import AngularResolution
from icecube_tools.utils.data import find_files, data_directory, IceCubeData, ddict
from icecube_tools.utils.vMF import get_kappa, get_theta_p


class R2021IRF(EnergyResolution, AngularResolution):
    """
    Special class to handle smearing effects given in the 2021 data release:
    1) kinematic angle, what the readme calls "PSF"
    2) misreconstruction of tracks, what the readme calls "AngErr"
    """
    
    def __init__(self, fetch=True, **kwargs):
        """
        Special class to handle smearing effects given in the 2021 data release.
        """

        #self._energy_type = TRUE_ENERGY
        self.read(fetch)

        self.year = 2012    # subject to change
        self.nu_type = "nu_mu"

        self.uniform = uniform(0, 2*np.pi)

        # self.reco_energy = {e: {d: {} for d in range(self.declination_bins.shape[0]-1)} for e in range(self.true_energy_bins.shape[0]-1)}
        # self.reco_energy = ddict()
        logging.debug('Creating Ereco distributions')
        self.reco_energy = np.empty((self.true_energy_bins.size-1, self.declination_bins.size-1), dtype=rv_histogram)
        self.reco_energy_bins = np.empty((self.true_energy_bins.size-1, self.declination_bins.size-1), dtype=np.ndarray)
        for c_e, e in enumerate(self.true_energy_bins[:-1]):
            for c_d, d in enumerate(self.declination_bins[:-1]):
                n, bins = self._marginalisation(c_e, c_d)
                self.reco_energy[c_e, c_d] = rv_histogram((n, bins))
                self.reco_energy_bins[c_e, c_d] = bins
        self._values = []
        logging.debug('Creating empty dicts for kinematic angle dists and angerr dists')

        self.marginal_pdf_psf = ddict()
        self.marginal_pdf_angerr = ddict()

        self.kinematic_angle_bin_list = []
        self.etrue_bin_list = []
        self.ereco_bin_list = []
        self.dec_bin_list = []
        

    def _get_index_function(self, dist, value):
        def index_function(value):
            #np.digitize goes here
            pass
        return index_function(value)


    def reset_lists(self):
        self.kinematic_angle_bins = []


    def sample_energy(self, Etrue, dec):
        """
        Sample reconstructed energy according to distribution depending on true energy and declination.
        :param Etrue: True energy in $\log_{10}(E/\mathrm{GeV})$
        :param dec: declination in rad
        :return: reconstructed energy in GeV
        """

        c_e, _, c_d, _ = self._return_etrue_bins(Etrue, dec)
        logging.debug(f'Energy and declination bins: {c_e}, {c_d}')
        #sample Ereco
        logging.debug('Sampling Ereco')
        Ereco = self.reco_energy(c_e, c_d, 'pdf').rvs(size=1)[0]
        logging.debug(f'Ereco: {Ereco}')
        return np.power(10, Ereco)


    @staticmethod
    def get_angle(vec1, vec2):
        """
        Calculate the angle between two vectors.
        :param vec1: First vector
        :param vec2: Second vector
        :return: Angle between vectors in degrees
        """

        return np.rad2deg(np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))))


    def sample(self, coord, Etrue):
        """
        Sample new ra, dec values given a true energy and direction.
        :param coord: Tuple indicident coordinates (ra, dec) in radians
        :param Etrue: True $\log_{10}(E/\mathrm{GeV})$ that's to be sampled.
        :param Ereco: If Ereco is float (in $\log_{10}(E/\mathrm{GeV})$) then sampling of Ereco is omitted 
        :return: new rectascension and new declination in rad of deflected particle, angle between incident and deflected direction in degrees
        """

        ra, dec = coord
        sky_coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs")
        sky_coord.representation_type = "cartesian"
        unit_vector = np.array([sky_coord.x, sky_coord.y, sky_coord.z])

        if isinstance(Etrue, np.ndarray):
            size = Etrue.size
        else:
            size = 1

        #Initialise empty arrays for data
        c_e, _, c_d, _ = self._return_etrue_bins(Etrue, dec)
        c_e_r = np.zeros(size)
        c_k = np.zeros(size)
        c_ang_err = np.zeros(size)
        Ereco = np.zeros(size)
        kinematic_angle = np.zeros(size)
        ang_err = np.zeros(size)
        new_ras = np.zeros(size)
        new_decs = np.zeros(size)


        logging.debug(f'Energy and declination bins: {c_e}, {c_d}')
        logging.debug('Sampling Ereco')

        #sample Ereco
        set_e = set(c_e)
        set_d = set(c_d)

        for idx_e in set_e:
            _index_e = np.argwhere(idx_e == c_e).squeeze()

            for idx_d in set_d:
                _index_d = np.argwhere(idx_d == c_d).squeeze()
                _index_f = (np.intersect1d(_index_d, _index_e),)

                Ereco[_index_f] = self.reco_energy[idx_e, idx_d].rvs(size=_index_f[0].size)
                current_c_e_r = self._return_reco_energy_bins(c_e, c_d, Ereco[_index_f])
                c_e_r[_index_f] = current_c_e_r

                logging.debug(f'Ereco: {Ereco[_index_f]}, bin: {current_c_e_r}')

                set_e_r = set(current_c_e_r)
                size_e_r = np.bincount(current_c_e_r)[np.nonzero(np.bincount(current_c_e_r) != 0)] 
                
                for idx_e_r in set_e_r:
                    _index_help = np.argwhere(c_e_r == idx_e_r).squeeze()
                    _index_r = (np.intersect1d(_index_f[0], _index_help),)

                    try:
                        kinematic_angle[_index_r] = self.marginal_pdf_psf(idx_e, idx_d, idx_e_r).rvs(size=size_e_r)

                    except KeyError:
                        #logging.debug(f'Creating kinematic angle dist for {c_e}, {c_d}, {c_e_r}')
                        n, bins = self._marginalize_over_angerr(idx_e, idx_d, idx_e_r)
                        self.marginal_pdf_psf.add(bins, idx_e, idx_d, idx_e_r, 'bins')
                        self.marginal_pdf_psf.add(rv_histogram((n, bins)), idx_e, idx_d, idx_e_r, 'pdf')
                        kinematic_angle[_index_r] = self.marginal_pdf_psf(idx_e, idx_d, idx_e_r, 'pdf').rvs(size=size_e_r)  
        
                        #logging.debug(f'kinematic angle: {kinematic_angle}')
                        #logging.debug(f'probability density of kin ang: {self.marginal_pdf_psf[c_e][c_d][c_e_r]["pdf"].pdf(kinematic_angle)}')

                    current_c_k = self._return_kinematic_bins(idx_e, idx_d, idx_e_r, kinematic_angle[_index_r])
                    c_k[_index_r] = current_c_k
                    set_k = set(current_c_k)
                    # logging.debug(f'Kinematic angle bin: {c_k}')
                    size_k = np.bincount(current_c_k)[np.nonzero(np.bincount(current_c_k) != 0)]

                    for idx_k in set_k:
                        _index_help = np.argwhere(c_k == idx_k).squeeze()
                        _index_k = (np.intersect1d(_index_r[0], _index_help),)

                        try:
                            ang_err[_index_k] = self.marginal_pdf_angerr(idx_e, idx_d, idx_e_r, idx_k, 'pdf').rvs(size=size_k)

                        except KeyError as KE:
                            #logging.debug(f'Creating AngErr dist for {c_e}, {c_d}, {c_e_r}, {c_k}')
                            n, bins = self._get_angerr_dist(idx_e, idx_d, idx_e_r, idx_k)
                            self.marginal_pdf_angerr.add(rv_histogram((n, bins)), idx_e, idx_d, idx_e_r, idx_k, 'pdf') 
                            self.marginal_pdf_angerr.add(bins, idx_e, idx_d, idx_e_r, idx_k, 'bins')
                            ang_err[_index_k] = self.marginal_pdf_angerr(idx_e, idx_d, idx_e_r, idx_k, 'pdf').rvs(size=size_k)


        #logging.debug(f'Angular error: {ang_err}')
        #logging.debug(f'probability density: {self.marginal_pdf_angerr(c_e, c_d, c_e_r, c_k, "pdf").pdf(ang_err)}')
        #kappa needs an angle in degrees, prob of containment, here 0.5 as stated in the paper
        for c, ang in enumerate(ang_err):
            kappa = get_kappa(np.power(10, ang), 0.5)
            new_unit_vector = sample_vMF(unit_vector, kappa, 1)[0]

            #create sky coordinates from rotated/deflected vector
            new_sky_coord = SkyCoord(
                x=new_unit_vector[0],
                y=new_unit_vector[1],
                z=new_unit_vector[2],
                representation_type="cartesian",
            )
            new_sky_coord.representation_type = "unitspherical"

            new_ras[c] = new_sky_coord.ra.rad
            new_decs[c] = new_sky_coord.dec.rad
            ang_err[c] = self.get_angle(new_unit_vector, unit_vector) 

        return new_ras, new_decs, ang_err, np.power(10, Ereco)


    def read(self, fetch):
        """
        Reads in IRFs of data set.
        For consistency and reducing the error-prone...iness, kinematic angles ("PSF") and angular errors are converted to log(degrees).
        """

        self.prob_contained = 0.68

        self.year = 2012    # subject to change
        self.nu_type = "nu_mu"

        if fetch:
            data_interface = IceCubeData()
            dataset = data_interface.find("20210126")
            data_interface.fetch(dataset)
            dataset_dir = data_interface.get_path_to(dataset[0])

        filename = find_files(data_directory, "IC86_II_smearing.csv")[0]
        self._filename = filename
        self.output = np.loadtxt(self._filename, comments="#")
        self.dataset = self.output
        true_energy_lower = np.array(list(set(self.output[:, 0])))
        true_energy_upper = np.array(list(set(self.output[:, 1])))

        #convert PSF and AngErr values to log(angle/degree)
        self.dataset[:, 6:-1] = np.log10(self.dataset[:, 6:-1])


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


    def _get_angerr_dist(self, c_e, c_d, c_e_r, c_psf):
        """
        Return the angular error distribution given other parameters.
        :param c_e: True energy bin index
        :param c_d: Declination bin index
        :param c_e_r: Reconstructed energy bin index
        :param c_psf: Kinematic angle bin index
        :return: normalised fractional counts, logarithmic bins in log(degrees)
        """

        presel_data = self.dataset[np.intersect1d(np.nonzero(np.isclose(self.dataset[:, 0], self.true_energy_bins[c_e])),
                                                  np.nonzero(np.isclose(self.dataset[:, 2], np.rad2deg(self.declination_bins[c_d]))))]

        reduced_data = presel_data[np.intersect1d(np.nonzero(np.isclose(presel_data[:, 4], self.reco_energy_bins[c_e, c_d][c_e_r])),
                                                  np.nonzero(np.isclose(presel_data[:, 6], self.marginal_pdf_psf(c_e, c_d, c_e_r, 'bins', c_psf))))]
        
        needed_vals = np.nonzero(reduced_data[:, 9] -  reduced_data[:, 8])
        bins = np.union1d(reduced_data[needed_vals, 9], reduced_data[needed_vals, 8])
        frac_counts = reduced_data[needed_vals, -1].squeeze()
        frac_counts /= np.sum(frac_counts)

        return frac_counts, bins


    def _return_etrue_bins(self, energy, declination):
        """
        Returns the lower bin edges and their indices for given energy and declination.
        :param energy: Energy in $\log_{10}(E/\mathrm{GeV})$
        :param declination: Declination in rad
        :return: Index of energy, energy at lower bin edge, index of declination, declination at lower bin edge
        :raises ValueError: if energy is outside of IRF-file range
        :raises ValueError: if declination is outside of $[-\pi/2, \pi/2]$
        """

        if np.all(energy >= self.true_energy_bins[0]) and np.all(energy <= self.true_energy_bins[-1]):
            c_e = np.digitize(energy, self.true_energy_bins)
            idx = np.nonzero(c_e < self.true_energy_bins.shape[0])
            c_e[idx] = c_e[idx] - 1
            idx = np.nonzero(c_e == self.true_energy_bins.shape[0])
            c_e[idx] = c_e[idx] - 2

            e = self.true_energy_bins[c_e]
        else:
            raise ValueError("Some energy out of bounds.")


        if np.all(declination >= self.declination_bins[0]) and np.all(declination <= self.declination_bins[-1]):
            
            c_d = np.digitize(declination, self.declination_bins)

            idx = np.nonzero(c_d < self.declination_bins.shape[0])
            c_d[idx] -= 1
            idx = np.nonzero(c_d == self.declination_bins.shape[0])
            c_d[idx] -= 2

            d = self.declination_bins[c_d]
        else:
            raise ValueError("Some declination out of bounds.")
        
        return c_e, e, c_d, d


    def _return_reco_energy_bins(self, c_e, c_d, Ereco):
        """
        Return bin index of reconstructed energy.
        :param c_e: Index of true energy bin
        :param c_d: Index of declination bin
        :param Ereco: Reconstructed energy in $\log_{10}(E/\mathrm{GeV})$
        """

        bins = self.reco_energy_bins[c_e, c_d][0]
        index = np.digitize(Ereco, bins)
        idx = np.nonzero(index < bins.shape[0])
        index[idx] = index[idx] - 1
        idx = np.nonzero(index == bins.shape[0])
        index[idx] = index[idx] - 2
        
        return index


    def _return_kinematic_bins(self, c_e, c_d, c_e_r, angle):
        """
        Returns bin index of kinematic angle given in log(degees).
        :param c_e: Bin index of true energy
        :param c_d: Bin index of declination
        :param c_e_r: Bin index of reconstructed energy
        :return: Bin index of kinematic angle
        """

        bins = self.marginal_pdf_psf(c_e, c_d, c_e_r, 'bins')
        c_k = np.digitize(angle, bins) - 1
        idx = np.nonzero(c_k == bins.shape[0] - 1)
        c_k[idx] = c_k[idx] - 1
        
        return c_k


    def _marginalisation(self, c_e, c_d, qoi="ERec"):
        """
        Function that marginalises over the smearing data provided for the 2021 release.
        Careful: Samples are drawn in logspace and converted to linspace upon return.
        :param int c_e: Index of energy bin
        :param int c_d: Index of declination bin
        :return: n, bins of the created distribution/histogram
        """
 
        if qoi == "ERec":
            needed_index = 4
        else:
            raise ValueError("Not other quantity of interest is available.")
        
        #do pre-selection of true energy and declination
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
        
        return frac_counts, bins


    def _marginalize_over_angerr(self, c_e, c_d, c_e_r): 
        """
        Function that marginalises over the smearing data provided for the 2021 release.
        :param int c_e: Index of energy bin
        :param int c_d: Index of declination bin
        :param_c_e_r: Index of reconstructed energy bin
        :return: n, bins of the created distribution/histogram
        """

        presel_data = self.dataset[np.intersect1d(np.nonzero(np.isclose(self.dataset[:, 0], self.true_energy_bins[c_e])),
                                                  np.nonzero(np.isclose(self.dataset[:, 2], np.rad2deg(self.declination_bins[c_d]))))]

        reduced_data = presel_data[np.nonzero(np.isclose(presel_data[:, 4], self.reco_energy_bins[c_e, c_d][c_e_r]))]


        bins = np.array(sorted(list(set(reduced_data[:, 6]).union(
                    set(reduced_data[:, 7])))))
        if bins.shape[0] != 0:
            frac_counts = np.zeros(bins.shape[0]-1)
 
            #marginalise over uninteresting quantities
            for c_b, b in enumerate(bins[:-1]):
                indices = np.nonzero(np.isclose(b, reduced_data[:, 6]))
                # logging.debug(f'{reduced_data[indices, -1]}')
                frac_counts[c_b] = np.sum(reduced_data[indices, -1])
                # logging.debug(f'{frac_counts[c_b]}')
                # logging.debug(f'{c_b}, {frac_counts[c_b]}')
            return frac_counts, bins

        else:
            return None, None


    def _do_rotation(self, vec, deflection):
        """
        Function called to sample deflections from appropriate distributions and
        rotate a coordinate vector by that amount.
        :param vec: Vector to be rotated/deflected
        :param deflection: Angle of deflection in log(degrees)
        :returns: rotated vector
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
        azimuth = self.uniform.rvs(size=1)[0]
        deflection = np.deg2rad(np.power(10, deflection))
        logging.debug(f'azimuth: {azimuth}\ndeflection: {deflection}')
        rot_vec_1 = make_perp(vec)
        rot_vec_1 *= deflection 
        #create rotation object from vector
        rot_1 = R.from_rotvec(rot_vec_1)
        
        rot_vec_2 = vec / np.linalg.norm(vec)
        rot_vec_2 *= azimuth
        rot_2 = R.from_rotvec(rot_vec_2)

        intermediate = rot_1.apply(vec)
        final = rot_2.apply(intermediate)
        return final

