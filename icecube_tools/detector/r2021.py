import numpy as np
from scipy.stats import rv_histogram, uniform
from scipy.spatial.transform import Rotation as R
from astropy.coordinates import SkyCoord
from astropy import units as u
import logging
seed = 42
from numpy.random import Generator, PCG64
scipy_randomGen = uniform
scipy_randomGen.random_state=Generator(PCG64(seed))
import matplotlib.pyplot as plt
logging.basicConfig(encoding='utf-8', level=logging.INFO)
#TODO inheritance or not?
class R2021IRFReader():
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


class R2021IRF():
    """
    Special class to handle smearing effects given in the 2021 data release:
    1) kinematic angle, what the readme calls "PSF"
    2) misreconstruction of tracks, what the readme calls "AngErr"
    """
    
    def __init__(self, filename, **kwargs):
        """
        Special class to handle smearing effects given in the 2021 data release.
        :param filename: Filename to be read in
        """



        #self._energy_type = TRUE_ENERGY
        self._filename = filename

        self.year = 2012    # subject to change
        self.nu_type = "nu_mu"

        self.read()

        self.true_energy_values = (
            self.true_energy_bins[0:-1] + np.diff(self.true_energy_bins) / 2
        )


        self.uniform = uniform(0, 2*np.pi)

        # Dictionary of dictionary of... for both PSF and AngErr, energy and dec bin to
        # store marginal pdfs once they are needed.
        # Keys are indices of self._true_energy_bins[:-1] and self._declination_bins[:-1]
        # marginal pdfs: for each Etrue, dec, Ereco one for PSF marginalised over AngErr
        # for each Etrue, dec, Ereco, PSF one for AngErr
        # contains for each Etrue bin, declination bin, histogram of Ereco and its bins
        # self.reco_energy = self._reader.reco_energy 


        """
        for c_e, e in enumerate(self.true_energy_bins[:-1]):
            for c_d, d in enumerate(self.declination_bins[:-1]):
                reduced_data = self.dataset[np.intersect1d(np.argwhere(
                    np.isclose(self.dataset[:, 0], self.true_energy_bins[c_e])),
                                np.argwhere(
                    np.isclose(self.dataset[:, 2], np.rad2deg(self.declination_bins[c_d]))))]
        
                 bins = np.array(sorted(list(set(reduced_data[:, ]).union(
                            set(reduced_data[:, needed_index+1])))))

        """

        self.reco_energy = {e: {d: {} for d in range(self.declination_bins.shape[0]-1)} for e in range(self.true_energy_bins.shape[0]-1)}
        logging.info('Creating Ereco distributions')
        for c_e, e in enumerate(self.true_energy_bins[:-1]):
            for c_d, d in enumerate(self.declination_bins[:-1]):
                n, bins = self._marginalisation(c_e, c_d)
                self.reco_energy[c_e][c_d]['pdf'] = rv_histogram((n, bins))
                self.reco_energy[c_e][c_d]['bins'] = bins

        self._values = []
        logging.info('Creating empty dicts for kinematic angle dists and angerr dists')
        self.marginal_pdf_psf = {etrue: {d: {} 
                                            for d in range(self.declination_bins[:-1].shape[0])} 
                                            for etrue in range(self.true_energy_bins[:-1].shape[0])}

        self.marginal_pdf_angerr = {etrue: {
                                        d: {} for d in range(self.declination_bins[:-1].shape[0])}
                                              for etrue in range(self.true_energy_bins[:-1].shape[0])}

        #loop over all bins up to reco energy
        #marginalize over angerr, make distribution of psf
        #while at it, make a distribution of angerr for every psf bin
        #TODO this is not feasible, do it only if necessary bc value is actually drawn
        """
        for c_e, e in self.reco_energy.items():
            #c_e, e are index/true energy-dict
            for c_d, d in e.items():
                #c_d, d are index/declination-dict
                for c_b, b in enumerate(d['bins'][:-1]):
                    n, bins = self._marginalize_over_angerr(c_e, c_d, c_b)
                    self.marginal_pdf_psf[c_e][c_d][c_b] = {}
                    self.marginal_pdf_psf[c_e][c_d][c_b]['bins'] = bins
                    if n is not None:
                        self.marginal_pdf_psf[c_e][c_d][c_b]['pdf'] = rv_histogram((n, bins))
                    else:
                        continue
                    self.marginal_pdf_angerr[c_e][c_d][c_b] = {}
                    for c_psf, psf_bin in enumerate(bins):
                        n, bins = self._get_angerr_dist(c_e, c_d, c_b, c_psf)
                        self.marginal_pdf_angerr[c_e][c_d][c_b][c_psf] = {}
                        self.marginal_pdf_angerr[c_e][c_d][c_b][c_psf]['bins'] = bins
                        print(n)
                        print(bins)
                        if n:
                            self.marginal_pdf_angerr[c_e][c_d][c_b][c_psf]['pdf'] = rv_histogram((n, bins))
                        else:
                            continue
        """
        """
        self.marginal_pdf_angerr = {etrue: {
                                        d: {} for d in range(self.declination_bins[:-1].shape[0])}
                                                for etrue in range(self.true_energy_bins[:-1].shape[0])}
        """
        """
        Need a marginal pdf for reco energy for each Etrue/dec bin
            - then for each reco energy:
                - marginalise over AngErr, make dist
                - sample PSF value
                    - for that PSF value, find bin
                    - sample from resulting AngErr dist
                    - that's the angular uncertainty
        """
        
        #TODO: delete after testing
        self._kinematic_angles = []
        self._angular_errors = []
        self._azimuth_1 = []
        self._azimuth_2 = []


    def _get_index_function(self, dist, value):
        def index_function(value):
            #np.digitize goes here
            pass
        return index_function(value)


    def sample(self, coord, Etrue, Ereco=None):
        """
        Sample new ra, dec values given a true energy and direction.
        :param Etrue: True $\log_{10}(E/\mathrm{GeV})$ that's to be sampled.
        :param coord: Tuple indicident coordinates (ra, dec) in radians
        :param Etype: Either "Ereco" or "Etrue"
        :returns: new rectascension and new declination of deflected particle, angle between incident and deflected direction in degrees
        """

        def get_angle(vec1, vec2):
            return np.rad2deg(np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))))
        ra, dec = coord
        sky_coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs")
        sky_coord.representation_type = "cartesian"
        unit_vector = np.array([sky_coord.x, sky_coord.y, sky_coord.z])


        """
        if Ereco is None:
            sample E reco from according dist of Etrue, dec
        elif Ereco is not None:
            just continue


        get indices of Etrue, Ereco

        sample according PSF, marginalised over AngErr

        sample AngErr given the sampled PSF value

        sample azimuth

        do rotation

        make new skycoord

        calculate angle between initial and final direction, but that should just be AngErr

        """

        ra, dec = coord

        c_e, _, c_d, _ = self._return_etrue_bins(Etrue, dec)
        logging.info(f'Energy and declination bins: {c_e}, {c_d}')
        if Ereco is None:
            #sample Ereco
            logging.info('Sampling Ereco')
            Ereco = self.reco_energy[c_e][c_d]['pdf'].rvs(size=1)[0]
        #get Ereco index
        c_e_r = self._return_reco_energy_bins(c_e, c_d, Ereco)    
        logging.info(f'Ereco: {Ereco}, bin: {c_e_r}')
        #sample appropriate psf distribution
        try:
            kinematic_angle = self.marginal_pdf_psf[c_e][c_d][c_e_r]['pdf'].rvs(size=1)[0]
            samples = self.marginal_pdf_psf[c_e][c_d][c_e_r]['pdf'].rvs(size=1000)
        except KeyError:
            logging.info(f'Creating kinematic angle dist for {c_e}, {c_d}, {c_e_r}')
            n, bins = self._marginalize_over_angerr(c_e, c_d, c_e_r)
            self.marginal_pdf_psf[c_e][c_d][c_e_r] = {}
            self.marginal_pdf_psf[c_e][c_d][c_e_r]['bins'] = bins
            self.marginal_pdf_psf[c_e][c_d][c_e_r]['pdf'] = rv_histogram((n, bins))
            # logging.info(f'{n}\n{bins}')
            kinematic_angle = self.marginal_pdf_psf[c_e][c_d][c_e_r]['pdf'].rvs(size=1)[0]  
            # samples = self.marginal_pdf_psf[c_e][c_d][c_e_r]['pdf'].rvs(size=1000)
        # x = np.linspace(*self.marginal_pdf_psf[c_e][c_d][c_e_r]['pdf'].support(), num=1000)
        # pdf = self.marginal_pdf_psf[c_e][c_d][c_e_r]['pdf'].pdf(x)
        # plt.plot(x, pdf)
        # plt.vlines(kinematic_angle, 0, pdf.max())
        logging.info(f'kinematic angle: {kinematic_angle}')
        logging.info(f'probability density of kin ang: {self.marginal_pdf_psf[c_e][c_d][c_e_r]["pdf"].pdf(kinematic_angle)}')
        # plt.hist(samples, bins=self.marginal_pdf_psf[c_e][c_d][c_e_r]['bins'], alpha=0.5, density=True)
        self._kinematic_angles.append(np.power(10, kinematic_angle))
        
        if np.isclose(self.marginal_pdf_psf[c_e][c_d][c_e_r]['pdf'].pdf(kinematic_angle), 0):
            logging.error('Sampled zero-chance value')
            # raise ValueError("Sampled zero-chance value")
        #get kinematic angle index
        c_k = self._return_kinematic_bins(c_e, c_d, c_e_r, kinematic_angle)
        logging.info(f'Kinematic angle bin: {c_k}')
        #sample appropriate ang_err distribution
        try:
            ang_err = self.marginal_pdf_angerr[c_e][c_d][c_e_r][c_k]['pdf'].rvs(1)
        except KeyError as KE1:
            logging.info(f'Creating AngErr dist for {c_e}, {c_d}, {c_e_r}, {c_k}')
            try:
                self.marginal_pdf_angerr[c_e][c_d][c_e_r]
            except KeyError as KE2:
                self.marginal_pdf_angerr[c_e][c_d][c_e_r] = {}
            self.marginal_pdf_angerr[c_e][c_d][c_e_r][c_k] = {}

            n, bins = self._get_angerr_dist(c_e, c_d, c_e_r, c_k)
            self.marginal_pdf_angerr[c_e][c_d][c_e_r][c_k]['pdf'] = rv_histogram((n, bins)) 
            self.marginal_pdf_angerr[c_e][c_d][c_e_r][c_k]['bins'] = bins
            # logging.info(f'{n}, {bins}')
            ang_err = self.marginal_pdf_angerr[c_e][c_d][c_e_r][c_k]['pdf'].rvs(size=1)[0]
        logging.info(f'Angular error: {ang_err}')
        logging.info(f'probability density: {self.marginal_pdf_angerr[c_e][c_d][c_e_r][c_k]["pdf"].pdf(ang_err)}')

        new_unit_vector = self._do_rotation(unit_vector, ang_err)
        

        #create sky coordinates from rotated/deflected vector
        new_sky_coord = SkyCoord(
            x=new_unit_vector[0],
            y=new_unit_vector[1],
            z=new_unit_vector[2],
            representation_type="cartesian",
        )

        new_sky_coord.representation_type = "unitspherical"

        new_ra = new_sky_coord.ra.deg

        new_dec = new_sky_coord.dec.deg
        reco_ang_err = get_angle(new_unit_vector, unit_vector)
        #return signature matches simulator.py

        # return unit_vector, intermediate_vector, new_unit_vector
        return new_ra, new_dec, reco_ang_err
        


    def read(self):

        self.prob_contained = 0.68

        self.year = 2012    # subject to change
        self.nu_type = "nu_mu"

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
        :return: normalised fractional counts, logarithmic bins in degrees
        """

        presel_data = self.dataset[np.intersect1d(np.nonzero(np.isclose(self.dataset[:, 0], self.true_energy_bins[c_e])),
                                                  np.nonzero(np.isclose(self.dataset[:, 2], np.rad2deg(self.declination_bins[c_d]))))]

        reduced_data = presel_data[np.intersect1d(np.nonzero(np.isclose(presel_data[:, 4], self.reco_energy[c_e][c_d]['bins'][c_e_r])),
                                                  np.nonzero(np.isclose(presel_data[:, 6], self.marginal_pdf_psf[c_e][c_d][c_e_r]['bins'][c_psf])))]
        
        needed_vals = np.nonzero(reduced_data[:, 9] -  reduced_data[:, 8])
        # logging.info(f'{needed_vals}')
        bins = np.union1d(reduced_data[needed_vals, 9], reduced_data[needed_vals, 8])
        # logging.info(f'{bins}')
        frac_counts = reduced_data[needed_vals, -1].squeeze()
        frac_counts /= np.sum(frac_counts)

        return frac_counts, bins


    def _return_etrue_bins(self, energy, declination):
        """
        Returns the lower bin edges and their indices for given energy and declination.
        :param float energy: Energy in $\log_{10}(E/\mathrm{GeV})$
        :param float declination: Declination in rad
        :return: Index of energy, energy at lower bin edge, index of declination, declination at lower bin edge
        :raises ValueError: if energy is outside of IRF-file range
        :raises ValueError: if declination is outside of $[-\pi/2, \pi/2]$
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
        
        return c_e, e, c_d, d


    def _return_reco_energy_bins(self, c_e, c_d, Ereco):
        """
        Return bin index of reconstructed energy.
        :param c_e: Index of true energy bin
        :param c_d: Index of declination bin
        :param Ereco: Reconstructed energy in $\log_{10}(E/\mathrm{GeV})$
        """

        try:
            bins = self.reco_energy[c_e][c_d]['bins']
            index = np.digitize(Ereco, bins)
            if index < bins.shape[0]:
                index -= 1
            else:
                index -= 2
        except KeyError as e:
            print(e)

        return index


    def _return_kinematic_bins(self, c_e, c_d, c_e_r, angle):
        bins = self.marginal_pdf_psf[c_e][c_d][c_e_r]['bins']
        c_k = np.digitize(angle, bins)        
        if c_k < bins.shape[0]:
            c_k -= 1
        else:
            c_k -= 2
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

        reduced_data = presel_data[np.nonzero(np.isclose(presel_data[:, 4], self.reco_energy[c_e][c_d]['bins'][c_e_r]))]
        

        bins = np.array(sorted(list(set(reduced_data[:, 6]).union(
                    set(reduced_data[:, 7])))))
        if bins.shape[0] != 0:
            frac_counts = np.zeros(bins.shape[0]-1)
 
            #marginalise over uninteresting quantities
            for c_b, b in enumerate(bins[:-1]):
                indices = np.nonzero(np.isclose(b, reduced_data[:, 6]))
                # logging.info(f'{reduced_data[indices, -1]}')
                frac_counts[c_b] = np.sum(reduced_data[indices, -1])
                # logging.info(f'{frac_counts[c_b]}')
                # logging.info(f'{c_b}, {frac_counts[c_b]}')
            return frac_counts, bins

        else:
            return None, None
    '''
    def _make_distribution(self, c_e, c_d, type_):
        """
        DEPRECATED?
        Create and store distribution of quantity of interest.
        :param c_e: Bin index of energy
        :param c_d: Bin index of declination
        :param type_: Either "PSF" or "AngErr"
        """

        n, bins = self._marginalisation(c_e, c_d, type_)
        self.marginal_pdfs[type_][c_e][c_d] = stats.rv_histogram((n, bins))
    '''
    '''
    def _get_ang_err(self, c_e, c_d, type_):
        """
        Returns random samples of deflection angle and azimuth for mis-reconstruction.
        :param c_e: Bin index of energy
        :param c_d: Bin index of declination
        :param type_: Either "PSF" or "AngErr"
        :returns: Sampled values of deflection and azimuth, both in radians
        """

        azimuth = self.uniform.rvs(1)[0]
        try:
            deflection = self.marginal_pdfs[type_][c_e][c_d].rvs(size=1)[0]
        except KeyError:
            n, bins = self._marginalisation(c_e, c_d, type_)
            self.marginal_pdfs[type_][c_e][c_d] = stats.rv_histogram((n, bins))
            deflection = self.marginal_pdfs[type_][c_e][c_d].rvs(size=1)[0]   # draws log(angle) values
        return np.deg2rad(np.power(10, deflection)), azimuth
    '''

    def _do_rotation(self, vec, deflection):
        """
        Function called to sample deflections from appropriate distributions and
        rotate a coordinate vector by that amount.
        :param vec: Vector to be rotated/deflected
        :param c_e: Bin index of energy
        :param c_d: Bin inde of declination
        :param type_: Either "PSF" or "AngErr"
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
        logging.info(f'azimuth: {azimuth}\ndeflection: {deflection}')
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

    '''
    def sample(self, coord, Etrue=None, Ereco=None):
        """
        Sample new ra, dec values given a true energy and direction.
        :param Etrue: True $\log_{10}(E/\mathrm{GeV})$ that's to be sampled.
        :param coord: Tuple indicident coordinates (ra, dec) in radians
        :param Etype: Either "Ereco" or "Etrue"
        :returns: new rectascension and new declination of deflected particle, angle between incident and deflected direction in degrees
        """

        def get_angle(vec1, vec2):
            return np.rad2deg(np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))))
        ra, dec = coord
        sky_coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs")
        sky_coord.representation_type = "cartesian"
        unit_vector = np.array([sky_coord.x, sky_coord.y, sky_coord.z])


        """
        if Ereco is None:
            sample E reco from according dist of Etrue, dec
        elif Ereco is not None:
            just continue


        get indices of Etrue, Ereco

        sample according PSF, marginalised over AngErr

        sample AngErr given the sampled PSF value

        sample azimuth

        do rotation

        make new skycoord

        calculate angle between initial and final direction, but that should just be AngErr

        """



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
        reco_ang_err = get_angle(new_unit_vector, unit_vector)
        #return signature matches simulator.py

        # return unit_vector, intermediate_vector, new_unit_vector
        return new_ra, new_dec, reco_ang_err
    '''

