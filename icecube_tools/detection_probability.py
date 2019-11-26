import numpy as np
from scipy.stats import poisson
from scipy.optimize import curve_fit, fsolve
import h5py

"""
Functions for calculating the detection probabilty 
from the results of point source analysis.
"""

def get_simulated_params(filename):
    """
    Returns the available declinations 
    and spectral indices in the file.
    """

    with h5py.File(filename, 'r') as f:

        dec_to_sim = f['dec_to_sim'][()]
        
        index_to_sim = f['index_to_sim'][()]

        
    return dec_to_sim, index_to_sim


def get_detection_probability(filename, index, dec, TS_threshold):
    """
    Find the detection probability as a function 
    of the expected number of source counts in 
    a detector.

    Returns Nsrc_list and Pdet
    :param filename: Filename
    :param index: Spectral index
    :param dec: Declination
    :param TS_threshold: TS <=> 5sigma threshold
    """

    with h5py.File(filename, 'r') as f:

        Nsrc_list = f['Nsrc_list'][()]

        Ntrials = f['Ntrials'][()]

        folder = f['dec_%.2f' % dec]
        
        subfolder = folder['index_%.2f' % index]

        TS = []

        for Nsrc in Nsrc_list:

            TS.append(subfolder['TS_'+str(Nsrc)][()])

    # Find Pdet for each expected Nsrc

    Pdet_at_Nsrc = []
    for i, Nsrc in enumerate(Nsrc_list):

        idx = np.where(~np.isnan(TS[i]))
        ts = TS[i][idx]
        
        P = len(ts[ts > TS_threshold]) / len(ts)

        Pdet_at_Nsrc.append(P)
    
    # Weight by poisson probability

    Pdet = []
    for Nsrc in Nsrc_list:

        P = sum([w * poisson(Nsrc).pmf(i) for i, w in enumerate(Pdet_at_Nsrc)])

        Pdet.append(P)

    return Nsrc_list, Pdet



def get_detection_probability_Braun2008(filename, index, TS_threshold):
    """
    Find the detection probability as a function 
    of the expected number of source counts in 
    a detector.

    Returns Nsrc_list and Pdet
    :param filename: Filename
    :param index: spectral index
    :param TS_threshold: TS <=> 5sigma threshold
    """

    with h5py.File(filename, 'r') as f:
        
        index_to_sim = f['index_to_sim'][()]

        Nsrc_list = f['Nsrc_list'][()]

        Ntrials = f['Ntrials'][()]

        folder = f['index_%.2f' % index]

        TS = []

        for Nsrc in Nsrc_list:

            TS.append(folder['TS_'+str(Nsrc)][()])

    # Find Pdet for each expected Nsrc

    Pdet_at_Nsrc = []
    for i, Nsrc in enumerate(Nsrc_list):

        idx = np.where(~np.isnan(TS[i]))
        ts = TS[i][idx]
        
        P = len(ts[ts > TS_threshold]) / len(ts)

        Pdet_at_Nsrc.append(P)
    
    # Weight by poisson probability

    Pdet = []
    for Nsrc in Nsrc_list:

        P = sum([w * poisson(Nsrc).pmf(i) for i, w in enumerate(Pdet_at_Nsrc)])

        Pdet.append(P)

    return Nsrc_list, Pdet


def get_TS_threshold(TS, level, above=5):
    """
    Return TS at specified threshold level.
    Used to approximate the 5 sigma level.

    :param TS: TS values
    :param level: Threshold level (e.g. 5.7e-7 for 5 sigma) 
    :param above: Fit above this value, defining the tail 
    """

    idx = np.where(~np.isnan(TS))
    
    values, bins = np.histogram(TS[idx], bins=50, density=True)

    cumsum = np.cumsum(values)
    cumulative = cumsum / max(cumsum)

    bin_c = bins[:-1] + np.diff(bins)/2

    out, cov = curve_fit(fit_func, bin_c[bin_c>above], 1-cumulative[bin_c>above])
    
    TS_thresh = fsolve(solve_func, x0=15, args=(out[0], out[1], level))[0]

    return TS_thresh, out, cov

    
def fit_func(x, a, b):
    
    return a * np.power(b, x)


def solve_func(x, a, b, level):

    return fit_func(x, a, b) - level
