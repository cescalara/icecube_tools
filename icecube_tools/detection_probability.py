import numpy as np
from scipy.stats import poisson
import h5py

"""
Functions for calculating the detection probabilty 
from the results of point source analysis.
"""


def get_detection_probability(filename, index, TS_threshold):
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

        P = len(TS[i][TS[i] > TS_threshold]) / len(TS[i])

        Pdet_at_Nsrc.append(P)
    
    # Weight by poisson probability

    Pdet = []
    for Nsrc in Nsrc_list:

        P = sum([w * poisson(Nsrc).pmf(i) for i, w in enumerate(Pdet_at_Nsrc)])

        Pdet.append(P)

    return Nsrc_list, Pdet
