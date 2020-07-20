from abc import ABC, abstractmethod
from scipy.stats import norm


"""
Prior module.
"""


class Prior(ABC):
    """
    Abstract base class for priors.
    """

    @abstractmethod
    def __call__(self):

        pass


class GaussianPrior(Prior):
    def __init__(self, mu, sigma):

        self._mu = mu

        self._sigma = sigma

    def __call__(self, value):

        return norm(loc=self._mu, scale=self._sigma).pdf(value)
