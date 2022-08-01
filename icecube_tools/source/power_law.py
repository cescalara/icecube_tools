import numpy as np
from scipy.stats import bernoulli, uniform


class BoundedPowerLaw(object):
    """
    Definition of a bounded power law distribution.
    
    pdf ~ x^(-alpha) betweem xmin and xmax.
    
    Thanks to H. Niederhausen (@HansN87) for this implementation.
    """

    def __init__(self, gamma, xmin, xmax):
        """
        Definition of a bounded power law distribution.

        pdf ~ x^(-alpha) betweem xmin and xmax.
        """
        self.gamma = gamma
        self.xmin = xmin
        self.xmax = xmax

        # calculate normalization and other useful terms
        if self.gamma != 1.0:
            self.int_gamma = 1.0 - self.gamma
            self.norm = (
                1.0
                / self.int_gamma
                * (self.xmax ** self.int_gamma - self.xmin ** self.int_gamma)
            )
            self.norm = 1.0 / self.norm

            self.cdf_factor = self.norm / self.int_gamma
            self.cdf_const = self.cdf_factor * (-self.xmin ** self.int_gamma)

            self.inv_cdf_factor = self.norm ** (-1) * self.int_gamma
            self.inv_cdf_const = self.xmin ** self.int_gamma
            self.inv_cdf_gamma = 1.0 / self.int_gamma

        else:
            self.norm = 1.0 / np.log(self.xmax / self.xmin)

    def pdf(self, x):
        """
        Evaluate the probability distribution function at x.
        """
        val = np.power(x, -self.gamma) * self.norm

        if not isinstance(x, np.ndarray):
            if x < self.xmin or x > self.xmax:
                return 0.0
            else:
                return val

        else:
            idx = np.logical_or(x < self.xmin, x > self.xmax)
            val[idx] = np.zeros(len(val[idx]))
            return val

    def cdf(self, x):
        """
        Evaluate the cumulative distribution function at x.
        """

        if self.gamma == 1:
            val = self.norm * np.log(x / self.xmin)
        else:
            val = self.cdf_factor * np.power(x, self.int_gamma) + self.cdf_const

        if not isinstance(x, np.ndarray):
            if x < self.xmin:
                return 0.0
            if x > self.xmax:
                return 1.0
            else:
                return val

        else:
            idx = x < self.xmin
            val[idx] = np.zeros(len(val[idx]))
            idx = x > self.xmax
            val[idx] = np.ones(len(val[idx]))
            return val

    def inv_cdf(self, x):
        """
        Evaluate the inverse cumulative distribution function at x.
        """
        if self.gamma == 1:
            return self.xmin * np.exp(x / self.norm)
        else:
            return np.power(
                (x * self.inv_cdf_factor) + self.inv_cdf_const, self.inv_cdf_gamma
            )

    def samples(self, nsamples):
        """
        Inverse CDF sample from the bounded power law distribution.
        """
        u = np.random.uniform(0, 1, size=nsamples)
        return self.inv_cdf(u)


class BrokenBoundedPowerLaw:
    """
    Sampling from a broken power law.

    Based on:
    https://github.com/grburgess/brokenpl_sample/blob/master/sample_broken_power_law.ipynb
    by J. M. Burgess (@grburgess).
    """

    def __init__(self, x0, x1, x2, gamma1, gamma2):
        """
        Sampling from a broken power law.
        Based on:
        https://github.com/grburgess/brokenpl_sample/blob/master/sample_broken_power_law.ipynb.
        
        :param x0: Lower bound
        :param x1: Break point
        :param x2: Upper bound
        :param gamma1: Index of first power law
        :param gamma2: Index of the second power law
        """

        self.x0 = x0

        self.x1 = x1

        self.x2 = x2

        self.gamma1 = gamma1

        self.gamma2 = gamma2

        w1, w2, total = self._integrate()

        self.norm = 1.0 / total

        self.weights = (w1, w2)

    def _integrate(self):
        """
        Compute the total integral and weights of each segment.
        """

        int_first_seg = (
            np.power(self.x1, self.gamma1 + 1.0) - np.power(self.x0, self.gamma1 + 1.0)
        ) / (self.gamma1 + 1.0)

        int_second_seg = (
            np.power(self.x1, self.gamma1 - self.gamma2)
            * (np.power(self.x2, self.gamma2 + 1) - np.power(self.x1, self.gamma2 + 1))
            / (self.gamma2 + 1)
        )

        total = int_first_seg + int_second_seg

        w1 = int_first_seg / total

        w2 = int_second_seg / total

        return w1, w2, total

    def samples(self, N):
        """
        Sample from the broken power law.

        :param N: number of samples.
        """

        u = np.random.uniform(0, 1, N)

        output = np.empty_like(u)

        idx = bernoulli.rvs(self.weights[0], size=len(u)).astype(bool)

        output[idx] = np.power(
            u[idx]
            * (
                np.power(self.x1, self.gamma1 + 1.0)
                - np.power(self.x0, self.gamma1 + 1.0)
            )
            + np.power(self.x0, self.gamma1 + 1.0),
            1.0 / (self.gamma1 + 1.0),
        )

        output[~idx] = np.power(
            u[~idx]
            * (
                np.power(self.x2, self.gamma2 + 1.0)
                - np.power(self.x1, self.gamma2 + 1.0)
            )
            + np.power(self.x1, self.gamma2 + 1.0),
            1.0 / (self.gamma2 + 1.0),
        )

        return output
