import numpy as np
from abc import ABC, abstractmethod

from .power_law import BoundedPowerLaw
from .power_law import BrokenBoundedPowerLaw

"""
Module for simple flux models used in 
neutrino detection calculations
"""


class FluxModel(ABC):
    """
    Abstract base class for flux models.
    """

    @abstractmethod
    def spectrum(self):

        pass

    @abstractmethod
    def integrated_spectrum(self):

        pass

    @abstractmethod
    def redshift_factor(self, z):

        pass


class PowerLawFlux(FluxModel):
    """
    Power law flux models.
    """

    def __init__(
        self,
        normalisation,
        normalisation_energy,
        index,
        lower_energy=1e2,
        upper_energy=np.inf,
    ):
        """
        Power law flux models.

        :param normalisation: Flux normalisation [GeV^-1 cm^-2 s^-1 sr^-1] or [GeV^-1 cm^-2 s^-1] for point sources.
        :param normalisation energy: Energy at which flux is normalised [GeV].
        :param index: Spectral index of the power law.
        :param lower_energy: Lower energy bound [GeV].
        :param upper_energy: Upper enegry bound [GeV], unbounded by default.
        """

        super().__init__()

        self._normalisation = normalisation

        self._normalisation_energy = normalisation_energy

        self._index = index

        self._lower_energy = lower_energy
        self._upper_energy = upper_energy

        self.power_law = BoundedPowerLaw(
            self._index, self._lower_energy, self._upper_energy
        )

    def spectrum(self, energy):
        """
        dN/dEdAdt or dN/dEdAdtdO depending on flux_type.
        """

        if (energy < self._lower_energy) or (energy > self._upper_energy):

            return np.nan

        else:

            return self._normalisation * np.power(
                energy / self._normalisation_energy, -self._index
            )

    def integrated_spectrum(self, lower_energy_bound, upper_energy_bound):
        """
        \int spectrum dE over finite energy bounds.

        :param lower_energy_bound: [GeV]
        :param upper_energy_bound: [GeV]
        """

        """
        if lower_energy_bound < self._lower_energy and upper_energy_bound < self._lower_energy:
            return 0
        elif lower_energy_bound < self._lower_energy and upper_energy_bound > self._lower_energy:
            lower_energy_bound = self._lower_energy

        if upper_energy_bound > self._upper_energy and lower_energy_bound > self._upper_energy:
            return 0
        elif upper_energy_bound > self._upper_energy and lower_energy_bound < self._upper_energy:
            upper_energy_bound = self._upper_energy
        """

        norm = self._normalisation / (
            np.power(self._normalisation_energy, -self._index) * (1 - self._index)
        )

        return norm * (
            np.power(upper_energy_bound, 1 - self._index)
            - np.power(lower_energy_bound, 1 - self._index)
        )

    def total_flux_density(self):
        """
        Total flux density in units of
        [GeV cm^-2 s^-1]
        """

        norm = self._normalisation
        index = self._index
        lower, upper = self._lower_energy, self._upper_energy

        if index == 2:
            # special case
            int_norm = norm / (np.power(self._normalisation_energy, -index))
            return int_norm * (np.log(upper / lower))

        int_norm = norm / (np.power(self._normalisation_energy, -index) * (2 - index))
        return int_norm * (np.power(upper, 2 - index) - np.power(lower, 2 - index))

    def sample(self, N):
        """
        Sample energies from the power law.
        Uses inverse transform sampling.

        :param min_energy: Minimum energy to sample from [GeV].
        :param N: Number of samples.
        """

        return self.power_law.samples(N)

    def _rejection_sample(self, min_energy):
        """
        Sample energies from the power law.
        Uses rejection sampling.

        :param min_energy: Minimum energy to sample from [GeV].
        """

        dist_upper_lim = self.spectrum(min_energy)

        accepted = False

        while not accepted:

            energy = np.random.uniform(min_energy, 1e3 * min_energy)
            dist = np.random.uniform(0, dist_upper_lim)

            if dist < self.spectrum(energy):

                accepted = True

        return energy

    def redshift_factor(self, z: float):

        return np.power(1 + z, 1 - self._index)


class BrokenPowerLawFlux(FluxModel):
    """
    Broken power law flux models.
    """

    def __init__(
        self,
        normalisation,
        break_energy,
        index1,
        index2,
        lower_energy=1e2,
        upper_energy=1e8,
    ):
        """
        Broken power law flux models.

        :param normalisation: Flux normalisation [GeV^-1 cm^-2 s^-1 sr^-1] or [GeV^-1 cm^-2 s^-1] for point sources.
        :param break_energy: Energy at which PL is broken and flux is normalised [GeV].
        :param index1: Index of the lower energy power law.
        :param index2: Index of the upper energy power law.
        :param lower_energy: Lower energy over which model is defined [GeV].
        :param upper_energy: Upper energy over which model is defined [GeV].
        """

        super().__init__()

        self._normalisation = normalisation

        self._break_energy = break_energy

        self._index1 = -index1

        self._index2 = -index2

        self._lower_energy = lower_energy

        self._upper_energy = upper_energy

        self.power_law = BrokenBoundedPowerLaw(
            lower_energy, break_energy, upper_energy, -index1, -index2
        )

    def spectrum(self, energy):
        """
        dN/dEdAdt or dN/dEdAdtdO depending on flux_type.
        """

        if (energy < self._lower_energy) or (energy > self._upper_energy):

            return np.nan

        else:

            norm = self._normalisation

            if energy < self._break_energy:

                output = norm * np.power(energy / self._break_energy, self._index1)

            elif energy == self._break_energy:

                output = norm

            elif energy > self._break_energy:

                output = norm * np.power(energy / self._break_energy, self._index2)

            return output

    def integrated_spectrum(self, lower_energy_bound, upper_energy_bound):
        """
        \int spectrum dE over finite energy bounds.

        :param lower_energy_bound: [GeV]
        :param upper_energy_bound: [GeV]
        """

        norm = self._normalisation

        if (lower_energy_bound < self._break_energy) and (
            upper_energy_bound <= self._break_energy
        ):

            # Integrate over lower segment
            output = (
                norm
                * (
                    np.power(upper_energy_bound, self._index1 + 1.0)
                    - np.power(lower_energy_bound, self._index1 + 1.0)
                )
                / ((self._index1 + 1.0) * np.power(self._break_energy, self._index1))
            )

        elif (lower_energy_bound < self._break_energy) and (
            upper_energy_bound > self._break_energy
        ):

            # Integrate across break
            lower = (
                np.power(self._break_energy, self._index1 + 1.0)
                - np.power(lower_energy_bound, self._index1 + 1.0)
            ) / ((self._index1 + 1.0) * np.power(self._break_energy, self._index1))

            upper = (
                np.power(upper_energy_bound, self._index2 + 1.0)
                - np.power(self._break_energy, self._index2 + 1.0)
            ) / ((self._index2 + 1.0) * np.power(self._break_energy, self._index2))

            output = norm * (lower + upper)

        elif (lower_energy_bound >= self._break_energy) and (
            upper_energy_bound > self._break_energy
        ):

            # Integrate over upper segment
            upper = (
                np.power(upper_energy_bound, self._index2 + 1.0)
                - np.power(lower_energy_bound, self._index2 + 1.0)
            ) / ((self._index2 + 1.0) * np.power(self._break_energy, self._index2))

            output = norm * upper

        return output

    def redshift_factor(self, z: float):

        raise NotImplementedError()

    def sample(self, N):
        """
        Sample energies from the power law.
        Uses inverse transform sampling.

        :param N: Number of samples.
        """

        return self.power_law.samples(N)
