from abc import ABC
from typing import Tuple

from .flux_model import FluxModel

"""
Module for simple neutirno source models. 
"""

DIFFUSE = 0
POINT = 1


class Source(ABC):
    """
    Abstract base class for neutrino sources.
    """

    def __init__(self, flux_model: FluxModel, z: float = 0.0):
        """
        :param flux_model: Shape of spectrum
        :param z: Redshift
        """

        self._flux_model = flux_model

        self._z = z

    @property
    def source_type(self):

        return self._source_type

    @source_type.setter
    def source_type(self, value):

        if value is not DIFFUSE and value is not POINT:

            raise ValueError(str(value) + " is not a recognised flux type")

        else:

            self._source_type = value

    @property
    def flux_model(self):

        return self._flux_model

    @flux_model.setter
    def flux_model(self, value):

        if not isinstance(value, FluxModel):

            raise ValueError(str(value) + " is not a recognised flux model")

        else:

            self._flux_model = value

    @property
    def z(self):

        return self._z


class DiffuseSource(Source):
    """
    A diffuse source. It is assumed to be isotropic
    over the full 4pi sky and has a spectrum described
    by its flux model.
    """

    def __init__(self, flux_model: FluxModel, z: float = 0.0):
        """
        A diffuse source. It is assumed to be isotropic
        over the full 4pi sky and has a spectrum described
        by its flux model.
        """

        super().__init__(flux_model=flux_model, z=z)

        self.source_type = DIFFUSE


class PointSource(Source):
    """
    A point source is localised to a point
    on the sky and has a spectrum described
    by its flux model.
    """

    def __init__(
        self,
        flux_model: FluxModel,
        z: float = 0.0,
        coord: Tuple[float, float] = (0.0, 0.0),
    ):
        """
        A point source is localised to a point
        on the sky and has a spectrum described
        by its flux model.

        :param coordinate: (ra, dec) coord.
        """

        super().__init__(flux_model=flux_model, z=z)

        self.source_type = POINT

        self._coord = coord

    @property
    def coord(self):

        return self._coord

    @coord.setter
    def coord(self, value: Tuple[float, float]):

        self._coord = value
