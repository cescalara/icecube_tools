from abc import ABC

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


class DiffuseSource(Source):
    """
    A diffuse source. It is assumed to be isotropic
    over the full 4pi sky and has a spectrum described 
    by its flux model.
    """

    def __init__(self, flux_model):
        """
        A diffuse source. It is assumed to be isotropic
        over the full 4pi sky and has a spectrum described 
        by its flux model.
            
        :param flux_model: A FluxModel object. 
        """

        super().__init__()

        self.source_type = DIFFUSE

        self.flux_model = flux_model


class PointSource(Source):
    """
    A point source is localised to a point 
    on the sky and has a spectrum described 
    by its flux model.
    """

    def __init__(self, flux_model, coord):
        """
        A point source is localised to a point 
        on the sky and has a spectrum described 
        by its flux model.
        
        :param flux_model: A FluxModel object.
        :param coordinate: (ra, dec) coord.
        """

        super().__init__()

        self.source_type = POINT

        self.flux_model = flux_model

        self._coord = coord

    @property
    def coord(self):

        return self._coord

    @coord.setter
    def coord(self, value):

        self._coord = value
