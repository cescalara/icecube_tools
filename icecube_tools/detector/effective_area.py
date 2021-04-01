import numpy as np
from abc import ABC, abstractmethod

from icecube_tools.utils.data import (
    IceCubeData,
    find_files,
    data_directory,
)

"""
Module for working with the public IceCube
effective area information.
"""

R2015_AEFF_FILENAME = "effective_area.h5"
R2018_AEFF_FILENAME = "TabulatedAeff.txt"
BRAUN2008_AEFF_FILENAME = "AeffBraun2008.csv"

_supported_dataset_ids = ["20150820", "20181018"]


class IceCubeAeffReader(ABC):
    """
    Abstract base class for a file reader to handle
    the different types of data files provided
    on the IceCube website.
    """

    def __init__(self, filename):
        """
        Abstract base class for a file reader to handle
        the different types of data files provided
        on the IceCube website.

        :param filename: name of the file to be read from (string).
        """

        self._filename = filename

        self.effective_area_values = None

        self.true_energy_bins = None

        self.cos_zenith_bins = None

        self._label_order = {"true_energy": 0, "cos_zenith": 1}

        self._units = {"effective_area": "m^2", "true_energy": "GeV", "cos_zenith": ""}

        self.read()

        super().__init__()

    @abstractmethod
    def read(self):
        """
        To be defined.
        """

        pass


class R2015AeffReader(IceCubeAeffReader):
    """
    Reader for the 2015 Aug 20 release.
    Link: https://icecube.wisc.edu/science/data/HE_NuMu_diffuse.
    """

    def __init__(self, filename, **kwargs):

        if "year" in kwargs:
            self.year = kwargs["year"]
        else:
            self.year = 2011

        if "nu_type" in kwargs:
            self.nu_type = kwargs["nu_type"]
        else:
            self.nu_type = "nu_mu"

        super().__init__(filename)

        self._label_order["reco_energy"] = 2

        self._units["reco_energy"] = "GeV"

    def read(self):
        """
        Read input from the provided HDF5 file.
        """

        import h5py

        with h5py.File(self._filename, "r") as f:

            directory = f[str(self.year) + "/" + self.nu_type + "/"]

            self.effective_area_values = directory["area"][()]

            self.true_energy_bins = directory["bin_edges_0"][()]

            self.cos_zenith_bins = directory["bin_edges_1"][()]

            self.reco_energy_bins = directory["bin_edges_2"][()]


class R2018AeffReader(IceCubeAeffReader):
    """
    Reader for the 2018 Oct 18 release.
    Link: https://icecube.wisc.edu/science/data/PS-3years.
    """

    def read(self):

        import pandas as pd

        self.year = int(self._filename[-22:-18])
        self.nu_type = "nu_mu"

        filelayout = ["Emin", "Emax", "cos(z)min", "cos(z)max", "Aeff"]
        output = pd.read_csv(
            self._filename, comment="#", delim_whitespace=True, names=filelayout
        ).to_dict()

        true_energy_lower = set(output["Emin"].values())
        true_energy_upper = set(output["Emax"].values())

        cos_zenith_lower = set(output["cos(z)min"].values())
        cos_zenith_upper = set(output["cos(z)max"].values())

        self.true_energy_bins = np.array(
            list(true_energy_upper.union(true_energy_lower))
        )
        self.true_energy_bins.sort()

        self.cos_zenith_bins = np.array(list(cos_zenith_upper.union(cos_zenith_lower)))
        self.cos_zenith_bins.sort()

        self.effective_area_values = np.reshape(
            list(output["Aeff"].values()),
            (len(true_energy_lower), len(cos_zenith_lower)),
        )


class Braun2008AeffReader(IceCubeAeffReader):
    """
    Reader for the Braun+2008 paper effective area
    Fig. 3, 140-150 deg.
    Braun, J. et al., 2008. Methods for point source
    analysis in high energy neutrino telescopes.
    Astroparticle Physics, 29(4), pp.299–305.
    """

    def read(self):

        self.nu_type = "nu_mu"

        out = np.loadtxt(self._filename, delimiter=",", comments="#")

        # Assume whole sky like this
        self.cos_zenith_bins = np.array([-1, 1])

        self.true_energy_bins = out.T[0]

        aeff = out.T[1]

        self.effective_area_values = aeff[:-1] + np.diff(aeff) / 2


class EffectiveArea:
    """
    IceCube effective area.
    """

    def __init__(self, filename, **kwargs):
        """
        IceCube effective area.

        :param filename: name of the file to be read from (string).
        :param kwargs: kwargs to be passed to reader if relevant.
        """

        self._filename = filename

        self._reader = self.get_reader(**kwargs)

        self.values = self._reader.effective_area_values

        self.true_energy_bins = self._reader.true_energy_bins

        self.cos_zenith_bins = self._reader.cos_zenith_bins

        self._integrate_out_ancillary_params()

    def get_reader(self, **kwargs):
        """
        Define an IceCubeAeffReader based on the filename.
        """

        if R2015_AEFF_FILENAME in self._filename:

            return R2015AeffReader(self._filename, **kwargs)

        elif R2018_AEFF_FILENAME in self._filename:

            return R2018AeffReader(self._filename)

        elif BRAUN2008_AEFF_FILENAME in self._filename:

            return Braun2008AeffReader(self._filename)

        else:

            raise ValueError(
                self._filename
                + " is not recognised as one of the known effective area files."
            )

    def _integrate_out_ancillary_params(self):
        """
        Sometimes the effective area is given as a
        function of ancillary parameters, e.g. the
        reconstructed muon energy. To give a unified
        interface, these can be integrated over.
        """

        if len(np.shape(self.values)) > 2:

            dim_to_int = []

            for key in self._reader._label_order:

                if "true_energy" not in key and "cos_zenith" not in key:

                    dim_to_int.append(self._reader._label_order[key])

            self.values = np.sum(self.values, axis=tuple(dim_to_int))

    def detection_probability(self, true_energy, true_cos_zenith, max_energy):
        """
        Give the relative detection probability for
        a given true energy and arrival direction.
        """

        scaled_values = self.values.copy()

        lower_bin_edges = self.true_energy_bins[:-1]
        scaled_values[lower_bin_edges > max_energy] = 0

        scaled_values = scaled_values / np.max(scaled_values)

        energy_index = np.digitize(true_energy, self.true_energy_bins) - 1

        if len(self.cos_zenith_bins) > 2:

            cosz_index = np.digitize(true_cos_zenith, self.cos_zenith_bins) - 1

            return scaled_values[energy_index][cosz_index]

        else:

            return scaled_values[energy_index]

    @classmethod
    def from_dataset(cls, dataset_id, fetch=True, **kwargs):
        """
        Build effective area from a public dataset.

        If relevant, uses the latest effective area
        within a dataset. Some datasets have multiple
        effective areas.

        :param dataset_id: Date of dataset release e.g. 20181018
        :param fetch: If true, download dataset if not existing
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

        # Find filename
        if dataset_id == "20181018":

            files = find_files(dataset_dir, R2018_AEFF_FILENAME)

            # Latest dataset
            aeff_file_name = files[1]

        elif dataset_id == "20150820":

            files = find_files(dataset_dir, R2015_AEFF_FILENAME)

            # Latest dataset
            aeff_file_name = files[0]

        return cls(aeff_file_name, **kwargs)
