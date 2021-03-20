import numpy as np
import os
import requests
import requests_cache
import time
from zipfile import ZipFile
from bs4 import BeautifulSoup
from tqdm import tqdm

icecube_data_base_url = "https://icecube.wisc.edu/data-releases"
data_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


class IceCubeData:
    """
    Handle the interface with IceCube's public data
    releases hosted on their website.
    """

    def __init__(
        self,
        base_url=icecube_data_base_url,
        data_directory=data_directory,
        cache_name=".cache",
        update=False,
    ):
        """
        Handle the interface with IceCube's public data
        releases hosted on their website.

        :param base_url: Base url for data releases
        :param data_directory: Where to put the data
        :param cache_name: Name of the requests cache
        :param update: Refresh the cache if true
        """

        self.base_url = base_url

        self.data_directory = data_directory

        requests_cache.install_cache(cache_name=cache_name)

        self.ls(verbose=False, update=update)

    def ls(self, verbose=True, update=False):
        """
        List the available datasets.

        :param verbose: Print the datasets if true
        :param update: Refresh the cache if true
        """

        self.datasets = []

        if update:

            requests_cache.clear()

        response = requests.get(self.base_url)

        if response.ok:

            soup = BeautifulSoup(response.content, "html.parser")

            links = soup.find_all("a")

            for link in links:

                href = link.get("href")

                if ".zip" in href:

                    self.datasets.append(href)

                    if verbose:
                        print(href)

    def find(self, search_string):
        """
        Find datasets containing search_string.
        """

        found_datasets = []

        for dataset in self.datasets:

            if search_string in dataset:

                found_datasets.append(dataset)

        return found_datasets

    def fetch(self, datasets, overwrite=False, write_to=None):
        """
        Downloads and unzips the given datasets.

        :param datasets: A list of dataset names
        :param overwrite: Overwrite existing files
        :param write_to: Optional custom location
        """

        if write_to:

            old_dir = self.data_directory

            self.data_directory = write_to

        for dataset in datasets:

            if dataset not in self.datasets:

                raise ValueError(
                    "Dataset %s is not in list of known datasets" % dataset
                )

            url = os.path.join(self.base_url, dataset)

            local_path = os.path.join(self.data_directory, dataset)

            # Only fetch if not already there!
            if not os.path.exists(os.path.splitext(local_path)[0]) or overwrite:

                # Don't cache this as we want to stream
                with requests_cache.disabled():

                    response = requests.get(url, stream=True)

                    if response.ok:

                        total = int(response.headers["content-length"])

                        # For progress bar description
                        short_name = dataset
                        if len(dataset) > 40:
                            short_name = dataset[0:40] + "..."

                        # Save locally
                        with open(local_path, "wb") as f, tqdm(
                            desc=short_name, total=total
                        ) as bar:

                            for chunk in response.iter_content(chunk_size=1024 * 1024):

                                size = f.write(chunk)
                                bar.update(size)

                        # Unzip
                        with ZipFile(local_path, "r") as zip_ref:

                            zip_ref.extractall(os.path.splitext(local_path)[0])

                        # Delete zipfile
                        os.remove(local_path)

                crawl_delay()

        if write_to:

            self.data_directory = old_dir

    def fetch_all_to(self, write_to, overwrite=False):
        """
        Download all data to a given location
        """

        self.fetch(self.datasets, write_to=write_to, overwrite=overwrite)


def crawl_delay():
    """
    Delay between sending HTML requests.
    """

    time.sleep(np.random.uniform(5, 10))
