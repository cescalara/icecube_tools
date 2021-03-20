from icecube_tools.utils.data import IceCubeData

my_data = IceCubeData()


def test_data_scan():

    assert my_data.datasets[0] == "20080911_AMANDA_7_Year_Data.zip"


def test_file_download(output_directory):

    found_dataset = my_data.find("AMANDA")

    my_data.fetch(found_dataset, write_to=output_directory)
