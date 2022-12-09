from icecube_tools.utils.data import IceCubeData, Uptime

my_data = IceCubeData()


def test_data_scan():

    assert my_data.datasets[1] == "20080911_AMANDA_7_Year_Data.zip"


def test_file_download(output_directory):

    found_dataset = my_data.find("AMANDA")

    my_data.fetch(found_dataset, write_to=output_directory)


def test_uptime():
    uptime = Uptime()
    live_time = uptime.time_span("IC40")
    obs_time = uptime.time_obs("IC40")
    assert obs_time["IC40"] <= live_time["IC40"]
