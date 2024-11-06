from icecube_tools.utils.data import IceCubeData, Uptime, available_data_periods

my_data = IceCubeData()

"""
def test_data_scan():

    assert my_data.datasets[1] == "20080911_AMANDA_7_Year_Data.zip"
"""

def test_file_download(output_directory):

    found_dataset = ["20181018"]

    my_data.fetch(found_dataset, write_to=output_directory)


def test_uptime():
    uptime = Uptime(*available_data_periods)
    live_time = uptime._time_span("IC40")
    obs_time = uptime.cumulative_time_obs()
    assert obs_time["IC40"] <= live_time
