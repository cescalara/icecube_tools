from icecube_tools.detector.effective_area import (
    EffectiveArea,
    _supported_dataset_ids,
)


def test_aeff_load():

    for dataset_id in _supported_dataset_ids:

        _ = EffectiveArea.from_dataset(dataset_id)
