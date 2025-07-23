"""Primary NWBConverter class for this dataset."""

import numpy as np
from typing import Optional
from datetime import timedelta
from pynwb import NWBFile
from pynwb.device import Device

from neuroconv import NWBConverter

from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.interfaces import (
    BORISBehavioralEventsInterface,
    SpyglassVideoInterface,
)


class MarbleInteractionNWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    data_interface_classes = dict(
        MarbleInteractionBehavior=BORISBehavioralEventsInterface,
        Video=SpyglassVideoInterface,
    )

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata, conversion_options: Optional[dict] = None):
        super().add_to_nwbfile(nwbfile, metadata, conversion_options)
        for device_metadata in metadata["Devices"]:
            # Add the device to the NWB file
            device = Device(**device_metadata)
            nwbfile.add_device(device)
