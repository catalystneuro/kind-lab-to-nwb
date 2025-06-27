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

    def temporally_align_data_interfaces(self, metadata: dict, conversion_options: dict | None = None):
        video_interface = self.data_interface_objects["Video"]
        video_timestamps = video_interface.get_timestamps()
        video_timestamps = np.concatenate(video_timestamps)
        if video_timestamps[0] < 0:
            shift_value = abs(video_timestamps[0])
            video_interface.set_aligned_timestamps(aligned_timestamps=video_timestamps + shift_value)
            if "MarbleInteractionBehavior" in self.data_interface_objects:
                boris_interface = self.data_interface_objects["MarbleInteractionBehavior"]
                boris_interface.set_aligned_starting_time(shift_value)

        session_start_time = metadata["NWBFile"]["session_start_time"]
        metadata["NWBFile"].update(session_start_time=session_start_time + timedelta(seconds=shift_value))

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata, conversion_options: Optional[dict] = None):
        super().add_to_nwbfile(nwbfile, metadata, conversion_options)
        for device_metadata in metadata["Devices"]:
            # Add the device to the NWB file
            device = Device(**device_metadata)
            nwbfile.add_device(device)
