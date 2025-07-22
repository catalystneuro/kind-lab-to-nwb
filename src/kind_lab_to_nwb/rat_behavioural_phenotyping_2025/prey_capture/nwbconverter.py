"""Primary NWBConverter class for this dataset."""
from datetime import timedelta
from typing import Optional
from warnings import warn

import numpy as np
from pynwb import NWBFile
from pynwb.device import Device

from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.utils import (
    parse_datetime_from_filename,
)
from neuroconv import ConverterPipe


class PreyCaptureNWBConverter(ConverterPipe):
    """ConverterPipe for the prey capture dataset."""

    def temporally_align_data_interfaces(self, metadata, conversion_options: Optional[dict] = None):
        video_interfaces = [
            interface_name for interface_name in self.data_interface_objects if "Video" in interface_name
        ]
        session_start_time = metadata["NWBFile"]["session_start_time"]
        for interface_name in video_interfaces:
            video_interface = self.data_interface_objects[interface_name]
            video_file_path = video_interface.source_data["file_paths"][0]
            datetime_from_filename = parse_datetime_from_filename(video_file_path.name)
            if datetime_from_filename.tzinfo is None and session_start_time.tzinfo is not None:
                datetime_from_filename = datetime_from_filename.replace(tzinfo=session_start_time.tzinfo)
            aligned_starting_time = (datetime_from_filename - session_start_time).total_seconds()
            video_timestamps = video_interface.get_timestamps()
            video_timestamps = np.concatenate(video_timestamps)
            if video_timestamps[0] < 0:
                if aligned_starting_time == 0.0:
                    # push the session start time forward to align with the video timestamps
                    metadata["NWBFile"].update(
                        session_start_time=session_start_time + timedelta(seconds=abs(video_timestamps[0])),
                    )
                aligned_starting_time = abs(video_timestamps[0]) + aligned_starting_time

            video_interface.set_aligned_timestamps(aligned_timestamps=video_timestamps + aligned_starting_time)

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata, conversion_options: Optional[dict] = None):
        for device_metadata in metadata["Devices"]:
            # Add the device to the NWB file
            device = Device(**device_metadata)
            nwbfile.add_device(device)
        super().add_to_nwbfile(nwbfile=nwbfile, metadata=metadata, conversion_options=conversion_options)
