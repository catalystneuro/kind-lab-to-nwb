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

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata, conversion_options: Optional[dict] = None):
        for device_metadata in metadata["Devices"]:
            # Add the device to the NWB file
            device = Device(**device_metadata)
            nwbfile.add_device(device)
        super().add_to_nwbfile(nwbfile=nwbfile, metadata=metadata, conversion_options=conversion_options)

    def temporally_align_data_interfaces(
        self, metadata: Optional[dict] = None, conversion_options: Optional[dict] = None
    ):
        audio_interfaces = [interface for interface in self.data_interface_objects if "Audio" in interface]
        for audio_interface_name in audio_interfaces:
            audio_interface = self.data_interface_objects[audio_interface_name]
            if audio_interface._segment_starting_times is None:
                # Use np.nan to indicate that the aligned starting times are not known
                num_usv_file_paths = len(audio_interface.source_data["file_paths"])
                audio_interface._segment_starting_times = [np.nan] * num_usv_file_paths
