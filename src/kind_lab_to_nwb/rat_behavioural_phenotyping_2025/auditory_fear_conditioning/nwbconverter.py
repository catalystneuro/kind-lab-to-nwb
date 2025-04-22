"""Primary NWBConverter class for this dataset."""
from typing import Optional

from pynwb import NWBFile
from pynwb.device import Device

from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.auditory_fear_conditioning.behaviordatainterface import (
    AuditoryFearConditioningBehavioralInterface,
)
from neuroconv import NWBConverter
from neuroconv.datainterfaces import VideoInterface


class AuditoryFearConditioningNWBConverter(NWBConverter):
    """NWBConverter for the auditory fear conditioning dataset."""

    data_interface_classes = dict(
        Video=VideoInterface,
        Behavior=AuditoryFearConditioningBehavioralInterface,
    )

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata, conversion_options: Optional[dict] = None):
        for device_metadata in metadata["Devices"]:
            # Add the device to the NWB file
            device = Device(**device_metadata)
            nwbfile.add_device(device)
        super().add_to_nwbfile(nwbfile=nwbfile, metadata=metadata, conversion_options=conversion_options)
