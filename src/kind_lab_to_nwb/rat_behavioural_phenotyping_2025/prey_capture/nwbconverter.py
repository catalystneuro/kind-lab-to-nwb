"""Primary NWBConverter class for this dataset."""
from typing import Optional

from pynwb import NWBFile
from pynwb.device import Device

from neuroconv import ConverterPipe


class PreyCaptureNWBConverter(ConverterPipe):
    """ConverterPipe for the prey capture dataset."""

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata, conversion_options: Optional[dict] = None):
        for device_metadata in metadata["Devices"]:
            # Add the device to the NWB file
            device = Device(**device_metadata)
            nwbfile.add_device(device)
        super().add_to_nwbfile(nwbfile=nwbfile, metadata=metadata, conversion_options=conversion_options)
