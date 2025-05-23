"""Primary NWBConverter class for this dataset."""

from typing import Optional

from pynwb import NWBFile
from pynwb.device import Device

from neuroconv import NWBConverter
from neuroconv.datainterfaces import ExternalVideoInterface

from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.interfaces import (
    BORISBehavioralEventsInterface,
)

from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.utils import parse_datetime_from_filename


class ObjectRecognitionNWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    data_interface_classes = dict(
        Video=ExternalVideoInterface,
        SampleVideo=ExternalVideoInterface,
        TestVideo=ExternalVideoInterface,
        TestObjectRecognitionBehavior=BORISBehavioralEventsInterface,
        SampleObjectRecognitionBehavior=BORISBehavioralEventsInterface,
    )

    def temporally_align_data_interfaces(self, metadata, conversion_options: Optional[dict] = None):
        if (
            "TestObjectRecognitionBehavior" in self.data_interface_objects
            and "SampleObjectRecognitionBehavior" in self.data_interface_objects
        ):
            video_file_path = self.data_interface_objects["SampleVideo"].source_data["file_paths"][0]
            sample_video_datetime = parse_datetime_from_filename(video_file_path.name)

            video_file_path = self.data_interface_objects["TestVideo"].source_data["file_paths"][0]
            test_video_datetime = parse_datetime_from_filename(video_file_path.name)

            # Align the start time of the test video to the start time of the sample video

            aligned_starting_time = (test_video_datetime - sample_video_datetime).total_seconds()
            self.data_interface_objects["TestObjectRecognitionBehavior"].set_aligned_starting_time(
                aligned_starting_time
            )
            self.data_interface_objects["TestVideo"].set_aligned_starting_time(aligned_starting_time)

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata, conversion_options: Optional[dict] = None):
        super().add_to_nwbfile(nwbfile, metadata, conversion_options)
        for device_metadata in metadata["Devices"]:
            # Add the device to the NWB file
            device = Device(**device_metadata)
            nwbfile.add_device(device)
