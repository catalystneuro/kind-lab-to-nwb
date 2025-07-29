"""Primary NWBConverter class for this dataset."""

from typing import Optional

import numpy as np

from pynwb import NWBFile
from pynwb.device import Device

from neuroconv import NWBConverter

from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.interfaces import (
    BORISBehavioralEventsInterface,
    SpyglassVideoInterface,
)

from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.utils import parse_datetime_from_filename


class ObjectRecognitionNWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    data_interface_classes = dict(
        Video=SpyglassVideoInterface,
        SampleVideo=SpyglassVideoInterface,
        TestVideo=SpyglassVideoInterface,
        TestObjectRecognitionBehavior=BORISBehavioralEventsInterface,
        SampleObjectRecognitionBehavior=BORISBehavioralEventsInterface,
    )

    def temporally_align_data_interfaces(self, metadata, conversion_options: Optional[dict] = None):
        """
        Aligns the start times of the test and sample videos based on their filenames.

        This method is invoked during the data conversion process to ensure temporal alignment
        between the behavioral and video data interfaces. It calculates the time difference
        between the sample and test video start times (extracted from their filenames) and
        adjusts the starting times of the corresponding data interfaces accordingly.

        Parameters:
        ----------
        metadata : dict
            Metadata dictionary containing information about the dataset.
        conversion_options : dict, optional
            Additional options for the conversion process.
        """
        if "TestVideo" in self.data_interface_objects and "SampleVideo" in self.data_interface_objects:
            video_file_path = self.data_interface_objects["SampleVideo"].source_data["file_paths"][0]
            sample_video_datetime = parse_datetime_from_filename(video_file_path.name)

            video_file_path = self.data_interface_objects["TestVideo"].source_data["file_paths"][0]
            test_video_datetime = parse_datetime_from_filename(video_file_path.name)

            # Align the start time of the test video to the start time of the sample video

            aligned_starting_time = (test_video_datetime - sample_video_datetime).total_seconds()
            if "STM" in metadata["NWBFile"]["session_id"] and aligned_starting_time == 0:
                # If the session is STM and the aligned starting time is 0, set it to a default value of 5 minutes
                aligned_starting_time = 20 * 60  # 15 min of sample video + 5 min of pause as described in the protocol

            if "TestObjectRecognitionBehavior" in self.data_interface_objects:
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
        if (
            "TestObjectRecognitionBehavior" in self.data_interface_objects
            and "SampleObjectRecognitionBehavior" in self.data_interface_objects
            and "NoveltyInformation" in metadata
        ):
            test_trial_info = metadata["NoveltyInformation"]["test_trial"]
            test_trial_events_table_name = conversion_options["TestObjectRecognitionBehavior"]["table_name"]
            test_trial_events_table = nwbfile.processing["behavior"][test_trial_events_table_name].to_dataframe()
            # add columns for object names, novelty, and positions
            object_names = []
            novelty_info = []
            positions = []
            for idx, row in test_trial_events_table.iterrows():
                for i, boris_label in enumerate(test_trial_info["boris_label"]):
                    if boris_label == row["label"]:
                        object_names.append(test_trial_info["object"][i])
                        novelty_info.append(test_trial_info["novelty"][i])
                        positions.append(test_trial_info["position"][i])

            nwbfile.processing["behavior"][test_trial_events_table_name].add_column(
                name="object_name",
                description="Name of the object in the test trial",
                data=object_names,
            )
            nwbfile.processing["behavior"][test_trial_events_table_name].add_column(
                name="novelty",
                description="Novelty of the object in the test trial",
                data=novelty_info,
            )
            nwbfile.processing["behavior"][test_trial_events_table_name].add_column(
                name="position",
                description="Position of the object in the test trial",
                data=positions,
            )

            sample_trial_info = metadata["NoveltyInformation"]["sample_trial"]
            sample_trial_events_table_name = conversion_options["SampleObjectRecognitionBehavior"]["table_name"]
            sample_trial_events_table = nwbfile.processing["behavior"][sample_trial_events_table_name].to_dataframe()
            # add columns for object names, novelty, and positions
            object_names = []
            novelty_info = []
            positions = []
            for idx, row in sample_trial_events_table.iterrows():
                for i, boris_label in enumerate(sample_trial_info["boris_label"]):
                    if boris_label == row["label"]:
                        object_names.append(sample_trial_info["object"][i])
                        novelty_info.append(sample_trial_info["novelty"][i])
                        positions.append(sample_trial_info["position"][i])

            nwbfile.processing["behavior"][sample_trial_events_table_name].add_column(
                name="object_name",
                description="Name of the object in the sample trial",
                data=object_names,
            )
            nwbfile.processing["behavior"][sample_trial_events_table_name].add_column(
                name="novelty",
                description="Novelty of the object in the sample trial",
                data=novelty_info,
            )
            nwbfile.processing["behavior"][sample_trial_events_table_name].add_column(
                name="position",
                description="Position of the object in the sample trial",
                data=positions,
            )
