"""
Create a mock NWB file with spyglass-compatible video data for testing purposes.

This file also contains epoch/task data, since it is necessary for the video data to be compatible with spyglass.
"""

from pynwb.testing.mock.file import mock_NWBFile
from pynwb import NWBHDF5IO
import numpy as np
from pathlib import Path
from ndx_franklab_novela import CameraDevice
from pynwb.image import ImageSeries
from pynwb.core import DynamicTable
from pynwb.behavior import BehavioralEvents

from jadhav_lab_to_nwb.olson_2024.tools.spikegadgets import readCameraModuleTimeStamps


def add_epoch(nwbfile):
    tasks_module = nwbfile.create_processing_module(name="tasks", description="tasks module")
    num_tasks = 2
    for i in range(1, num_tasks + 1):
        task_table = DynamicTable(name=f"task_table_{i}", description="my task table")
        task_table.add_column(name="task_name", description="Name of the task.")
        task_table.add_column(name="task_description", description="Description of the task.")
        task_table.add_column(name="camera_id", description="Camera ID.")
        task_table.add_column(name="task_epochs", description="Task epochs.")
        task_table.add_row(
            task_name=f"task{i}", task_description=f"task{i} description", camera_id=[1], task_epochs=[i]
        )
        tasks_module.add(task_table)

    nwbfile.add_epoch_column(name="custom_data_string", description="Custom epoch column")
    nwbfile.add_epoch(start_time=0.0, stop_time=1.0, tags=["01"], custom_data_string="custom_value1")
    nwbfile.add_epoch(start_time=268.0, stop_time=2110.0, tags=["02"], custom_data_string="custom_value2")


def add_video(nwbfile):
    camera_device = CameraDevice(
        name="camera_device 1",
        meters_per_pixel=1.0,
        model="my_model",
        lens="my_lens",
        camera_name="my_camera_name",
    )
    nwbfile.add_device(camera_device)

    video_file_path = "/Volumes/T7/CatalystNeuro/Jadhav/SubLearnProject/SL18_D19/SL18_D19_S01_F01_BOX_SLP_20230503_112642/SL18_D19_S01_F01_BOX_SLP_20230503_112642.1.h264"
    video_timestamps_file_path = "/Volumes/T7/CatalystNeuro/Jadhav/SubLearnProject/SL18_D19/SL18_D19_S01_F01_BOX_SLP_20230503_112642/SL18_D19_S01_F01_BOX_SLP_20230503_112642.1.videoTimeStamps"
    timestamps, _ = readCameraModuleTimeStamps(video_timestamps_file_path)
    image_series = ImageSeries(
        name="my_image_series",
        description="my_description",
        unit="n.a.",
        external_file=[video_file_path],
        format="external",
        timestamps=timestamps,
        device=camera_device,
    )

    behavior_module = nwbfile.create_processing_module(name="behavior", description="behavior module")
    behavioral_events = BehavioralEvents(name="video")
    behavioral_events.add_timeseries(image_series)
    behavior_module.add(behavioral_events)


def main():
    nwbfile = mock_NWBFile(identifier="my_identifier", session_description="my_session_description")
    add_epoch(nwbfile)
    add_video(nwbfile)

    nwbfile_path = Path("/Volumes/T7/CatalystNeuro/Spyglass/raw/mock_video.nwb")
    if nwbfile_path.exists():
        nwbfile_path.unlink()
    with NWBHDF5IO(nwbfile_path, "w") as io:
        io.write(nwbfile)
    print(f"mock video NWB file successfully written at {nwbfile_path}")


if __name__ == "__main__":
    main()
