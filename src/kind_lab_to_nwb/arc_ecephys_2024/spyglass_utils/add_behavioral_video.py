"""
Create a mock NWB file with spyglass-compatible video data for testing purposes.

This file also contains epoch/task data, since it is necessary for the video data to be compatible with spyglass.
"""

from typing import Optional, Union
from pathlib import Path
import numpy as np

from pynwb import NWBFile
from pynwb.core import DynamicTable
from pynwb.image import ImageSeries
from pynwb.behavior import BehavioralEvents

from ndx_franklab_novela import CameraDevice
from neuroconv.datainterfaces.behavior.video.video_utils import get_video_timestamps


def add_behavioral_video(
    nwbfile: NWBFile,
    metadata,
    video_file_path: Union[Path, str],
    task_metadata: dict,
    alligned_timestamps: Optional[list[np.ndarray]] = None,
) -> None:

    camera_device = CameraDevice(**metadata["Devices"]["CameraDevice"])
    nwbfile.add_device(camera_device)

    # Add a custom processing module for tasks
    # This is necessary for the video data to be compatible with spyglass.
    tasks_module = nwbfile.create_processing_module(name="tasks", description="tasks module")

    task_table = DynamicTable(
        name="task_table", description="The task table is needed for the video data to be compatible with spyglass. "
    )
    task_table.add_column(name="task_name", description="Name of the task.")
    task_table.add_column(name="task_description", description="Description of the task.")
    task_table.add_column(name="camera_id", description="Camera ID.")
    task_table.add_column(name="task_epochs", description="Task epochs.")
    task_table.add_column(name="environment", description="Environment where the task is carried out.")
    task_table.add_row(
        task_name=task_metadata["name"],
        task_description=task_metadata["session_description"],
        camera_id=[task_metadata["camera_id"]],
        task_epochs=[task_metadata["task_epochs"]],
        environment=task_metadata["environment"],
    )
    tasks_module.add(task_table)

    if alligned_timestamps is not None:
        timestamps = alligned_timestamps
    else:
        timestamps = get_video_timestamps(file_path=video_file_path)

    nwbfile.add_epoch_column(name="task_name", description="Name of the task associated with the epoch.")
    nwbfile.add_epoch(start_time=timestamps[0], stop_time=timestamps[-1], task_name=task_metadata["name"])

    image_series = ImageSeries(
        name=f"Video {Path(video_file_path).stem}",
        description="Video recorded by camera.",
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
