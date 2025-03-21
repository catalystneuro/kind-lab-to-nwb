"""
Create a mock NWB file with spyglass-compatible video data for testing purposes.

This file also contains epoch/task data, since it is necessary for the video data to be compatible with spyglass.
"""

from typing import Optional, Union
from pathlib import Path
import numpy as np

from pynwb import NWBFile
from pynwb.image import ImageSeries
from pynwb.behavior import BehavioralEvents
from neuroconv.datainterfaces import (
    VideoInterface,
)  # requires spikeinterface>0.99.0 which is incompatible with spyglass-neuro

from ndx_franklab_novela import CameraDevice
from neuroconv.datainterfaces.behavior.video.video_utils import get_video_timestamps


def add_behavioral_video(
    nwbfile: NWBFile,
    metadata,
    video_file_path: Union[Path, str],
    alligned_timestamps: Optional[list[np.ndarray]] = None,
) -> None:

    camera_device = CameraDevice(**metadata["Devices"]["CameraDevice"])
    nwbfile.add_device(camera_device)

    if alligned_timestamps is not None:
        timestamps = alligned_timestamps
    else:
        timestamps = get_video_timestamps(file_path=video_file_path)

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
