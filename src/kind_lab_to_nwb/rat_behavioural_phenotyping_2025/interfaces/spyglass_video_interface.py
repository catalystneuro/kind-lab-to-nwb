from copy import deepcopy
from typing import Literal

import numpy as np
from pydantic import FilePath
from pynwb import NWBFile
from pynwb.core import DynamicTable
from pynwb.image import ImageSeries

from neuroconv.datainterfaces import ExternalVideoInterface
from neuroconv.tools.nwb_helpers import get_module

from ndx_franklab_novela import CameraDevice


class SpyglassVideoInterface(ExternalVideoInterface):
    """Data interface for converting behavioral video data to NWB format.

    This class manages video files across multiple epochs and segments,
    handles camera device metadata, and coordinates
    temporal synchronization with other experimental data streams.

    The interface supports multi-camera setups and provides flexible video naming
    and task association through abstract methods. It integrates with the broader
    NWB conversion pipeline by managing video data organization and camera device
    registration using the ndx-franklab-novela extension.

    """

    display_name = "Video"
    keywords = ("video", "behavior")
    associated_suffixes = (".mp4", ".avi", ".wmv", ".mov", ".flx", ".mkv")
    # Other suffixes, while they can be opened by OpenCV, are not supported by DANDI so should probably not list here
    info = "Interface for handling standard video file formats and writing them as ImageSeries with external_files."

    def __init__(
        self,
        file_paths: list[FilePath],
        verbose: bool = False,
        *,
        video_name: str | None = None,
    ):
        super().__init__(file_paths=file_paths, video_name=video_name, verbose=verbose)
        self._default_device_name = "camera_device 0"

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        task_metadata: dict,
        metadata: dict | None = None,
        starting_frames: list[int] | None = None,
        parent_container: Literal["acquisition", "processing/behavior"] = "acquisition",
        module_description: str | None = None,
    ):

        if parent_container not in {"acquisition", "processing/behavior"}:
            raise ValueError(
                f"parent_container must be either 'acquisition' or 'processing/behavior', not {parent_container}."
            )
        metadata = metadata or dict()

        file_paths = self.source_data["file_paths"]

        # Be sure to copy metadata at this step to avoid mutating in-place
        videos_metadata = deepcopy(metadata).get("Behavior", dict()).get("ExternalVideos", None)
        # If no metadata is provided use the default metadata
        if videos_metadata is None or self.video_name not in videos_metadata:
            videos_metadata = deepcopy(self.get_metadata()["Behavior"]["ExternalVideos"])
        image_series_kwargs = videos_metadata[self.video_name]
        image_series_kwargs["name"] = self.video_name
        camera_device_kwargs = image_series_kwargs.pop("device", None)

        if camera_device_kwargs is not None:
            if camera_device_kwargs["name"] in nwbfile.devices:
                camera_device = nwbfile.devices[camera_device_kwargs["name"]]
            else:
                camera_device = CameraDevice(**camera_device_kwargs)
                nwbfile.add_device(camera_device)
            image_series_kwargs["device"] = camera_device

        if self._number_of_files > 1 and starting_frames is None:
            raise TypeError("Multiple paths were specified for the ImageSeries, but no starting_frames were specified!")
        elif starting_frames is not None and len(starting_frames) != self._number_of_files:
            raise ValueError(
                f"Multiple paths ({self._number_of_files}) were specified for the ImageSeries, "
                f"but the length of starting_frames ({len(starting_frames)}) did not match the number of paths!"
            )
        elif starting_frames is not None:
            image_series_kwargs.update(starting_frame=starting_frames)

        image_series_kwargs.update(format="external", external_file=file_paths)

        # Add a custom processing module for tasks
        # This is necessary for the video data to be compatible with spyglass.
        tasks_module = nwbfile.create_processing_module(name="tasks", description="tasks module")

        task_table = DynamicTable(
            name="task_table",
            description="The task table is needed for the video data to be compatible with spyglass. ",
        )
        task_table.add_column(name="task_name", description="Name of the task.")
        task_table.add_column(name="task_description", description="Description of the task.")
        task_table.add_column(name="camera_id", description="Camera ID.")
        task_table.add_column(name="task_epochs", description="Task epochs.")
        task_table.add_column(name="environment", description="Environment where the task is carried out.")
        task_table.add_row(
            task_name=task_metadata["name"],
            task_description=task_metadata["session_description"],
            camera_id=task_metadata["camera_id"],
            task_epochs=task_metadata["task_epochs"],
            environment=task_metadata["environment"],
        )
        tasks_module.add(task_table)

        # Alway write timestamps for spyglass compatibility
        timestamps = self.get_timestamps()
        image_series_kwargs.update(timestamps=np.concatenate(timestamps))

        nwbfile.add_epoch_column(name="task_name", description="Name of the task associated with the epoch.")
        nwbfile.add_epoch(
            start_time=timestamps[0], stop_time=timestamps[-1], tags=["01"], task_name=task_metadata["name"]
        )

        # Attach image series
        image_series = ImageSeries(**image_series_kwargs)
        if parent_container == "acquisition":
            nwbfile.add_acquisition(image_series)
        elif parent_container == "processing/behavior":
            get_module(nwbfile=nwbfile, name="behavior", description=module_description).add(image_series)

        return nwbfile
