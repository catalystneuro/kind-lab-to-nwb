from open_ephys.analysis import Session
from pydantic import DirectoryPath, FilePath
from typing import Optional
from neuroconv.datainterfaces.behavior.video.video_utils import get_video_timestamps


def get_first_CS_time(folder_path: DirectoryPath, rate: float = 2000.0) -> float:
    """
    Get the first CS time.

    Args:
        folder_path (DirectoryPath): Path to the TDT files.
        rate (float): Sampling rate of the TDT files in Hz. Default is 2000.0.

    Returns:
        float: The first CS time in seconds.
    """
    session = Session(folder_path)
    recording = session.recordings[0]
    event_table = recording.events

    for channel, events in event_table.groupby("channel"):
        if channel == 2:  # Assuming channel 2 is the one for CS events
            events = events.sort_values("timestamp")
            first_CS_frame = events["timestamp"].to_list()[0]
            return first_CS_frame / rate


def get_first_CS_video_frame(file_path: FilePath, subject_id: str) -> Optional[int]:
    """
    Get the first CS video frame index from the CS video frames file.

    Args:
        file_path (FilePath): Path to the CS video frames file.
        subject_id (str): Subject ID.

    Returns:
        int: The first CS video frame index.
    """
    import pandas as pd

    df = pd.read_excel(file_path)
    subject_row = df[df["rat_ID"] == subject_id]
    if not subject_row.empty:
        return subject_row["first_CS_frame_index"].values[0]
    else:
        None


def compute_time_offset(video_file_path: FilePath, first_CS_video_frame: float, first_CS_time: float) -> float:
    """
    Compute the time offset between the first CS video frame and the first CS time.

    Args:
        video_file_path (FilePath): Path to the video file.
        first_CS_video_frame (int): The index of the first CS video frame.
        first_CS_time (float): The first CS time in seconds.

    Returns:
        float: The time offset in seconds.
    """
    video_timestamps = get_video_timestamps(file_path=video_file_path)
    return first_CS_time - video_timestamps[first_CS_video_frame]
