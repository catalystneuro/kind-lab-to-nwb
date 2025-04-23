import subprocess
from pydantic import FilePath
from typing import List
import pandas as pd
from copy import deepcopy
from datetime import datetime
import importlib.metadata
from pathlib import Path
import uuid

from pynwb.file import Subject

from ndx_events import NdxEventsNWBFile


def make_ndx_event_nwbfile_from_metadata(metadata: dict) -> NdxEventsNWBFile:
    """Make NdxEventsNWBFile from available metadata."""

    assert metadata is not None, "Metadata is required to create an NWBFile but metadata=None was passed."

    nwbfile_kwargs = deepcopy(metadata["NWBFile"])
    # convert ISO 8601 string to datetime
    if isinstance(nwbfile_kwargs.get("session_start_time"), str):
        nwbfile_kwargs["session_start_time"] = datetime.fromisoformat(nwbfile_kwargs["session_start_time"])
    if "session_description" not in nwbfile_kwargs:
        nwbfile_kwargs["session_description"] = "No description."
    if "identifier" not in nwbfile_kwargs:
        nwbfile_kwargs["identifier"] = str(uuid.uuid4())
    if "source_script" not in nwbfile_kwargs:
        neuroconv_version = importlib.metadata.version("neuroconv")
        nwbfile_kwargs["source_script"] = f"Created using NeuroConv v{neuroconv_version}"
        nwbfile_kwargs["source_script_file_name"] = __file__  # Required for validation

    if "Subject" in metadata:
        nwbfile_kwargs["subject"] = metadata["Subject"]
        # convert ISO 8601 string to datetime
        if "date_of_birth" in nwbfile_kwargs["subject"] and isinstance(nwbfile_kwargs["subject"]["date_of_birth"], str):
            nwbfile_kwargs["subject"]["date_of_birth"] = datetime.fromisoformat(
                nwbfile_kwargs["subject"]["date_of_birth"]
            )
        nwbfile_kwargs["subject"] = Subject(**nwbfile_kwargs["subject"])

    return NdxEventsNWBFile(**nwbfile_kwargs)


def extract_subject_metadata_from_excel(subjects_metadata_file_path: FilePath) -> List[dict]:
    """Extract subject metadata from an excel sheet for all subjects.

    Parameters
    ----------
    subjects_metadata_file_path : Path
        Path to the Excel file containing subject metadata

    Returns
    -------
    List[dict]
        List of dictionaries containing the metadata for each subject
    """

    # Read the Excel file
    df = pd.read_excel(subjects_metadata_file_path)
    # Remove rows with all NaN values
    df = df.dropna(how="all")

    # Convert dataframe to list of dictionaries
    subjects_metadata = df.to_dict("records")

    # For the field "tasks conducted", convert comma-separated string to list of strings
    for subject in subjects_metadata:
        if "tasks conducted" in subject and isinstance(subject["tasks conducted"], str):
            subject["tasks conducted"] = [task.strip() for task in subject["tasks conducted"].split(",")]

    # For the fiels "animal ID" and "cohort ID" convert to int
    for subject in subjects_metadata:
        subject["animal ID"] = int(subject["animal ID"])
        subject["cohort no."] = int(subject["cohort no."])

    return subjects_metadata


def get_subject_metadata_from_task(subjects_metadata: List[dict], task_acronym: str) -> List[dict]:
    """Get subject metadata for subjects that have conducted a given task.

    Parameters
    ----------
    subjects_metadata : List[dict]
        List of dictionaries containing the metadata for each subject
    task_acronym : str
        The task acronym

    Returns
    -------
    List[dict]
        List of dictionaries containing the metadata for each subject that has conducted the given task
    """

    return [
        subject
        for subject in subjects_metadata
        if "tasks conducted" in subject and task_acronym in subject["tasks conducted"]
    ]


def get_session_ids_from_excel(subjects_metadata_file_path: FilePath, task_acronym: str) -> List[str]:
    # Read the Excel file, specifically the "Task acronyms & structure" sheet
    df = pd.read_excel(subjects_metadata_file_path, sheet_name="Task acronyms & structure")

    # Find the row corresponding to the given task_acronym
    task_row = df[df.iloc[:, 0] == task_acronym]

    if task_row.empty:
        raise ValueError(f"Task acronym '{task_acronym}' not found in Excel file")

    # Get all columns except the first three (acronym, name, pattern)
    session_ids = task_row.iloc[0, 3:].dropna().tolist()

    # Remove trailing whitespace from each session ID
    session_ids = [session_id.rstrip() for session_id in session_ids]
    return session_ids


SUPPORTED_SUFFIXES = [".avi", ".mp4", ".wmv", ".mov", ".flx", ".mkv"]  # video file's suffixes supported by DANDI


def convert_ts_to_mp4(video_file_paths: List[FilePath]) -> List[FilePath]:
    """Convert .ts video files to .mp4 format using ffmpeg.
    If the file is not in .mp4 format, it will create a new .mp4 file in a subdirectory called "converted".

    Parameters
    ----------
    video_file_path : Path
        Path to the .ts video file

    Returns
    -------
    Path
        Path to the converted .mp4 video file
    """

    output_file_paths = []

    for video_file_path in video_file_paths:
        # Create a subdirectory called "converted"
        output_dir = video_file_path.parent / "converted"
        output_dir.mkdir(parents=True, exist_ok=True)
        # Check if the input file exists
        if not video_file_path.is_file():
            raise FileNotFoundError(f"The file {video_file_path} does not exist.")

        # Check if the input file is a .ts file
        if video_file_path.suffix.lower() != ".ts":
            # Check if the file is already in a supported format
            if video_file_path.suffix.lower() in SUPPORTED_SUFFIXES:
                print(
                    f"Skipping conversion: {video_file_path.name} is already in {video_file_path.suffix} format, which is supported by DANDI."
                )
                output_file_paths.append(video_file_path)
                continue
            else:
                raise ValueError(
                    f"Unsupported file format: {video_file_path.name} has extension {video_file_path.suffix}, but only .ts files can be converted."
                )

        # Define the output file path by replacing the extension with .mp4
        output_file_path = output_dir / (video_file_path.stem + ".mp4")

        # Check if the output file already exists
        if output_file_path.is_file():
            print(f"The file {output_file_path} already exists. Skipping conversion.")
            output_file_paths.append(output_file_path)
        else:  # Use ffmpeg to convert the video file
            try:
                subprocess.run(
                    ["ffmpeg", "-i", video_file_path, "-c:v", "copy", "-c:a", "aac", output_file_path],
                    check=True,
                    capture_output=True,
                )
                output_file_paths.append(output_file_path)

            except subprocess.CalledProcessError as e:
                print(f"An error occurred: {e}")

    return output_file_paths
