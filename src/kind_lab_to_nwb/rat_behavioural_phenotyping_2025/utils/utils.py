import re
import struct
import subprocess
from datetime import datetime
from pathlib import Path
from shutil import rmtree
from typing import List

import pandas as pd
from pydantic import DirectoryPath, FilePath


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

    # For the fields "animal ID" and "cohort ID" convert to int
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

    # Drop the Notes column if it exists
    df = df.drop(columns=["Notes"], errors="ignore")

    # Find the row corresponding to the given task_acronym
    task_row = df[df.iloc[:, 0] == task_acronym]

    if task_row.empty:
        raise ValueError(f"Task acronym '{task_acronym}' not found in Excel file")

    # Get all columns except the first three (acronym, name, pattern)
    session_ids = task_row.iloc[0, 3:].dropna().tolist()

    # Remove trailing whitespace from each session ID
    session_ids = [session_id.rstrip() for session_id in session_ids]
    # Remove parenthesis from session IDs
    session_ids = [session_id.replace("(", "").replace(")", "") for session_id in session_ids]
    return session_ids


SUPPORTED_SUFFIXES = [".avi", ".mp4", ".wmv", ".mov", ".flx", ".mkv"]  # video file's suffixes supported by DANDI


def convert_ts_to_mp4(video_file_paths: List[FilePath]) -> List[FilePath]:
    """Converts video files with .ts extension to .mp4 format using ffmpeg.
    Converted files are saved in a 'converted' subdirectory within the parent directory
    of the original files. If a file is already in .mp4 or another supported format,
    it is skipped. If the output file already exists, conversion is also skipped.

    Parameters
    ----------
    video_file_paths : List[FilePath]
        The list of paths to the video files (.ts) to be converted.

    Returns
    -------
    List[FilePath]
        List of paths to the output video files (either converted .mp4 files
        or the original files if they were already in a supported format).

    Raises
    ------
    FileNotFoundError
        If any of the input video files does not exist.
    ValueError
        If any input file has an unsupported format (not .ts and not in SUPPORTED_SUFFIXES).
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


def convert_ffii_files_to_avi(ffii_file_paths: List[str], frame_rate: int = 15) -> List[str]:
    """
    Convert a list of .ffii files to .avi files using ffmpeg.
    Converted files are saved in a 'converted' subdirectory within the parent directory
    of the original files. If a file is already in .avi or another supported format,
    it is skipped. If the output file already exists, conversion is also skipped.

    ## Requirements
        - [ffmpeg](http://ffmpeg.org/download.html)

    Parameters
    ----------
    ffii_file_paths : List[str]
        List of paths to .ffii files to convert.
    frame_rate : int, optional
        Frame rate for the output video, by default 15.

    Returns
    -------
    List[str]
        List of output .avi file paths.
    """

    output_file_paths = []
    for video_file_path in ffii_file_paths:
        video_file_path = Path(video_file_path)
        # Check if the input file exists
        if not video_file_path.is_file():
            raise FileNotFoundError(f"The file {video_file_path} does not exist.")
        # Check if the input file is a .ffii file
        if video_file_path.suffix.lower() != ".ffii":
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
        # Create a subdirectory called "converted"
        output_dir = Path(video_file_path).parent / "converted"
        output_dir.mkdir(parents=True, exist_ok=True)
        # Define the output file path by replacing the extension with .avi
        output_file_path = output_dir / (video_file_path.stem + ".avi")
        if output_file_path.exists():
            print(f"The file {output_file_path} already exists. Skipping conversion.")
            output_file_paths.append(output_file_path)
            continue
        print(f"Converting {video_file_path}...")
        with open(video_file_path, "rb") as f:
            m = f.read(8)
            if len(m) < 8:
                print(f"File {video_file_path} is too short to contain header, skipping.")
                continue
            height, width = struct.unpack(">2I", m)
            rate = str(frame_rate)
            cmdstr = (
                "ffmpeg",
                "-y",
                "-r",
                rate,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-s",
                f"{width}x{height}",
                "-i",
                "-",
                "-c:v",
                "ffv1",  # Lossless codec
                output_file_path,
            )
            p = subprocess.Popen(cmdstr, stdin=subprocess.PIPE, shell=False)
            try:
                while True:
                    img = f.read(width * height)
                    if not img:
                        break
                    p.stdin.write(img)
                    m = f.read(8)
                    if not m:
                        break
                    height, width = struct.unpack(">2I", m)
                p.stdin.close()
                p.wait()
                print(f"Saved in {output_file_path}")
                output_file_paths.append(output_file_path)
            except Exception as e:
                print(f"Error during ffmpeg process for {video_file_path}: {e}")
                p.kill()
    return output_file_paths


def parse_datetime_from_filename(filename: str) -> datetime:
    """
    Parse datetime from a filename with one of the following formats:
    1. "2024-03-20 10-32-22_597L_598R.ts" - Date and time separated by space ("%Y-%m-%d %H-%M-%S_")
    2. "2022-08-01_302_303_compressed.mp4" - Only date ("%Y-%m-%d_")
    3. "2023-03-31_10-53-05_471.mp4" - Date and time separated by underscore ("%Y-%m-%d_%H-%M-%S_")

    Parameters
    ----------
    filename : str
        The filename to parse

    Returns
    -------
    datetime
        The parsed datetime object

    Raises
    ------
    ValueError
        If the filename doesn't match any of the expected formats
    """
    # Extract just the filename if a path is provided
    if isinstance(filename, Path):
        filename = filename.name

    # Pattern 1: Date and time separated by space (2024-03-20 10-32-22_... or 2024-03-20 10-32-22 ...)
    pattern1 = r"(\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2})[_ ]"
    match = re.search(pattern1, filename)
    if match:
        datetime_str = match.group(1)
        return datetime.strptime(datetime_str, "%Y-%m-%d %H-%M-%S")

    # Pattern 3: Date and time separated by underscore (2023-03-31_10-53-05_...)
    pattern3 = r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_"
    match = re.search(pattern3, filename)
    if match:
        datetime_str = match.group(1)
        return datetime.strptime(datetime_str, "%Y-%m-%d_%H-%M-%S")

    # Pattern 2: Only date (2022-08-01_... or 2022-08-01 ...)
    pattern2 = r"(\d{4}-\d{2}-\d{2})[_ ]"
    match = re.search(pattern2, filename)
    if match:
        datetime_str = match.group(1)
        return datetime.strptime(datetime_str, "%Y-%m-%d")

    # If no pattern matches, raise an error
    raise ValueError(f"Filename '{filename}' doesn't match any of the expected datetime formats")


def dandi_ember_upload(
    nwb_folder_path: DirectoryPath,
    dandiset_folder_path: DirectoryPath,
    dandiset_id: str,
    version: str = "draft",
    files_mode: str = "copy",
    media_files_mode: str = "copy",
    cleanup: bool = True,
):
    """
    Upload NWB files to a Dandiset on the DANDI-EMBER archive (https://dandi.emberarchive.org/).

    This function automates the process of uploading NWB files to a DANDI-EMBER archive. It performs the following steps:
    1. Downloads the specified Dandiset from the DANDI-EMBER archive (metadata only, not assets).
    2. Organizes the provided NWB files into the downloaded Dandiset folder structure using DANDI's organize utility.
    3. Uploads the organized NWB files to the DANDI-EMBER instance.
    4. Cleans up any temporary folders created during the process.

    Parameters
    ----------
    nwb_folder_path : DirectoryPath
        Path to the folder containing the NWB files to be uploaded.
    dandiset_folder_path : DirectoryPath
        Path to a folder where the Dandiset will be downloaded and organized. This folder will be created if it does not exist and will be deleted after upload.
    dandiset_id : str
        The identifier for the Dandiset to which the NWB files will be uploaded (e.g., "000199").
    version : str, optional
        The version of the Dandiset to download from the archive (default is "draft").
    files_mode : str, optional
        The file operation mode for organizing files: 'copy' or 'move' (default is 'copy').
    media_files_mode : str, optional
        The file operation mode for media files: 'copy' or 'move' (default is 'copy').
    cleanup : bool, optional
        Whether to clean up the temporary Dandiset folder and NWB folder after upload (default is True).

    Raises
    ------
    AssertionError
        If the Dandiset download or organization fails.
    Exception
        If the upload process encounters an error, it will be logged and the function will proceed to clean up temporary files.

    Notes
    -----
    - This function will delete both the dandiset_folder_path and nwb_folder_path after upload, so ensure these are temporary or backed up if needed.
    - Uses DANDI's Python API for download, organize, and upload operations.
    - Designed for use with the DANDI-EMBER archive (https://dandi.emberarchive.org/).
    """
    from dandi.download import download as dandi_download
    from dandi.organize import CopyMode, FileOperationMode
    from dandi.organize import organize as dandi_organize
    from dandi.upload import upload as dandi_upload

    # Map string to enum
    files_mode_enum = FileOperationMode.COPY if files_mode.lower() == "copy" else FileOperationMode.MOVE
    media_files_mode_enum = CopyMode.COPY if media_files_mode.lower() == "copy" else CopyMode.MOVE

    dandiset_folder_path = Path(dandiset_folder_path)
    dandiset_folder_path.mkdir(parents=True, exist_ok=True)

    dandiset_path = dandiset_folder_path / dandiset_id
    dandiset_url = f"https://dandi.emberarchive.org/dandiset/{dandiset_id}/{version}"
    dandi_download(urls=dandiset_url, output_dir=str(dandiset_folder_path), get_metadata=True, get_assets=False)
    assert dandiset_path.exists(), "DANDI download failed!"

    dandi_organize(
        paths=str(nwb_folder_path),
        dandiset_path=str(dandiset_path),
        devel_debug=True,
        update_external_file_paths=True,
        files_mode=files_mode_enum,
        media_files_mode=media_files_mode_enum,
    )
    assert len(list(dandiset_path.iterdir())) > 1, "DANDI organize failed!"

    try:
        organized_nwbfiles = [str(x) for x in dandiset_path.rglob("*.nwb")]
        dandi_upload(
            paths=organized_nwbfiles,
            dandi_instance="ember",
        )
    except Exception as e:
        print(f"Error during DANDI upload: {e}")

    finally:
        # Clean up the temporary DANDI folder
        if cleanup:
            rmtree(path=dandiset_folder_path)
            rmtree(path=nwb_folder_path)


def get_cage_ids_from_excel_files(pooled_data_folder_path: DirectoryPath) -> dict:
    """
    Extract a dictionary mapping (animal_id, cohort_id) to cage ID from pooled data Excel files.

    Parameters
    ----------
    pooled_data_folder_path : DirectoryPath
        The path to the folder containing the pooled data Excel files.

    Returns
    -------
    dict
        A dictionary with keys as (animal_id, cohort_id) tuples and values as cage IDs.
    """
    pooled_data_file_paths = list(Path(pooled_data_folder_path).glob(f"*POOLED*.xlsx"))
    if not pooled_data_file_paths:
        raise FileNotFoundError(
            f"No Excel files containing 'POOLED' in their name found in '{pooled_data_folder_path}'."
        )
    cage_id_identifier_columns = ["animal ID", "cohort ID", "cage ID"]
    cage_ids = dict()
    for pooled_data_file_path in pooled_data_file_paths:
        df = pd.read_excel(pooled_data_file_path, sheet_name=0)
        df = df.dropna(how="all")
        df.columns = df.columns.str.strip()
        df.rename(columns={"Unique animal ID": "animal ID"}, inplace=True)
        for col in cage_id_identifier_columns:
            if col not in df.columns:
                raise KeyError(f"Expected column '{col}' is not present in '{pooled_data_file_path}'.")
        for _, row in df.iterrows():
            try:
                animal_id = int(row["animal ID"])
                cohort_id = str(row["cohort ID"])
                cage_id = row["cage ID"]
                cage_ids[(animal_id, cohort_id)] = cage_id
            except Exception as e:
                print(
                    f"Error processing {pooled_data_file_path.stem} row:\n {row[cage_id_identifier_columns].values}: {e}"
                )
                continue  # skip rows with missing or invalid data
    return cage_ids


def update_subjects_metadata_with_cage_ids(subjects_metadata: List[dict], cage_ids: dict) -> List[dict]:
    """
    Update the subjects metadata dictionary with the corresponding cage_id from a lookup dict.
    The lookup dict should use (animal_id, cohort_id) as the key.

    Parameters
    ----------
    subjects_metadata : list[dict]
        List of subject metadata dictionaries.
    cage_ids : dict
        Dictionary mapping (animal_id, cohort_id) tuples to cage IDs.

    Returns
    -------
    List[dict]
        The updated list of subject metadata dictionaries with "cage ID" added.
    """
    for subject_metadata in subjects_metadata:
        subject_metadata["cage ID"] = cage_ids.get(
            (subject_metadata.get("animal ID"), subject_metadata.get("cohort ID")), None
        )
    return subjects_metadata
