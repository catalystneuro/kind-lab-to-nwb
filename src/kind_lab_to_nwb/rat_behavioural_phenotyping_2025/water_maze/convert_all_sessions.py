"""Script to convert all Water Maze sessions to NWB format, following the structure of auditory_fear_conditioning/convert_all_sessions.py."""

import traceback
from pathlib import Path
from pprint import pformat
from typing import Union
from tqdm import tqdm
import natsort
import pandas as pd
import numpy as np

from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.water_maze.convert_session import (
    process_session,
)
from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.utils.utils import (
    get_session_ids_from_excel,
    extract_subject_metadata_from_excel,
    get_subject_metadata_from_task,
)


def water_maze_dataset_to_nwb(
    *,
    data_dir_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    subjects_metadata_file_path: Union[str, Path],
    task_acronym: str = "WM",
    overwrite: bool = False,
    verbose: bool = True,
):
    """Convert the entire dataset to NWB.

    Parameters
    ----------
    data_dir_path : Union[str, Path]
        The path to the directory containing the raw data.
    output_dir_path : Union[str, Path]
        The path to the directory where the NWB files will be saved.
    subjects_metadata_file_path : Union[str, Path], optional
        The path to the Excel file containing subject metadata, by default None
    task_acronym : str, optional
        The acronym of the task, by default "WM"
    overwrite : bool, optional
        Whether to overwrite existing NWB files, by default False
    verbose : bool, optional
        Whether to print verbose output, by default True
    """
    data_dir_path = Path(data_dir_path)
    output_dir_path = Path(output_dir_path)
    if not data_dir_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir_path} does not exist.")
    if not Path(subjects_metadata_file_path).exists():
        raise FileNotFoundError(f"Metadata file {subjects_metadata_file_path} does not exist.")

    output_dir_path.mkdir(
        parents=True,
        exist_ok=True,
    )
    session_to_nwb_kwargs_per_session = get_session_to_nwb_kwargs_per_session(
        data_dir_path=data_dir_path, subjects_metadata_file_path=subjects_metadata_file_path, task_acronym=task_acronym
    )
    if verbose:
        print(f"Found {len(session_to_nwb_kwargs_per_session)} sessions to convert")

    for session_to_nwb_kwargs in tqdm(session_to_nwb_kwargs_per_session, desc="Converting sessions"):
        session_to_nwb_kwargs["output_dir_path"] = output_dir_path

        # Create meaningful error file name using subject and session info
        subject_id = f"{session_to_nwb_kwargs['subject_metadata']['animal ID']}_{session_to_nwb_kwargs['subject_metadata']['cohort ID']}"
        session_id = session_to_nwb_kwargs["session_id"]

        exception_file_path = output_dir_path / f"ERROR_sub_{subject_id}-ses_{session_id}.txt"

        safe_session_to_nwb(
            session_to_nwb_kwargs=session_to_nwb_kwargs,
            exception_file_path=exception_file_path,
        )


def safe_session_to_nwb(
    *,
    session_to_nwb_kwargs: dict,
    exception_file_path: Union[Path, str],
):
    """Convert a session to NWB while handling any errors by recording error messages to the exception_file_path.

    Parameters
    ----------
    session_to_nwb_kwargs : dict
        The arguments for session_to_nwb.
    exception_file_path : Path
        The path to the file where the exception messages will be saved.
    """
    exception_file_path = Path(exception_file_path)
    try:
        process_session(**session_to_nwb_kwargs)
    except Exception as e:
        with open(
            exception_file_path,
            mode="w",
        ) as f:
            f.write(f"session_to_nwb_kwargs: \n {pformat(session_to_nwb_kwargs)}\n\n")
            f.write(traceback.format_exc())


def get_session_to_nwb_kwargs_per_session(
    *,
    data_dir_path: Union[str, Path],
    subjects_metadata_file_path: Union[str, Path],
    task_acronym: str = "WM",
):
    """Get the kwargs for session_to_nwb for each session in the dataset.

    Parameters
    ----------
    data_dir_path : Union[str, Path]
        The path to the directory containing the raw data.
    subjects_metadata_file_path : Union[str, Path], optional
        The path to the Excel file containing subject metadata, by default None
    task_acronym : str, optional
        The acronym of the task, by default "WM"

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries containing the kwargs for session_to_nwb for each session.
    """
    data_dir_path = Path(data_dir_path)
    subjects_metadata_file_path = Path(subjects_metadata_file_path)
    exception_file_path = data_dir_path / f"exceptions_for_task_{task_acronym}.txt"

    session_ids = get_session_ids_from_excel(subjects_metadata_file_path, task_acronym)

    subjects_metadata = extract_subject_metadata_from_excel(subjects_metadata_file_path)
    subjects_metadata = get_subject_metadata_from_task(subjects_metadata, task_acronym)

    session_to_nwb_kwargs_per_session = []

    for subject_metadata in subjects_metadata:

        subject_id = subject_metadata["animal ID"]
        cohort_id = subject_metadata["cohort ID"]
        line = subject_metadata["line"]
        with open(exception_file_path, mode="a") as f:
            f.write(f"Subject {cohort_id}_{subject_id}\n")

        cohort_folder_path = Path(data_dir_path) / line / f"{cohort_id}_{task_acronym}"
        if not cohort_folder_path.exists():
            # raise FileNotFoundError(f"Folder {cohort_folder_path} does not exist")
            with open(exception_file_path, mode="a") as f:
                f.write(f"Folder {cohort_folder_path} does not exist\n\n")
            continue
        for session_id in session_ids:
            video_folder_path = cohort_folder_path / session_id
            if not video_folder_path.exists():
                # raise FileNotFoundError(f"Folder {video_folder_path} does not exist")
                with open(exception_file_path, mode="a") as f:
                    f.write(f"Session {session_id}\n")
                    f.write(f"Folder {video_folder_path} does not exist\n\n")
                continue
            all_video_file_paths = natsort.natsorted(video_folder_path.glob(f"*{subject_id}*.avi"))

            # Try to find the analysis CSV file for this session
            csv_file_paths = list(cohort_folder_path.glob(f"*.csv"))
            analysis_csv_file_path = [p for p in csv_file_paths if session_id.split("_")[1] in p.name]
            if not analysis_csv_file_path:
                # raise FileNotFoundError(f"Folder {video_folder_path} does not exist")
                with open(exception_file_path, mode="a") as f:
                    f.write(f"Session {session_id}\n")
                    f.write(
                        f"No analysis CSV file found for subject '{subject_id}' session '{session_id}' in '{cohort_folder_path}'."
                    )
                continue
            elif len(analysis_csv_file_path) > 1:
                with open(exception_file_path, mode="a") as f:
                    f.write(f"Session {session_id}\n")
                    f.write(
                        f"Multiple analysis CSV files found for subject '{subject_id}' session '{session_id}' in '{cohort_folder_path}'."
                    )
                continue
            else:
                analysis_csv_file_path = analysis_csv_file_path[0]

            session_to_nwb_kwargs_per_session.append(
                {
                    "session_id": f"WM_{session_id}",
                    "subject_metadata": subject_metadata,
                    "all_video_file_paths": all_video_file_paths,
                    "analysis_csv_file_path": analysis_csv_file_path,
                }
            )
    return session_to_nwb_kwargs_per_session


if __name__ == "__main__":
    # Parameters for conversion
    data_dir_path = Path("D:/Kind-CN-data-share/behavioural_pipeline/Water Maze")
    output_dir_path = Path("D:/kind_lab_conversion_nwb/behavioural_pipeline/water_maze")
    subjects_metadata_file_path = Path("D:/Kind-CN-data-share/behavioural_pipeline/general_metadata.xlsx")
    water_maze_dataset_to_nwb(
        data_dir_path=data_dir_path,
        output_dir_path=output_dir_path,
        subjects_metadata_file_path=subjects_metadata_file_path,
        verbose=True,
        overwrite=False,
    )
