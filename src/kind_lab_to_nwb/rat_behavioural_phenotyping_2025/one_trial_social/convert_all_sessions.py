"""Primary script to run to convert all sessions in a dataset using session_to_nwb."""

import traceback
from pathlib import Path
from pprint import pformat
from typing import Union
from tqdm import tqdm
import numpy as np


from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.one_trial_social.convert_session import (
    session_to_nwb,
)
from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.utils.utils import (
    get_session_ids_from_excel,
    extract_subject_metadata_from_excel,
    get_subject_metadata_from_task,
    parse_datetime_from_filename,
)


def one_trial_social_dataset_to_nwb(
    *,
    data_dir_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    subjects_metadata_file_path: Union[str, Path],
    task_acronym: str = "1TS",
    verbose: bool = False,
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
        The acronym of the task, by default "1TS"
    verbose : bool, optional
        Whether to print verbose output, by default True
    """
    data_dir_path = Path(data_dir_path)
    output_dir_path = Path(output_dir_path)
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
        session_to_nwb(**session_to_nwb_kwargs)
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
    task_acronym: str = "1TS",
):
    """Get the kwargs for session_to_nwb for each session in the dataset.

    Parameters
    ----------
    data_dir_path : Union[str, Path]
        The path to the directory containing the raw data.
    subjects_metadata_file_path : Union[str, Path], optional
        The path to the Excel file containing subject metadata, by default None
    task_acronym : str, optional
        The acronym of the task, by default "1TS"

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
        with open(exception_file_path, mode="a") as f:
            f.write(f"Subject {subject_metadata['cohort ID']}_{subject_metadata['animal ID']}\n")
        for session_id in session_ids:

            cohort_folder_path = (
                data_dir_path / subject_metadata["line"] / f"{subject_metadata['cohort ID']}_{task_acronym}"
            )
            if not cohort_folder_path.exists():
                # raise FileNotFoundError(f"Folder {cohort_folder_path} does not exist")
                with open(exception_file_path, mode="a") as f:
                    f.write(f"Session {session_id}\n")
                    f.write(f"Folder {cohort_folder_path} does not exist\n\n")
                continue

            # check if boris file exists on the cohort folder
            boris_file_paths = list(cohort_folder_path.glob("*.boris"))
            if len(boris_file_paths) == 0:
                boris_file_path = None
                # warnings.warn(f"No BORIS file found in {cohort_folder_path}")
                # with open(exception_file_path, mode="a") as f:
                #     f.write(f"No BORIS file found in {cohort_folder_path} for session {session_id}\n\n")
            else:
                boris_file_path = boris_file_paths[0]

            video_folder_path = cohort_folder_path / session_id
            if not video_folder_path.exists():
                # raise FileNotFoundError(f"Folder {video_folder_path} does not exist")
                with open(exception_file_path, mode="a") as f:
                    f.write(f"Session {session_id}\n")
                    f.write(f"Folder {video_folder_path} does not exist\n\n")
                continue

            video_file_paths = list(video_folder_path.glob(f"*{subject_metadata['animal ID']}*"))
            if len(video_file_paths) == 0:
                cage_id = subject_metadata["cage ID"]
                if not np.isnan(cage_id):
                    cage_id = int(cage_id)
                    video_file_paths = list(video_folder_path.glob(f"*cage{cage_id}*"))
                if len(video_file_paths) == 0:
                    with open(exception_file_path, mode="a") as f:
                        f.write(f"Session {session_id}\n")
                        f.write(f"No video files found in {video_folder_path} for session {session_id}\n\n")
                    continue

            video_path = Path(video_file_paths[0])
            session_start_time = parse_datetime_from_filename(video_path.name)
            # TODO add timezone information
            session_kwargs = {
                "video_file_paths": video_file_paths,
                "boris_file_path": boris_file_path,
                "subject_metadata": subject_metadata,
                "session_id": f"{task_acronym}_{session_id}",
                "session_start_time": session_start_time,
            }
            session_to_nwb_kwargs_per_session.append(session_kwargs)
        with open(exception_file_path, mode="a") as f:
            f.write(f"------------------------------------------------------------\n\n")
    return session_to_nwb_kwargs_per_session


if __name__ == "__main__":

    data_dir_path = Path("D:/Kind-CN-data-share/behavioural_pipeline/1 Trial Social")
    output_dir_path = Path("D:/kind_lab_conversion_nwb/behavioural_pipeline/one_trial_social")
    subjects_metadata_file_path = Path("D:/Kind-CN-data-share/behavioural_pipeline/general_metadata.xlsx")
    task_acronym = "1TS"

    verbose = False

    one_trial_social_dataset_to_nwb(
        data_dir_path=data_dir_path,
        output_dir_path=output_dir_path,
        subjects_metadata_file_path=subjects_metadata_file_path,
        task_acronym=task_acronym,
        verbose=verbose,
    )
