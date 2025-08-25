"""Primary script to run to convert all sessions in a dataset using session_to_nwb."""

import traceback
from pathlib import Path
from pprint import pformat
from typing import Union
from tqdm import tqdm
import numpy as np


from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.object_recognition.convert_session import (
    session_to_nwb,
)
from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.utils.utils import (
    get_session_ids_from_excel,
    extract_subject_metadata_from_excel,
    get_subject_metadata_from_task,
    parse_datetime_from_filename,
)


def object_recognition_dataset_to_nwb(
    *,
    data_dir_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    subjects_metadata_file_path: Union[str, Path],
    task_acronym: str = "OR",
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
        The acronym of the task, by default "OR"
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
    task_acronym: str = "OR",
):
    """Get the kwargs for session_to_nwb for each session in the dataset.

    Parameters
    ----------
    data_dir_path : Union[str, Path]
        The path to the directory containing the raw data.
    subjects_metadata_file_path : Union[str, Path], optional
        The path to the Excel file containing subject metadata, by default None
    task_acronym : str, optional
        The acronym of the task, by default "OR"

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

        cohort_folder_path = (
            data_dir_path / subject_metadata["line"] / f"{subject_metadata['cohort ID']}_{task_acronym}"
        )
        if not cohort_folder_path.exists():
            # raise FileNotFoundError(f"Folder {cohort_folder_path} does not exist")
            with open(exception_file_path, mode="a") as f:
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

        # check if boris info file exists
        boris_info_file_paths = list(data_dir_path.glob(f"{subject_metadata['cohort ID']}_OR_borris_info.xlsx"))
        if len(boris_info_file_paths) == 0:
            boris_info_file_path = None
            # warnings.warn(f"No BORIS info excel file found in {analysis_folder_path}")
        else:
            boris_info_file_path = boris_info_file_paths[0]

        for session_id in session_ids:
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

            session_start_times_dict = {}
            if "LTM" in session_id:
                # In the long term memory sessions the sample and the test trial happens on consecutive days
                # we have to group the video files when they are from consecutive days (date_str difference is 1 day)
                session_start_times = []
                for video_file_path in video_file_paths:
                    session_start_times.append(parse_datetime_from_filename(video_file_path.name))
                session_start_times.sort()
                for i in range(len(session_start_times) - 1):
                    if (session_start_times[i + 1].day - session_start_times[i].day) == 1:
                        # consecutive days
                        date_str = session_start_times[i].strftime("%Y-%m-%d")
                        if date_str not in session_start_times_dict:
                            session_start_times_dict[date_str] = []
                        session_start_times_dict[date_str].append((video_file_paths[i], session_start_times[i]))
                        session_start_times_dict[date_str].append((video_file_paths[i + 1], session_start_times[i + 1]))

            else:
                for video_file_path in video_file_paths:
                    session_start_time = parse_datetime_from_filename(video_file_path.name)
                    date_str = session_start_time.strftime("%Y-%m-%d")
                    if date_str not in session_start_times_dict:
                        session_start_times_dict[date_str] = []
                    session_start_times_dict[date_str].append((video_file_path, session_start_time))

            for date_str, video_start_times in session_start_times_dict.items():
                if "Test" in session_id:
                    if len(video_start_times) != 2:
                        with open(exception_file_path, mode="a") as f:
                            f.write(f"Session {session_id}\n")
                            f.write(f"Expected 2 video files on {date_str} found {len(video_start_times)}\n")
                        continue
                    # We have both sample and test videos for this date
                    sample_video_path, sample_start_time = video_start_times[0]
                    test_video_path, test_start_time = video_start_times[1]
                    session_kwargs = {
                        "video_file_paths": [sample_video_path, test_video_path],
                        "boris_file_path": boris_file_path,
                        "boris_info_file_path": boris_info_file_path,
                        "subject_metadata": subject_metadata,
                        "session_id": f"{task_acronym}_{session_id.split(' ')[1]}",
                        "session_start_time": min(sample_start_time, test_start_time),
                    }
                    session_to_nwb_kwargs_per_session.append(session_kwargs)

                elif "Hab" in session_id:
                    if len(video_start_times) != 1:
                        with open(exception_file_path, mode="a") as f:
                            f.write(f"Session {session_id}\n")
                            f.write(f"Expected 1 video file on {date_str} found {len(video_start_times)}\n")
                        continue
                    video_file_path, session_start_time = video_start_times[0]
                    session_kwargs = {
                        "video_file_paths": [video_file_path],
                        "boris_file_path": None,
                        "boris_info_file_path": None,
                        "subject_metadata": subject_metadata,
                        "session_id": f"{task_acronym}_{session_id}",
                        "session_start_time": session_start_time,
                    }
                    session_to_nwb_kwargs_per_session.append(session_kwargs)

        with open(exception_file_path, mode="a") as f:
            f.write(f"------------------------------------------------------------\n\n")

    return session_to_nwb_kwargs_per_session


if __name__ == "__main__":

    data_dir_path = Path("D:/Kind-CN-data-share/behavioural_pipeline/Object Recognition")
    output_dir_path = Path("D:/kind_lab_conversion_nwb/behavioural_pipeline/object_recognition")
    subjects_metadata_file_path = Path("D:/Kind-CN-data-share/behavioural_pipeline/general_metadata.xlsx")
    task_acronym = "OR"

    verbose = False

    object_recognition_dataset_to_nwb(
        data_dir_path=data_dir_path,
        output_dir_path=output_dir_path,
        subjects_metadata_file_path=subjects_metadata_file_path,
        task_acronym=task_acronym,
        verbose=verbose,
    )
