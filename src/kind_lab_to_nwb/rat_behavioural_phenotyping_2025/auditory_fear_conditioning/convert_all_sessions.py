"""Primary script to run to convert all sessions in a dataset using session_to_nwb."""
import csv
import logging
import traceback
import warnings
from pathlib import Path
from typing import Union

import pandas as pd
from tqdm import tqdm

from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.auditory_fear_conditioning.convert_session import (
    session_to_nwb,
)
from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.utils import (
    extract_subject_metadata_from_excel,
    get_session_ids_from_excel,
    get_subject_metadata_from_task,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def dataset_to_nwb(
    *,
    data_dir_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    subjects_metadata_file_path: Union[str, Path],
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
    subjects_metadata_file_path : Union[str, Path]
        The path to the file containing subject metadata.
    overwrite : bool, optional
        Whether to overwrite existing NWB files, by default False.
    verbose : bool, optional
        Whether to print verbose output, by default True.

    Raises
    ------
    FileNotFoundError
        If the data directory or metadata file does not exist.
    """
    data_dir_path = Path(data_dir_path)
    output_dir_path = Path(output_dir_path)

    if not data_dir_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir_path} does not exist.")

    if not Path(subjects_metadata_file_path).exists():
        raise FileNotFoundError(f"Metadata file {subjects_metadata_file_path} does not exist.")

    session_to_nwb_kwargs_per_session = get_session_to_nwb_kwargs_per_session(
        data_dir_path=data_dir_path,
        subjects_metadata_file_path=subjects_metadata_file_path,
        task_acronym="AFC",
    )[
        :5
    ]  # temporary limit to 5 sessions for testing

    results = []

    for session_kwargs in tqdm(
        session_to_nwb_kwargs_per_session,
        desc="Converting sessions to NWB",
        disable=not verbose,
        unit="session",
    ):
        session_id = session_kwargs["session_id"]
        subject_metadata = session_kwargs["subject_metadata"]
        video_file_paths = session_kwargs["video_file_paths"]
        freeze_log_file_path = session_kwargs["freeze_log_file_path"]
        freeze_scores_file_path = session_kwargs["freeze_scores_file_path"]

        nwbfile_path = output_dir_path / f"sub-{subject_metadata['animal ID']}_ses-{session_id}.nwb"

        try:
            if nwbfile_path.exists() and not overwrite:
                logging.info(f"NWB file {nwbfile_path} already exists. Skipping conversion.")
                results.append(
                    {
                        "session_id": session_id,
                        "subject_id": subject_metadata["animal ID"],
                        "status": "skipped",
                        "nwbfile_path": str(nwbfile_path),
                        "error": "",
                        **session_kwargs,
                    }
                )
                continue

            logging.info(f"Converting session {session_id} for subject {subject_metadata['animal ID']} to NWB...")

            session_to_nwb(
                nwbfile_path=nwbfile_path,
                video_file_paths=video_file_paths,
                freeze_log_file_path=freeze_log_file_path,
                freeze_scores_file_path=freeze_scores_file_path,
                session_id=session_id,
                subject_metadata=subject_metadata,
                overwrite=overwrite,
            )
            results.append(
                {
                    "session_id": session_id,
                    "subject_id": subject_metadata["animal ID"],
                    "status": "success",
                    "nwbfile_path": str(nwbfile_path),
                    "error": "",
                    **session_kwargs,
                }
            )
        except Exception as e:
            error_message = traceback.format_exc()
            logging.warning(f"Error converting session {session_id}: {e}")
            results.append(
                {
                    "session_id": session_id,
                    "subject_id": subject_metadata["animal ID"],
                    "status": "failed",
                    "nwbfile_path": str(nwbfile_path),
                    "error": error_message,
                }
            )

    results_csv_path = output_dir_path / "AFC_conversion_results.csv"
    results_data = pd.DataFrame(results)
    results_data.to_csv(results_csv_path, index=False)
    logging.info(f"Conversion completed. Results saved to {results_csv_path}")


def get_session_to_nwb_kwargs_per_session(
    *,
    data_dir_path: Union[str, Path],
    subjects_metadata_file_path: Union[str, Path] = None,
    task_acronym: str = "AFC",
):
    """Get the kwargs for session_to_nwb for each session in the dataset.

    Parameters
    ----------
    data_dir_path : Union[str, Path]
        The path to the directory containing the raw data.
    subjects_metadata_file_path : Union[str, Path], optional
        The path to the file containing subject metadata, by default None.
    task_acronym : str, optional
        The acronym of the task for which to get session IDs, by default "AFC".

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries containing the kwargs for session_to_nwb for each session.

    Raises
    ------
    FileNotFoundError
        If the data directory or metadata file does not exist.
    """

    session_ids = get_session_ids_from_excel(
        subjects_metadata_file_path=subjects_metadata_file_path,
        task_acronym=task_acronym,
    )

    subjects_metadata = extract_subject_metadata_from_excel(subjects_metadata_file_path)
    subjects_metadata = get_subject_metadata_from_task(subjects_metadata, task_acronym)

    session_to_nwb_kwargs = []
    for subject_metadata in subjects_metadata:
        for session_id in session_ids:

            try:
                session_to_nwb_kwargs.append(
                    get_session_to_nwb_kwargs(
                        data_dir_path=data_dir_path,
                        subject_metadata=subject_metadata,
                        session_id=session_id,
                        task_acronym=task_acronym,
                    )
                )
            except Exception as e:
                logging.warning(
                    f"Error getting session_to_nwb kwargs for subject {subject_metadata['animal ID']} "
                    f"session {session_id}: {e}"
                )
                continue
    return session_to_nwb_kwargs


def get_session_to_nwb_kwargs(data_dir_path, subject_metadata, session_id, task_acronym):
    cohort_folder_path = data_dir_path / subject_metadata["line"] / f"{subject_metadata['cohort ID']}_{task_acronym}"
    if not cohort_folder_path.exists():
        raise FileNotFoundError(f"Folder {cohort_folder_path} does not exist")

    video_folder_path = cohort_folder_path / session_id
    if not video_folder_path.exists():
        raise FileNotFoundError(f"Folder {video_folder_path} does not exist")

    ffii_file_paths = list(video_folder_path.glob(f"*{subject_metadata['animal ID']}*.ffii"))
    if len(ffii_file_paths) == 0:
        raise FileNotFoundError(
            f"No video files found in for animal ID {subject_metadata['animal ID']} in '{video_folder_path}'."
        )
    elif len(ffii_file_paths) > 1:
        raise FileExistsError(
            f"Multiple video files found for animal ID {subject_metadata['animal ID']} in {video_folder_path}."
        )

    freeze_scores_file_paths = list(video_folder_path.glob(f"*{subject_metadata['line']}*.csv"))
    if len(freeze_scores_file_paths):
        freeze_scores_file_path = freeze_scores_file_paths[0]
    else:
        freeze_scores_file_path = None
        warnings.warn(f"No freeze scores file (.csv) found in {video_folder_path}.")

    freeze_log_file_path = video_folder_path / "Freeze_Log.xls"
    if not freeze_log_file_path.exists():
        raise FileNotFoundError(f"Freeze log file {freeze_log_file_path} does not exist")

    return dict(
        session_id=f"{task_acronym}_{session_id}",
        subject_metadata=subject_metadata,
        video_file_paths=ffii_file_paths,
        freeze_log_file_path=freeze_log_file_path,
        freeze_scores_file_path=freeze_scores_file_path,
    )


if __name__ == "__main__":

    # Parameters for conversion
    data_dir_path = Path("/Volumes/T9/Behavioural Pipeline/Auditory Fear Conditioning")
    output_dir_path = Path("/Users/weian/data/Kind/nwbfiles")

    subjects_metadata_file_path = Path("/Users/weian/data/RAT ID metadata Yunkai copy - updated 12.2.25.xlsx")

    dataset_to_nwb(
        data_dir_path=data_dir_path,
        output_dir_path=output_dir_path,
        subjects_metadata_file_path=subjects_metadata_file_path,
        verbose=False,
        overwrite=True,
    )
