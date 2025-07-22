"""Script to convert all Water Maze sessions to NWB format, following the structure of auditory_fear_conditioning/convert_all_sessions.py."""
import csv
import logging
import traceback
from pathlib import Path
from typing import Union

import natsort
import pandas as pd
from tqdm import tqdm

from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.utils import (
    extract_subject_metadata_from_excel,
    get_session_ids_from_excel,
    get_subject_metadata_from_task,
)
from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.utils.utils import (
    dandi_ember_upload,
)
from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.water_maze.convert_session import (
    process_session,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_session_to_nwb_kwargs_per_session(
    *,
    data_dir_path: Union[str, Path],
    subjects_metadata_file_path: Union[str, Path],
    task_acronym: str = "WM",
):
    session_ids = get_session_ids_from_excel(
        subjects_metadata_file_path=subjects_metadata_file_path,
        task_acronym=task_acronym,
    )
    subjects_metadata = extract_subject_metadata_from_excel(subjects_metadata_file_path)
    subjects_metadata = get_subject_metadata_from_task(subjects_metadata, task_acronym)
    session_to_nwb_kwargs = []
    for subject_metadata in subjects_metadata:
        subject_id = subject_metadata["animal ID"]
        cohort_id = subject_metadata["cohort ID"]
        line = subject_metadata["line"]
        cohort_folder_path = Path(data_dir_path) / line / f"{cohort_id}_{task_acronym}"
        if not cohort_folder_path.exists():
            logging.warning(f"Cohort folder {cohort_folder_path} does not exist. Skipping subject {subject_id}.")
            continue
        for session_id in session_ids:
            video_folder_path = cohort_folder_path / session_id
            video_file_paths = natsort.natsorted(video_folder_path.glob(f"*{subject_id}*.avi"))
            if not video_file_paths:
                logging.warning(
                    f"No video files found for animal ID {subject_id} in '{video_folder_path}'. Skipping session."
                )
                continue
            # Try to find the analysis CSV file for this session
            csv_file_paths = list(cohort_folder_path.glob(f"*.csv"))
            analysis_csv_file_path = [p for p in csv_file_paths if session_id.split("_")[1] in p.name]
            if not analysis_csv_file_path:
                logging.warning(
                    f"No analysis CSV file found for subject '{subject_id}' session '{session_id}' in '{cohort_folder_path}'."
                )
                continue
            elif len(analysis_csv_file_path) > 1:
                logging.warning(
                    f"Multiple analysis CSV files found for session '{session_id}' in '{cohort_folder_path}'\n '{analysis_csv_file_path}'. Using the first one."
                )
                analysis_csv_file_path = analysis_csv_file_path[0]
            else:
                analysis_csv_file_path = analysis_csv_file_path[0]

            session_to_nwb_kwargs.append(
                {
                    "session_id": f"WM_{session_id}",
                    "subject_metadata": subject_metadata,
                    "video_file_paths": video_file_paths,
                    "analysis_csv_file_path": analysis_csv_file_path,
                }
            )
    return session_to_nwb_kwargs


def dataset_to_nwb(
    *,
    data_dir_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    subjects_metadata_file_path: Union[str, Path],
    overwrite: bool = False,
    verbose: bool = True,
):
    data_dir_path = Path(data_dir_path)
    output_dir_path = Path(output_dir_path)
    if not data_dir_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir_path} does not exist.")
    if not Path(subjects_metadata_file_path).exists():
        raise FileNotFoundError(f"Metadata file {subjects_metadata_file_path} does not exist.")

    session_to_nwb_kwargs_per_session = get_session_to_nwb_kwargs_per_session(
        data_dir_path=data_dir_path,
        subjects_metadata_file_path=subjects_metadata_file_path,
        task_acronym="WM",
    )[
        :5
    ]  # Limit to first 5 sessions for testing; remove this slice for full conversion

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
        analysis_csv_file_path = session_kwargs["analysis_csv_file_path"]
        subject_id = subject_metadata["animal ID"]
        cohort_id = subject_metadata["cohort ID"]
        nwbfile_path = output_dir_path / f"sub-{subject_id}_{cohort_id}_ses-{session_id}.nwb"
        try:
            if nwbfile_path.exists() and not overwrite:
                logging.info(f"NWB file {nwbfile_path} already exists. Skipping conversion.")
                results.append(
                    {
                        "session_id": session_id,
                        "subject_id": subject_id,
                        "status": "skipped",
                        "nwbfile_path": str(nwbfile_path),
                        "error": "",
                        **session_kwargs,
                    }
                )
                continue
            logging.info(f"Converting session {session_id} for subject {subject_id} to NWB...")
            process_session(
                analysis_csv_file_path=analysis_csv_file_path,
                all_video_file_paths=video_file_paths,
                output_dir_path=output_dir_path,
                session_id=session_id,
                subject_metadata=subject_metadata,
            )
            results.append(
                {
                    "session_id": session_id,
                    "subject_id": subject_id,
                    "status": "success",
                    "nwbfile_path": str(nwbfile_path),
                    "error": "",
                    **session_kwargs,
                }
            )
        except Exception as e:
            error_message = traceback.format_exc()
            logging.warning(f"Error converting session {session_id} for subject {subject_id}: {e}")
            results.append(
                {
                    "session_id": session_id,
                    "subject_id": subject_id,
                    "status": "failed",
                    "nwbfile_path": str(nwbfile_path),
                    "error": error_message,
                    **session_kwargs,
                }
            )
    results_csv_path = output_dir_path / "WM_conversion_results.csv"
    results_data = pd.DataFrame(results)
    results_data.to_csv(results_csv_path, index=False)
    logging.info(f"Conversion completed. Results saved to {results_csv_path}")


if __name__ == "__main__":
    # Parameters for conversion
    data_dir_path = Path("/Volumes/T9/Behavioural Pipeline/Water Maze")
    output_dir_path = Path("/Users/weian/data/Water Maze/nwbfiles")
    subjects_metadata_file_path = Path("/Users/weian/data/RAT ID metadata Yunkai copy - updated 12.2.25.xlsx")
    dataset_to_nwb(
        data_dir_path=data_dir_path,
        output_dir_path=output_dir_path,
        subjects_metadata_file_path=subjects_metadata_file_path,
        verbose=True,
        overwrite=True,
    )

    dandiset_folder_path = Path("/Users/weian/data/Kind/tmp")
    dandi_ember_upload(
        nwb_folder_path=output_dir_path,
        dandiset_folder_path=dandiset_folder_path,
        dandiset_id="000205",
    )
