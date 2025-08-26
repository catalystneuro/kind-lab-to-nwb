"""Primary script to run to convert an entire session for of data using the NWBConverter."""

import subprocess
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import FilePath
from pynwb import NWBHDF5IO

from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.auditory_fear_conditioning import (
    AuditoryFearConditioningNWBConverter,
)
from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.utils import (
    extract_subject_metadata_from_excel,
    get_session_ids_from_excel,
    get_subject_metadata_from_task,
)
from neuroconv.utils import dict_deep_update, load_dict_from_file


def _convert_ffii_to_avi(
    folder_path: Union[str, Path], convert_ffii_repo_path: Optional[Union[str, Path]] = None, frame_rate: int = 15
) -> None:
    """
    Convert ffii files in a directory to avi format using the convert-ffii tool.
    This utility function requires ffmpeg (https://ffmpeg.org/download.html) to be installed on the system prior to use.

    Notes
    -----
    The video conversion script is forked from https://github.com/jf-lab/convert-ffii.git
    The converted avi files will be saved in the same directory as the input files.

    This utility function checks if the convert-ffii repository is available,
    clones it if needed, and runs the conversion script on the specified directory.

    Parameters
    ----------
    folder_path : Union[str, Path]
        Path to the directory containing the ffii files to be converted
    convert_ffii_repo_path : Optional[Union[str, Path]], optional
        Path where the convert-ffii repository should be or will be cloned to.
        If None, will use the current working directory.
    frame_rate : int, optional
        Frame rate to use for the video conversion, by default 15 Hz

    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Input directory '{folder_path}' does not exist.")

    convert_ffii_repo_path = Path(convert_ffii_repo_path) if convert_ffii_repo_path else None
    if convert_ffii_repo_path is None:
        convert_ffii_repo_path = Path.cwd() / "convert-ffii"

    if not convert_ffii_repo_path.exists():
        print(f"Cloning convert-ffii repository to {str(convert_ffii_repo_path)}...")
        subprocess.run(
            ["git", "clone", "https://github.com/weiglszonja/convert-ffii.git", str(convert_ffii_repo_path)], check=True
        )
    else:
        print(f"Using existing convert-ffii repository at '{str(convert_ffii_repo_path)}'.")

    # Run the conversion script
    ffii_to_avi_conversion_script = convert_ffii_repo_path / "ffii2avi_recursive.py"
    if not ffii_to_avi_conversion_script.exists():
        raise FileNotFoundError(f"Conversion script not found at '{ffii_to_avi_conversion_script}'.")

    subprocess.run(
        ["python", str(ffii_to_avi_conversion_script), str(folder_path), "--fps", str(frame_rate)], check=True
    )


def session_to_nwb(
    output_dir_path: Union[str, Path],
    video_file_path: Union[FilePath, str],
    freeze_log_file_path: Union[FilePath, str],
    session_id: str,
    subject_metadata: dict,
    freeze_scores_file_path: Optional[Union[FilePath, str]] = None,
    overwrite: bool = False,
):
    """
    Convert a session of auditory fear conditioning task to NWB format.

    Parameters
    ----------
    output_dir_path : Union[str, Path]
        The folder path where the NWB file will be saved.
    video_file_path: Union[FilePath, str]
        The path to the video file (.avi) to be converted.
    freeze_log_file_path: Union[FilePath, str]
        The path to the freeze log file.
    session_id: str
        The session ID to be used in the metadata.
    subject_metadata: dict
        The metadata for the subject, including animal ID and cohort ID.
    freeze_scores_file_path: Union[FilePath, str], optional
        The path to the freeze scores file (.csv).
    overwrite: bool, optional
        Whether to overwrite the NWB file if it already exists, by default False.
    """
    output_dir_path = Path(output_dir_path)
    output_dir_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    subject_id = f"{subject_metadata['animal ID']}_{subject_metadata['cohort ID']}"
    nwbfile_path = output_dir_path / f"sub-{subject_id}_ses-{session_id}.nwb"

    source_data = dict()
    conversion_options = dict()

    editable_metadata_path = Path(__file__).parent / "metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    task_metadata = editable_metadata["SessionTypes"][session_id]

    # Add Behavioral Video
    source_data.update(dict(Video=dict(file_paths=[video_file_path], video_name="BehavioralVideo")))
    conversion_options.update(dict(Video=dict(task_metadata=task_metadata)))

    if freeze_scores_file_path is not None:
        # Add Freeze Scores as trials
        source_data.update(dict(Behavior=dict(file_path=freeze_scores_file_path, subject_id=subject_id)))

    converter = AuditoryFearConditioningNWBConverter(source_data=source_data, verbose=True)

    metadata = converter.get_metadata()
    # Update default metadata with the editable in the corresponding yaml file
    editable_metadata_path = Path(__file__).parent / "metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    metadata = dict_deep_update(
        metadata,
        editable_metadata,
    )

    metadata["Subject"]["subject_id"] = subject_id
    metadata["Subject"]["date_of_birth"] = subject_metadata["DOB (DD/MM/YYYY)"]
    metadata["Subject"]["genotype"] = subject_metadata["genotype"].upper()
    metadata["Subject"]["strain"] = subject_metadata["line"]
    sex = {"male": "M", "female": "F"}.get(subject_metadata["sex"], "U")
    metadata["Subject"].update(sex=sex)
    # Add session ID to metadata
    metadata["NWBFile"]["session_id"] = session_id
    metadata["NWBFile"]["session_description"] = metadata["SessionTypes"][session_id]["session_description"]

    # Read freeze log file
    freeze_log = pd.read_excel(freeze_log_file_path, header=None)
    # TODO: replace with the actual column names
    row = freeze_log.values[0]
    # Combine into a single datetime object
    try:
        session_start_time = datetime.combine(row[-2].date(), row[-1])
    except AttributeError:
        session_start_time = datetime.combine(datetime.strptime(row[-2], "%d/%m/%Y").date(), row[-1])
    session_start_time = session_start_time.replace(tzinfo=ZoneInfo("Europe/London"))
    metadata["NWBFile"].update(session_start_time=session_start_time)

    # Update devices metadata based on session type
    chamber_to_remove = "ChamberA" if session_id == "AFC_3_Conditioning" else "ChamberB"
    metadata["Devices"] = [device for device in metadata["Devices"] if device["name"] != chamber_to_remove]

    # Run conversion
    converter.run_conversion(
        metadata=metadata,
        nwbfile_path=nwbfile_path,
        conversion_options=conversion_options,
        overwrite=overwrite,
    )

    with NWBHDF5IO(nwbfile_path, mode="r") as io:
        nwbfile_in = io.read()
        print(nwbfile_in.trials.to_dataframe())


if __name__ == "__main__":

    # Parameters for conversion
    data_dir_path = Path("/Users/weian/data/Auditory Fear Conditioning")
    output_dir_path = data_dir_path / "nwbfiles"

    subjects_metadata_file_path = Path("/Users/weian/data/RAT ID metadata Yunkai copy - updated 12.2.25.xlsx")
    task_acronym = "AFC"
    session_ids = get_session_ids_from_excel(
        subjects_metadata_file_path=subjects_metadata_file_path,
        task_acronym=task_acronym,
    )

    subjects_metadata = extract_subject_metadata_from_excel(subjects_metadata_file_path)
    subjects_metadata = get_subject_metadata_from_task(subjects_metadata, task_acronym)

    session_id = session_ids[-2]  # 4_RecallD1
    subject_metadata = subjects_metadata[120]  # subject 831_Grin2b(4)

    cohort_folder_path = data_dir_path / subject_metadata["line"] / f"{subject_metadata['cohort ID']}_{task_acronym}"
    if not cohort_folder_path.exists():
        raise FileNotFoundError(f"Folder {cohort_folder_path} does not exist")

    video_folder_path = cohort_folder_path / session_id

    # TODO: move this to convert_sessions.py
    # _convert_ffii_to_avi(
    #     folder_path=video_folder_path,
    #     convert_ffii_repo_path="convert-ffii",
    #     frame_rate=15, # TODO: confirm frame rate
    # )

    if not video_folder_path.exists():
        raise FileNotFoundError(f"Folder {cohort_folder_path} does not exist")
    video_file_paths = list(video_folder_path.glob(f"*{subject_metadata['animal ID']}*.avi"))
    if len(video_file_paths) == 0:
        raise FileNotFoundError(
            f"No video files found in for animal ID {subject_metadata['animal ID']} in '{video_folder_path}'."
        )
    elif len(video_file_paths) > 1:
        raise FileExistsError(
            f"Multiple video files found for animal ID {subject_metadata['animal ID']} in {video_folder_path}."
        )
    video_file_path = video_file_paths[0]

    freeze_scores_file_paths = list(video_folder_path.glob(f"*{subject_metadata['line']}*.csv"))
    if len(freeze_scores_file_paths):
        freeze_scores_file_path = freeze_scores_file_paths[0]
    else:
        freeze_scores_file_path = None
        warnings.warn(f"No freeze scores file (.csv) found in {video_file_path}.")

    # Path to the excel file containing metadata
    freeze_log_file_path = video_folder_path / "Freeze_Log.xls"

    stub_test = False
    # Whether to overwrite the NWB file if it already exists
    overwrite = True

    session_to_nwb(
        output_dir_path=output_dir_path,
        video_file_path=video_file_path,
        freeze_log_file_path=freeze_log_file_path,
        freeze_scores_file_path=freeze_scores_file_path,
        session_id=f"{task_acronym}_{session_id}",
        subject_metadata=subject_metadata,
        overwrite=overwrite,
    )
