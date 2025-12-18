"""Primary script to run to convert an entire session for of data using the NWBConverter."""

import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import FilePath, DirectoryPath

from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.auditory_fear_conditioning import (
    AuditoryFearConditioningNWBConverter,
)
from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.utils import (
    extract_subject_metadata_from_excel,
    get_session_ids_from_excel,
    get_subject_metadata_from_task,
)
from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.utils.utils import (
    convert_ffii_files_to_avi,
)
from neuroconv.utils import dict_deep_update, load_dict_from_file


def session_to_nwb(
    output_dir_path: DirectoryPath,
    video_file_path: FilePath,
    freeze_log_file_path: FilePath,
    session_id: str,
    subject_metadata: dict,
    freeze_scores_file_path: None | FilePath = None,
    overwrite: bool = False,
):
    """
    Convert a session of auditory fear conditioning task to NWB format.

    Parameters
    ----------
    output_dir_path : DirectoryPath
        The path where the NWB file will be saved.
    video_file_path: FilePath
        The path to the video file (.avi) to be converted.
    freeze_log_file_path: FilePath
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

    # Update default metadata with the editable in the corresponding yaml file
    editable_metadata_path = Path(__file__).parent / "metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    task_metadata = editable_metadata["SessionTypes"][session_id]

    # Add Behavioral Video
    file_paths = convert_ffii_files_to_avi([str(video_file_path)])
    source_data.update(dict(Video=dict(file_paths=file_paths, video_name="BehavioralVideo")))
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
    metadata["Subject"][
        "description"
    ] = f"Subject housed in {subject_metadata['housing']} housing conditions. Cage identifier: {subject_metadata['cage ID']}."
    metadata["Subject"]["date_of_birth"] = subject_metadata["DOB (DD/MM/YYYY)"]
    metadata["Subject"]["genotype"] = subject_metadata["genotype"].upper()
    metadata["Subject"]["strain"] = subject_metadata["line"]
    sex = {"male": "M", "female": "F"}.get(subject_metadata["sex"], "U")
    metadata["Subject"].update(sex=sex)

    metadata["NWBFile"]["session_id"] = f"{session_id}"
    metadata["NWBFile"]["session_description"] = metadata["SessionTypes"][session_id]["session_description"]
    experimenters = []
    task_acronym = session_id.split("_")[0]
    if not pd.isna(subject_metadata[f"{task_acronym} exp"]):
        experimenters.append(subject_metadata[f"{task_acronym} exp"])
    if (
        not pd.isna(subject_metadata[f"{task_acronym} sco"])
        and subject_metadata[f"{task_acronym} sco"] != subject_metadata[f"{task_acronym} exp"]
    ):
        experimenters.append(subject_metadata[f"{task_acronym} sco"])
    metadata["NWBFile"]["experimenter"] = experimenters

    # Read freeze log file
    try:
        freeze_log = pd.read_csv(freeze_log_file_path, sep="\t", header=None)
    except:
        freeze_log = pd.read_excel(freeze_log_file_path, header=None)
    matching_rows = []
    for row in freeze_log.values:
        if str(subject_metadata["animal ID"]) in row[0]:
            matching_rows.append(row)
    if len(matching_rows) == 0:
        raise ValueError(
            f"No matching rows found in freeze log file '{freeze_log_file_path}' for subject ID '{subject_metadata['animal ID']}'."
        )
    elif len(matching_rows) > 1:
        raise ValueError(
            f"Multiple matching rows found in freeze log file '{freeze_log_file_path}' for subject ID '{subject_metadata['animal ID']}'."
        )
    row = matching_rows[0]
    # Combine into a single datetime object
    try:
        session_start_time = datetime.combine(row[-2].date(), row[-1])
    except:
        session_start_time = datetime.combine(
            datetime.strptime(row[-2], "%d/%m/%Y").date(), datetime.strptime(row[-1], "%H:%M").time()
        )
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


if __name__ == "__main__":

    # Parameters for conversion
    data_dir_path = Path("E:/Kind-CN-data-share/behavioural_pipeline/Auditory Fear Conditioning")
    output_dir_path = Path("E:/kind_lab_conversion_nwb/behavioural_pipeline/auditory_fear_conditioning")
    subjects_metadata_file_path = Path("E:/Kind-CN-data-share/behavioural_pipeline/general_metadata.xlsx")
    task_acronym = "AFC"
    session_ids = get_session_ids_from_excel(
        subjects_metadata_file_path=subjects_metadata_file_path,
        task_acronym=task_acronym,
    )

    subjects_metadata = extract_subject_metadata_from_excel(subjects_metadata_file_path)
    subjects_metadata = get_subject_metadata_from_task(subjects_metadata, task_acronym)

    session_id = session_ids[2]  # 2_HabD2
    subject_metadata = subjects_metadata[0]  # subject 635_Arid1b(11)

    cohort_folder_path = data_dir_path / subject_metadata["line"] / f"{subject_metadata['cohort ID']}_{task_acronym}"
    if not cohort_folder_path.exists():
        raise FileNotFoundError(f"Folder {cohort_folder_path} does not exist")

    video_folder_path = cohort_folder_path / session_id

    if not video_folder_path.exists():
        raise FileNotFoundError(f"Folder {cohort_folder_path} does not exist")
    video_file_paths = list(video_folder_path.glob(f"*{subject_metadata['animal ID']}*.ffii"))
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
        warnings.warn(f"No freeze scores file (.csv) found in {video_folder_path}.")

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
