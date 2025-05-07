import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union
from zoneinfo import ZoneInfo

from pydantic import FilePath

from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.interfaces import (
    get_observation_ids,
)
from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.prey_capture import (
    PreyCaptureNWBConverter,
)
from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.utils import (
    convert_ts_to_mp4,
    extract_subject_metadata_from_excel,
    get_session_ids_from_excel,
    get_subject_metadata_from_task,
)
from neuroconv.utils import dict_deep_update, load_dict_from_file


def session_to_nwb(
    output_dir_path: Union[str, Path],
    video_file_paths: List[Union[FilePath, str]],
    session_id: str,
    subject_metadata: dict,
    boris_file_path: Optional[Union[FilePath, str]] = None,
    overwrite: bool = False,
):
    """
    Convert a session of prey capture task to NWB format.

    Parameters
    ----------
    output_dir_path : Union[str, Path]
        The folder path where the NWB file will be saved.
    video_file_paths: List[Union[FilePath, str]]
        The list of video file paths to be converted.
    session_id: str
        The session ID to be used in the metadata.
    subject_metadata: dict
        The metadata for the subject, including animal ID and cohort ID.
    boris_file_path: Optional[Union[FilePath, str]]
        The path to the BORIS file (.boris) for the session, if available.
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

    # Add Behavioral Video
    if len(video_file_paths) == 1:
        file_paths = convert_ts_to_mp4(video_file_paths)
        source_data.update(dict(Video=dict(file_paths=file_paths, video_name="BehavioralVideo")))
        conversion_options.update(dict(Video=dict()))
    elif len(video_file_paths) > 1:
        raise ValueError(f"Multiple video files found for {subject_id}.")

    # Add Prey Capture Annotated events from BORIS output
    if boris_file_path is not None and "Test" in session_id:
        observation_ids = get_observation_ids(boris_file_path)
        observation_id = next((obs_id for obs_id in observation_ids if subject_metadata["animal ID"] in obs_id), None)
        if observation_id is None:
            raise ValueError(f"No observation ID found containing subject ID '{subject_id}'")
        source_data.update(dict(Behavior=dict(file_path=boris_file_path, observation_id=observation_id)))
        conversion_options.update(dict(Behavior=dict()))

    converter = PreyCaptureNWBConverter(source_data=source_data, verbose=True)

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
    sex = {"male": "M", "female": "F"}.get(subject_metadata["sex"], "U")
    metadata["Subject"].update(sex=sex)
    # Add session ID to metadata
    metadata["NWBFile"]["session_id"] = session_id
    metadata["NWBFile"]["session_description"] = metadata["SessionTypes"][session_id]["session_description"]

    if "session_start_time" not in metadata["NWBFile"]:
        video_path = Path(video_file_paths[0])
        video_date_time_parts = video_path.stem.split("_")[:-1]
        session_start_time = " ".join(video_date_time_parts)
        try:
            # Convert to datetime
            session_start_time = datetime.strptime(session_start_time, "%Y-%m-%d %H-%M-%S")
            session_start_time = session_start_time.replace(tzinfo=ZoneInfo("Europe/London"))
            metadata["NWBFile"]["session_start_time"] = session_start_time
        except ValueError:
            warnings.warn(f"Could not parse session start time from video filename {video_path.name}")

    # Run conversion
    converter.run_conversion(
        metadata=metadata,
        nwbfile_path=nwbfile_path,
        conversion_options=conversion_options,
        overwrite=overwrite,
    )

    # with NWBHDF5IO(nwbfile_path, mode="r") as io:
    #     nwbfile_in = io.read()
    #     print(nwbfile_in.trials.to_dataframe())


if __name__ == "__main__":

    # Parameters for conversion
    data_dir_path = Path("/Volumes/T9/Behavioural Pipeline/Prey Capture")
    output_dir_path = data_dir_path / "nwbfiles"

    subjects_metadata_file_path = Path("/Users/weian/data/RAT ID metadata Yunkai copy - updated 12.2.25.xlsx")
    task_acronym = "PC"
    session_ids = get_session_ids_from_excel(
        subjects_metadata_file_path=subjects_metadata_file_path,
        task_acronym=task_acronym,
    )

    subjects_metadata = extract_subject_metadata_from_excel(subjects_metadata_file_path)
    subjects_metadata = get_subject_metadata_from_task(subjects_metadata, task_acronym)

    session_id = session_ids[1]  # HabD2
    subject_metadata = subjects_metadata[0]  # subject 408_Arid1b(3)

    cohort_folder_path = data_dir_path / subject_metadata["line"] / f"{subject_metadata['cohort ID']}_{task_acronym}"
    if not cohort_folder_path.exists():
        raise FileNotFoundError(f"Folder {cohort_folder_path} does not exist")

    video_folder_path = cohort_folder_path / session_id

    if not video_folder_path.exists():
        raise FileNotFoundError(f"Folder {cohort_folder_path} does not exist")
    # TODO: for HabD1 need to figure out how to match the cage number with the animal ID
    if session_id == "HabD1":
        video_file_paths = list(video_folder_path.glob(f"**"))
    else:
        video_file_paths = list(video_folder_path.glob(f"*{subject_metadata['animal ID']}*"))

    # If the data has been scored, the cohort folder would contain BORIS files (.boris) of the behaviors of interest
    # TODO: there are no example boris files for PC shared yet
    boris_file_paths = list(cohort_folder_path.glob("*.boris"))
    if len(boris_file_paths) == 0:
        boris_file_path = None
        warnings.warn(f"No BORIS file found in {cohort_folder_path}")
    else:
        boris_file_path = boris_file_paths[0]

    stub_test = False
    # Whether to overwrite the NWB file if it already exists
    overwrite = True

    session_to_nwb(
        output_dir_path=output_dir_path,
        video_file_paths=video_file_paths,
        session_id=f"{task_acronym}_{session_id}",
        subject_metadata=subject_metadata,
        overwrite=overwrite,
    )
