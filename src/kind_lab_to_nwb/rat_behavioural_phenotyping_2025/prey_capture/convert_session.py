import warnings
from pathlib import Path
from typing import List, Optional, Union
from zoneinfo import ZoneInfo

import natsort
from pydantic import FilePath
from pynwb import NWBHDF5IO

from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.interfaces import (
    BORISBehavioralEventsInterface,
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
    parse_datetime_from_filename,
)
from neuroconv.datainterfaces import AudioInterface, ExternalVideoInterface
from neuroconv.utils import dict_deep_update, load_dict_from_file


def session_to_nwb(
    output_dir_path: Union[str, Path],
    video_file_paths: List[Union[FilePath, str]],
    session_id: str,
    subject_metadata: dict,
    boris_file_path: Optional[Union[FilePath, str]] = None,
    usv_file_paths: Optional[List[Union[FilePath, str]]] = None,
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
    usv_file_paths: Optional[List[Union[FilePath, str]]]
        The list of USV file paths to be converted, if available.
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

    conversion_options = dict()

    first_video_file_name = Path(video_file_paths[0]).stem
    session_start_time = parse_datetime_from_filename(first_video_file_name)

    if not any(valid_session_id in session_id for valid_session_id in ["Hab", "Test", "Weeto"]):
        raise ValueError(f"Session ID '{session_id}' is not valid. It should contain 'Hab', 'Test', or 'Weeto'.")

    # Add Behavioral Video
    data_interfaces = []
    # Habitation sessions have only one video file
    if "Hab" in session_id:
        if len(video_file_paths) == 1:
            file_paths = convert_ts_to_mp4(video_file_paths)
            video_interface = ExternalVideoInterface(file_paths=file_paths, video_name="BehavioralVideo")
            data_interfaces.append(video_interface)
        elif len(video_file_paths) > 1:
            raise ValueError(f"Multiple video files found for {subject_id}.")
    # Test sessions have 4-5 video files, weeto trials have 2 video files
    else:
        for i, video_file_path in enumerate(video_file_paths):
            file_paths = convert_ts_to_mp4([video_file_path])
            video_name = f"BehavioralVideoTestTrial{i+1}" if "Test" in session_id else f"BehavioralVideoWeetoTrial{i+1}"
            video_interface = ExternalVideoInterface(file_paths=file_paths, video_name=video_name)
            video_file_name = Path(video_file_path).stem
            datetime_from_filename = parse_datetime_from_filename(video_file_name)
            starting_time = (datetime_from_filename - session_start_time).total_seconds()
            video_interface._starting_time = starting_time
            data_interfaces.append(video_interface)

    # Add Prey Capture Annotated events from BORIS output
    if boris_file_path is not None and "Test" in session_id:
        observation_ids = get_observation_ids(boris_file_path)
        observation_id = next((obs_id for obs_id in observation_ids if subject_metadata["animal ID"] in obs_id), None)
        if observation_id is None:
            raise ValueError(f"No observation ID found containing subject ID '{subject_id}'")
        boris_interface = BORISBehavioralEventsInterface(file_path=boris_file_path, observation_id=observation_id)
        data_interfaces.append(boris_interface)

    # Add USV files
    if usv_file_paths is not None:
        for usv_file_path in usv_file_paths:
            # TODO: remove this when fixed in neuroconv
            old_path = Path(usv_file_path)
            # Replace "4.1.wav" with "4-1.wav"
            new_name = old_path.stem.replace(".", "-") + old_path.suffix
            new_path = old_path.with_name(new_name)
            old_path.rename(new_path)
            audio_interface = AudioInterface(file_paths=[usv_file_path])
            data_interfaces.append(audio_interface)

    converter = PreyCaptureNWBConverter(data_interfaces=data_interfaces, verbose=True)

    metadata = converter.get_metadata()
    # Update default metadata with the editable in the corresponding yaml file
    editable_metadata_path = Path(__file__).parent / "metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    metadata = dict_deep_update(
        metadata,
        editable_metadata,
    )

    videos_metadata = metadata["Behavior"]["ExternalVideos"]
    if len(videos_metadata) > 1 and "BehavioralVideo" in videos_metadata:
        default_metadata = videos_metadata.pop("BehavioralVideo")
        for video_metadata_key, video_metadata in videos_metadata.items():
            video_metadata.update(description=default_metadata["description"], device=default_metadata["device"])

    metadata["Subject"]["subject_id"] = subject_id
    metadata["Subject"]["date_of_birth"] = subject_metadata["DOB (DD/MM/YYYY)"]
    sex = {"male": "M", "female": "F"}.get(subject_metadata["sex"], "U")
    metadata["Subject"].update(sex=sex)
    # Add session ID to metadata
    metadata["NWBFile"]["session_id"] = session_id
    metadata["NWBFile"]["session_description"] = metadata["SessionTypes"][session_id]["session_description"]

    if "session_start_time" not in metadata["NWBFile"]:
        session_start_time = session_start_time.replace(tzinfo=ZoneInfo("Europe/London"))
        metadata["NWBFile"].update(session_start_time=session_start_time)

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

    session_id = session_ids[-3]  # HabD2
    subject_metadata = subjects_metadata[0]  # subject 408_Arid1b(3)

    cohort_folder_path = data_dir_path / subject_metadata["line"] / f"{subject_metadata['cohort ID']}_{task_acronym}"
    if not cohort_folder_path.exists():
        raise FileNotFoundError(f"Folder {cohort_folder_path} does not exist")

    video_folder_path = cohort_folder_path / session_id

    if not video_folder_path.exists():
        raise FileNotFoundError(f"Folder {cohort_folder_path} does not exist")
    # TODO: for HabD1 need to figure out how to match the cage number with the animal ID
    if session_id == "HabD1":
        video_file_paths = natsort.natsorted(video_folder_path.glob(f"**"))
    else:
        video_file_paths = natsort.natsorted(video_folder_path.glob(f"*{subject_metadata['animal ID']}*"))

    # If the data has been scored, the cohort folder would contain BORIS files (.boris) of the behaviors of interest
    # TODO: there are no example boris files for PC shared yet
    boris_file_paths = list(cohort_folder_path.glob("*.boris"))
    if len(boris_file_paths) == 0:
        boris_file_path = None
        warnings.warn(f"No BORIS file found in {cohort_folder_path}")
    else:
        boris_file_path = boris_file_paths[0]

    # Optional, add USV files
    usv_file_paths = natsort.natsorted(video_folder_path.rglob(f"*{subject_metadata['animal ID']}*.wav"))
    if len(usv_file_paths) == 0:
        usv_file_paths = None
        warnings.warn(f"No USV file found in {video_folder_path}")

    stub_test = False
    # Whether to overwrite the NWB file if it already exists
    overwrite = True

    session_to_nwb(
        output_dir_path=output_dir_path,
        video_file_paths=video_file_paths,
        session_id=f"{task_acronym}_{session_id}",
        subject_metadata=subject_metadata,
        usv_file_paths=usv_file_paths,
        overwrite=overwrite,
    )
