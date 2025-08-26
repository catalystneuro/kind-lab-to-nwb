import warnings
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Union
from zoneinfo import ZoneInfo
import numpy as np

import natsort
from pydantic import FilePath

from neuroconv.datainterfaces import AudioInterface, ExternalVideoInterface
from neuroconv.utils import dict_deep_update, load_dict_from_file

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


def session_to_nwb(
    output_dir_path: Union[str, Path],
    video_file_paths: List[Union[FilePath, str]],
    session_id: str,
    subject_metadata: dict,
    boris_file_path: Optional[Union[FilePath, str]] = None,
    usv_file_paths: Optional[List[Union[FilePath, str]]] = None,
    usv_starting_times: Optional[List[float]] = None,
    stub_test: bool = False,
    overwrite: bool = False,
):
    """
    Convert a session of prey capture task to NWB format.

    Parameters
    ----------
    output_dir_path : Union[str, Path]
        The directory where the NWB file will be saved.
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
    usv_starting_times: Optional[List[float]]
        The list of starting times for the USV files, if available.
    overwrite: bool, optional
        Whether to overwrite the NWB file if it already exists, by default False.
    """
    output_dir_path = Path(output_dir_path)
    if stub_test:
        output_dir_path = output_dir_path / "nwb_stub"
    output_dir_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    subject_id = f"{subject_metadata['animal ID']}-{subject_metadata['cohort ID']}"
    nwbfile_path = output_dir_path / f"sub-{subject_id}_ses-{session_id}.nwb"

    conversion_options = dict()

    first_video_file_name = Path(video_file_paths[0]).stem
    session_start_time = parse_datetime_from_filename(first_video_file_name)

    if not any(valid_session_id in session_id for valid_session_id in ["Hab", "Test", "Weeto"]):
        raise ValueError(f"Session ID '{session_id}' is not valid. It should contain 'Hab', 'Test', or 'Weeto'.")

    # Add Behavioral Video
    data_interfaces = []
    video_starting_times = []
    # Habitation sessions have only one video file
    if "Hab" in session_id:
        if len(video_file_paths) == 1:
            file_paths = convert_ts_to_mp4(video_file_paths)
            video_interface = ExternalVideoInterface(file_paths=file_paths, video_name="BehavioralVideo")
            data_interfaces.append(video_interface)
        elif len(video_file_paths) > 1:
            # For Scn2a(11) cohort there are two videos per session, one for cricket and one for weeto trial.
            if subject_metadata["cohort ID"] == "Scn2a(11)":
                for i, video_file_path in enumerate(video_file_paths):
                    file_paths = convert_ts_to_mp4([video_file_path])
                    video_file_name = Path(video_file_path).stem
                    trial_name = video_file_name.split("_")[-1]
                    video_name = f"BehavioralVideo{trial_name.capitalize()}"
                    video_interface = ExternalVideoInterface(file_paths=file_paths, video_name=video_name)
                    datetime_from_filename = parse_datetime_from_filename(video_file_name)
                    starting_time = (datetime_from_filename - session_start_time).total_seconds()
                    video_starting_times.append(starting_time)
                    video_interface._starting_time = starting_time
                    data_interfaces.append(video_interface)

            else:
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
            video_starting_times.append(starting_time)
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
        audio_interface = AudioInterface(file_paths=usv_file_paths)
        if usv_starting_times is not None:
            assert len(usv_file_paths) == len(usv_starting_times), (
                f"Number of USV files ({len(usv_file_paths)}) does not match number of starting times "
                f"({len(usv_starting_times)})."
            )
            audio_interface._segment_starting_times = usv_starting_times
        data_interfaces.append(audio_interface)
        conversion_options.update(dict(AudioInterface=dict(write_as="acquisition")))  # TODO: stub_test

    converter = PreyCaptureNWBConverter(data_interfaces=data_interfaces, verbose=False)

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
    audios_metadata = metadata["Behavior"]["Audio"]
    if len(audios_metadata) > 1:
        default_metadata_copy = deepcopy(audios_metadata[0])
        for i, audio_metadata in enumerate(audios_metadata):
            audio_metadata.update(
                name=f"AcousticWaveformSeriesTestTrial{i+1}", description=default_metadata_copy["description"]
            )
    elif usv_file_paths is None and len(audios_metadata):
        # If no USV files are provided, remove the audio metadata
        metadata["Behavior"].pop("Audio")

    metadata["Subject"]["subject_id"] = subject_id
    metadata["Subject"][
        "description"
    ] = f"Subject housed in {subject_metadata['housing']} housing conditions. Cage identifier: {subject_metadata['cage ID']}."
    metadata["Subject"]["date_of_birth"] = subject_metadata["DOB (DD/MM/YYYY)"]
    metadata["Subject"]["genotype"] = subject_metadata["genotype"].upper()
    metadata["Subject"]["strain"] = subject_metadata["line"]
    sex = {"male": "M", "female": "F"}.get(subject_metadata["sex"], "U")
    metadata["Subject"].update(sex=sex)
    # Add session ID to metadata
    metadata["NWBFile"]["session_id"] = session_id
    metadata["NWBFile"]["session_description"] = metadata["SessionTypes"][session_id]["session_description"]
    metadata["NWBFile"]["session_description"] = metadata["SessionTypes"][session_id]["session_description"]
    experimenters = []
    task_acronym = session_id.split("_")[0]
    if str(subject_metadata[f"{task_acronym} exp"]) != "nan":
        experimenters.append(subject_metadata[f"{task_acronym} exp"])
    if (
        str(subject_metadata[f"{task_acronym} sco"]) != "nan"
        and subject_metadata[f"{task_acronym} sco"] != subject_metadata[f"{task_acronym} exp"]
    ):
        experimenters.append(subject_metadata[f"{task_acronym} sco"])
    metadata["NWBFile"]["experimenter"] = experimenters

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

    print(f"Conversion completed for {subject_id} session {session_id}.")
    print(f"NWB file saved to {nwbfile_path}.")


if __name__ == "__main__":

    # Parameters for conversion
    data_dir_path = Path("D:/Kind-CN-data-share/behavioural_pipeline/Prey Capture")
    output_dir_path = Path("D:/kind_lab_conversion_nwb/behavioural_pipeline/prey_capture")
    subjects_metadata_file_path = Path("D:/Kind-CN-data-share/behavioural_pipeline/general_metadata.xlsx")
    task_acronym = "PC"
    session_ids = get_session_ids_from_excel(
        subjects_metadata_file_path=subjects_metadata_file_path,
        task_acronym=task_acronym,
    )

    subjects_metadata = extract_subject_metadata_from_excel(subjects_metadata_file_path)
    subjects_metadata = get_subject_metadata_from_task(subjects_metadata, task_acronym)

    session_id = session_ids[0]  # HabD1
    subject_metadata = subjects_metadata[0]  # subject 408_Arid1b(3)

    cohort_folder_path = data_dir_path / subject_metadata["line"] / f"{subject_metadata['cohort ID']}_{task_acronym}"
    if not cohort_folder_path.exists():
        raise FileNotFoundError(f"Folder {cohort_folder_path} does not exist")

    # check if boris file exists on the cohort folder
    # TODO: there are no example boris files for PC shared yet
    boris_file_paths = list(cohort_folder_path.glob("*.boris"))
    if len(boris_file_paths) == 0:
        boris_file_path = None
        warnings.warn(f"No BORIS file found in {cohort_folder_path}")
    else:
        boris_file_path = boris_file_paths[0]

    video_folder_path = cohort_folder_path / session_id
    if not video_folder_path.exists():
        raise FileNotFoundError(f"Folder {cohort_folder_path} does not exist")
    video_file_paths = natsort.natsorted(video_folder_path.glob(f"*{subject_metadata['animal ID']}*"))
    if len(video_file_paths) == 0:
        cage_id = subject_metadata["cage ID"]
        if not np.isnan(cage_id):
            cage_id = int(cage_id)
            video_file_paths = natsort.natsorted(video_folder_path.glob(f"*cage{cage_id}*"))

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
