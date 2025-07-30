"""Primary script to run to convert an entire session for of data using the NWBConverter."""

from pathlib import Path
from typing import Union
from pydantic import FilePath
import warnings
from datetime import datetime
import numpy as np

from neuroconv.utils import (
    dict_deep_update,
    load_dict_from_file,
)
from neuroconv.datainterfaces import AudioInterface

from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.one_trial_social.nwbconverter import (
    OneTrialSocialNWBConverter,
)
from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.interfaces import (
    BORISBehavioralEventsInterface,
    SpyglassVideoInterface,
    get_observation_ids,
)
from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.utils import (
    extract_subject_metadata_from_excel,
    get_subject_metadata_from_task,
    get_session_ids_from_excel,
    convert_ts_to_mp4,
    parse_datetime_from_filename,
)


def session_to_nwb(
    output_dir_path: Union[str, Path],
    video_file_paths: Union[FilePath, str],
    boris_file_path: Union[FilePath, str],
    subject_metadata: dict,
    session_id: str,
    session_start_time: datetime,
    audio_file_path: Union[FilePath, str] = None,
    stub_test: bool = False,
    overwrite: bool = False,
):

    output_dir_path = Path(output_dir_path)
    if stub_test:
        output_dir_path = output_dir_path / "nwb_stub"
    output_dir_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    subject_id = f"{subject_metadata['animal ID']}_{subject_metadata['cohort ID']}"
    nwbfile_path = output_dir_path / f"sub-{subject_id}_ses-{session_id}.nwb"

    conversion_options = dict()

    editable_metadata_path = Path(__file__).parent / "metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    task_metadata = editable_metadata["SessionTypes"][session_id]

    # Add Behavioral Video
    data_interfaces = []
    video_starting_times = []
    if len(video_file_paths) == 1:
        file_paths = convert_ts_to_mp4(video_file_paths)
        video_interface = SpyglassVideoInterface(file_paths=file_paths, video_name="BehavioralVideo")
        data_interfaces.append(video_interface)
        conversion_options.update(dict(SpyglassVideoInterface=dict(task_metadata=task_metadata)))
    elif len(video_file_paths) > 1:
        if "HabD1" in session_id:
            for i, video_file_path in enumerate(video_file_paths):
                file_paths = convert_ts_to_mp4([video_file_path])
                video_name = f"BehavioralVideoHabD1Trial{i+1}"
                video_interface = SpyglassVideoInterface(file_paths=file_paths, video_name=video_name)
                video_file_name = Path(video_file_path).stem
                datetime_from_filename = parse_datetime_from_filename(video_file_name)
                starting_time = (datetime_from_filename - session_start_time).total_seconds()
                video_starting_times.append(starting_time)
                video_interface._starting_time = starting_time
                data_interfaces.append(video_interface)
                test_task_metadata = task_metadata.copy()
                test_task_metadata["name"] = task_metadata["name"] + "_trial" + str(i + 1)
                test_task_metadata["task_epochs"] = [i + 1]
                conversion_options.update({f"SpyglassVideoInterface00{i+1}": dict(task_metadata=test_task_metadata)})

        else:
            raise ValueError(f"Multiple video files found for {subject_id}.")

    # Add One Trial Social Annotated events from BORIS output
    if boris_file_path is not None and "Test" in session_id:
        observation_ids = get_observation_ids(boris_file_path)
        observation_id = next(
            (obs_id for obs_id in observation_ids if str(subject_metadata["animal ID"]) in obs_id), None
        )
        if observation_id is None:
            raise ValueError(f"No observation ID found containing subject ID '{subject_id}'")
        boris_interface = BORISBehavioralEventsInterface(file_path=boris_file_path, observation_id=observation_id)
        data_interfaces.append(boris_interface)

    # Add Audio if available
    if audio_file_path is not None:
        audio_interface = AudioInterface(file_paths=audio_file_path)
        data_interfaces.append(audio_interface)
        conversion_options.update(dict(AudioInterface=dict(stub_test=stub_test, write_as="acquisition")))

    converter = OneTrialSocialNWBConverter(data_interfaces=data_interfaces, verbose=True)

    # Add datetime to conversion
    metadata = converter.get_metadata()

    # Update default metadata with the editable in the corresponding yaml file
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

    metadata["NWBFile"]["session_id"] = session_id
    metadata["NWBFile"]["session_description"] = metadata["SessionTypes"][session_id]["session_description"]
    experimenters = []
    task_acronym = session_id.split("_")[0]
    if subject_metadata[f"{task_acronym} exp"] is not np.nan:
        experimenters.append(subject_metadata[f"{task_acronym} exp"])
    if (
        subject_metadata[f"{task_acronym} sco"] is not np.nan
        and subject_metadata[f"{task_acronym} sco"] != subject_metadata[f"{task_acronym} exp"]
    ):
        experimenters.append(subject_metadata[f"{task_acronym} sco"])
    metadata["NWBFile"]["experimenter"] = experimenters

    # Check if session_start_time exists in metadata
    if "session_start_time" not in metadata["NWBFile"]:
        metadata["NWBFile"]["session_start_time"] = session_start_time

    # Run conversion
    converter.run_conversion(
        metadata=metadata,
        nwbfile_path=nwbfile_path,
        # nwbfile=nwbfile,
        conversion_options=conversion_options,
        overwrite=overwrite,
    )

    print(f"Conversion completed for {subject_id} session {session_id}.")
    print(f"NWB file saved to {nwbfile_path}.")


if __name__ == "__main__":

    # Parameters for conversion
    data_dir_path = Path("D:/Kind-CN-data-share/behavioural_pipeline/1 Trial Social")
    output_dir_path = Path("D:/kind_lab_conversion_nwb/behavioural_pipeline/1_trial_social")
    subjects_metadata_file_path = Path("D:/Kind-CN-data-share/behavioural_pipeline/general_metadata.xlsx")
    task_acronym = "1TS"
    session_ids = get_session_ids_from_excel(subjects_metadata_file_path, task_acronym)

    subjects_metadata = extract_subject_metadata_from_excel(subjects_metadata_file_path)
    subjects_metadata = get_subject_metadata_from_task(subjects_metadata, task_acronym)

    session_id = session_ids[0]  # Test
    subject_metadata = subjects_metadata[52]  # subject 635Arid1b(11)

    cohort_folder_path = data_dir_path / subject_metadata["line"] / f"{subject_metadata['cohort ID']}_{task_acronym}"
    if not cohort_folder_path.exists():
        raise FileNotFoundError(f"Folder {cohort_folder_path} does not exist")

    # check if boris file exists on the cohort folder
    boris_file_paths = list(cohort_folder_path.glob("*.boris"))
    if len(boris_file_paths) == 0:
        boris_file_path = None
        warnings.warn(f"No BORIS file found in {cohort_folder_path}")
    else:
        boris_file_path = boris_file_paths[0]

    video_folder_path = cohort_folder_path / session_id
    if not video_folder_path.exists():
        raise FileNotFoundError(f"Folder {cohort_folder_path} does not exist")
    video_file_paths = list(video_folder_path.glob(f"*{subject_metadata['animal ID']}*"))
    if len(video_file_paths) == 0:
        cage_id = subject_metadata["cage ID"]
        if not np.isnan(cage_id):
            cage_id = int(cage_id)
            video_file_paths = list(video_folder_path.glob(f"*cage{cage_id}*"))

    session_start_times = []
    for video_file_path in video_file_paths:
        session_start_times.append(parse_datetime_from_filename(video_file_path.name))

    session_start_time = min(session_start_times)

    audio_folder_path = video_folder_path / "USVs"
    if audio_folder_path.exists():
        audio_file_paths = list(audio_folder_path.glob(f"*{subject_metadata['animal ID']}*.wav"))
        if len(audio_file_paths) == 0:
            audio_file_path = None
            warnings.warn(f"No audio file found in {audio_folder_path}")
        elif len(audio_file_paths) > 1:
            raise ValueError(
                f"Multiple audio files for subject {subject_metadata['animal ID']} found in {audio_folder_path}"
            )
        else:
            audio_file_path = audio_file_paths[0]
    else:
        audio_file_path = None

    stub_test = False
    overwrite = True

    session_to_nwb(
        output_dir_path=output_dir_path,
        video_file_paths=video_file_paths,
        boris_file_path=boris_file_path,
        audio_file_path=audio_file_path,
        subject_metadata=subject_metadata,
        session_id=f"{task_acronym}_{session_id}",
        session_start_time=session_start_time,
        stub_test=stub_test,
        overwrite=overwrite,
    )
