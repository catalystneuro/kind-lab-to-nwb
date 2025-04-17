"""Primary script to run to convert an entire session for of data using the NWBConverter."""

from pathlib import Path
from typing import Union
from pydantic import FilePath
import warnings
from datetime import datetime

from neuroconv.utils import (
    dict_deep_update,
    load_dict_from_file,
)
from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.object_location_memory.nwbconverter import (
    ObjectLocationMemoryNWBConverter,
)
from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.interfaces import get_observation_ids
from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.utils import (
    extract_subject_metadata_from_excel,
    get_subject_metadata_from_task,
    get_session_ids_from_excel,
    make_ndx_event_nwbfile_from_metadata,
)
from pynwb.device import Device


def session_to_nwb(
    output_dir_path: Union[str, Path],
    video_file_paths: Union[FilePath, str],
    boris_file_path: Union[FilePath, str],
    subject_metadata: dict,
    session_id: str,
    stub_test: bool = False,
    overwrite: bool = False,
):
    subject_id = f"{subject_metadata['animal ID']}_{subject_metadata['cohort ID']}"

    if not video_file_paths:
        warnings.warn(
            f"No video file found. NWB file will not be created for subject {subject_id} session {session_id}."
        )
        return

    output_dir_path = Path(output_dir_path)
    if stub_test:
        output_dir_path = output_dir_path / "nwb_stub"
    output_dir_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    nwbfile_path = output_dir_path / f"sub-{subject_id}_ses-{session_id}.nwb"

    source_data = dict()
    conversion_options = dict()

    # Add Behavioral Video
    if len(video_file_paths) == 1:
        file_path = video_file_paths[0]
        source_data.update(dict(Video=dict(file_path=file_path, video_name="BehavioralVideo")))
        conversion_options.update(dict(Video=dict(stub_test=stub_test)))
    elif len(video_file_paths) > 1:
        raise ValueError(f"Multiple video files found for {subject_id}.")

    # Add Marble Interaction Annotated events from BORIS output
    if boris_file_path is not None and "test" in session_id:
        observation_ids = get_observation_ids(boris_file_path)
        observation_id = next(
            (
                obs_id
                for obs_id in observation_ids
                if str(subject_metadata["animal ID"]) in obs_id
                and session_id.replace("OR_", "").lower() in obs_id.lower()
            ),
            None,
        )
        if observation_id is not None:
            source_data.update(
                dict(ObjectRecognitionBehavior=dict(file_path=boris_file_path, observation_id=observation_id))
            )
            conversion_options.update(dict(ObjectRecognitionBehavior=dict()))
        else:
            subject_id(f"Observation ID not found in BORIS file {boris_file_path}.")

    converter = ObjectLocationMemoryNWBConverter(source_data=source_data)

    # Add datetime to conversion
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
    # TODO add genotype

    metadata["NWBFile"]["session_id"] = session_id
    metadata["NWBFile"]["session_description"] = metadata["SessionTypes"][session_id]["session_description"]

    # Check if session_start_time exists in metadata
    if "session_start_time" not in metadata["NWBFile"]:
        # Extract date from first video filename
        video_path = Path(video_file_paths[0])
        date_str = video_path.stem.split("_")[0]  # Get "2022-11-23" from filename
        try:
            # Convert to datetime
            session_start_time = datetime.strptime(date_str, "%Y-%m-%d")
            metadata["NWBFile"]["session_start_time"] = session_start_time
        except ValueError:
            warnings.warn(f"Could not parse session start time from video filename {video_path.name}")

    nwbfile = make_ndx_event_nwbfile_from_metadata(metadata=metadata)

    # Add other devices to the NWB file
    for device_metadata in metadata["Devices"]:
        # Add the device to the NWB file
        device = Device(**device_metadata)
        nwbfile.add_device(device)

    # Run conversion
    converter.run_conversion(
        metadata=metadata,
        nwbfile_path=nwbfile_path,
        nwbfile=nwbfile,
        conversion_options=conversion_options,
        overwrite=overwrite,
    )


if __name__ == "__main__":

    # Parameters for conversion
    data_dir_path = Path("D:/Kind-CN-data-share/behavioural_pipeline/Object Recognition")
    output_dir_path = Path("D:/kind_lab_conversion_nwb/object_recognition")
    subjects_metadata_file_path = Path("D:/Kind-CN-data-share/behavioural_pipeline/RAT ID metadata Yunkai.xlsx")
    task_acronym = "OLM"
    session_ids = get_session_ids_from_excel(subjects_metadata_file_path, task_acronym)

    subjects_metadata = extract_subject_metadata_from_excel(subjects_metadata_file_path)
    subjects_metadata = get_subject_metadata_from_task(subjects_metadata, task_acronym)

    session_id = session_ids[-1]  # Test
    subject_metadata = subjects_metadata[137]  # subject 617Scn2a

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

    stub_test = True
    overwrite = True

    if "Test" in session_id:
        session_id = f"{session_id.split(' ')[1]}_{session_id.split(' ')[0].lower()}"
        video_file_paths = [
            video_file_path for video_file_path in video_file_paths if "test" in video_file_path.name.lower()
        ]

    session_to_nwb(
        output_dir_path=output_dir_path,
        video_file_paths=video_file_paths,
        boris_file_path=boris_file_path,
        subject_metadata=subject_metadata,
        session_id=f"{task_acronym}_{session_id}",
        stub_test=stub_test,
        overwrite=overwrite,
    )
