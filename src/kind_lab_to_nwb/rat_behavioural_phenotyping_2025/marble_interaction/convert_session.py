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

from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.marble_interaction.nwbconverter import (
    MarbleInteractionNWBConverter,
)
from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.interfaces import get_observation_ids
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

    source_data = dict()
    conversion_options = dict()

    editable_metadata_path = Path(__file__).parent / "metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    task_metadata = editable_metadata["SessionTypes"][session_id]

    # Add Behavioral Video
    if len(video_file_paths) == 1:
        file_paths = convert_ts_to_mp4(video_file_paths)
        source_data.update(dict(Video=dict(file_paths=file_paths, video_name="BehavioralVideo")))
        conversion_options.update(dict(Video=dict(task_metadata=task_metadata)))
    elif len(video_file_paths) > 1:
        raise ValueError(f"Multiple video files found for {subject_id}.")

    # Add Marble Interaction Annotated events from BORIS output
    if boris_file_path is not None and "Test" in session_id:
        observation_ids = get_observation_ids(boris_file_path)
        observation_id = next((obs_id for obs_id in observation_ids if subject_id.lower() in obs_id), None)
        if observation_id is None:
            raise ValueError(f"No observation ID found containing subject ID '{subject_id}'")
        source_data.update(
            dict(MarbleInteractionBehavior=dict(file_path=boris_file_path, observation_id=observation_id))
        )
        conversion_options.update(dict(MarbleInteractionBehavior=dict()))

    converter = MarbleInteractionNWBConverter(source_data=source_data)

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

    # Add other devices to the NWB file
    if "Test" not in session_id:
        # Find and remove Marbles from devices if it exists
        for i, device in enumerate(metadata.get("Devices", [])):
            if device.get("name") == "Marbles":
                metadata["Devices"].pop(i)
            break

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
    data_dir_path = Path("D:/Kind-CN-data-share/behavioural_pipeline/Marble Interaction")
    output_dir_path = Path("D:/kind_lab_conversion_nwb/behavioural_pipeline/marble_interaction")
    subjects_metadata_file_path = Path("D:/Kind-CN-data-share/behavioural_pipeline/general_metadata.xlsx")
    task_acronym = "MI"
    session_ids = get_session_ids_from_excel(subjects_metadata_file_path, task_acronym)

    subjects_metadata = extract_subject_metadata_from_excel(subjects_metadata_file_path)
    subjects_metadata = get_subject_metadata_from_task(subjects_metadata, task_acronym)

    session_id = session_ids[-1]  # Test
    subject_metadata = subjects_metadata[13]  # subject 408Arid1b

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

    video_path = Path(video_file_paths[0])
    session_start_time = parse_datetime_from_filename(video_path.name)

    stub_test = False
    overwrite = True

    session_to_nwb(
        output_dir_path=output_dir_path,
        video_file_paths=video_file_paths,
        boris_file_path=boris_file_path,
        subject_metadata=subject_metadata,
        session_id=f"{task_acronym}_{session_id}",
        session_start_time=session_start_time,
        stub_test=stub_test,
        overwrite=overwrite,
    )
