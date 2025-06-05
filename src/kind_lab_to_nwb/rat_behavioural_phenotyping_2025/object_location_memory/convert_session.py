"""Primary script to run to convert an entire session for of data using the NWBConverter."""

from pathlib import Path
from typing import Union
from pydantic import FilePath
import warnings
from datetime import datetime
import pandas as pd

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
    convert_ts_to_mp4,
    parse_datetime_from_filename,
)


def get_novelty_information_for_the_object_positions(
    boris_info_file_path: Union[FilePath, str], animal_id: str, session_id: str
) -> dict:
    """
    Extracts novelty information on the objects from the BORIS info file.

    Parameters
    ----------
    boris_info_file_path : Union[FilePath, str]
        Path to the BORIS info file.
    animal_id : str
        Animal ID to filter the data.
    session_id : str
        Session ID to filter the data.

    Returns
    -------
    dict
        Dictionary containing novelty information on the objects.
    """
    if not boris_info_file_path:
        return {}

    df = pd.read_excel(boris_info_file_path)
    # select only relevant rows: where animal_id and session_id are contained in the Filename column
    df = df[df["Filename"].str.contains(animal_id) & df["Filename"].str.contains(session_id)]
    if df.empty:
        warnings.warn(f"No novelty information found for animal {animal_id} in session {session_id}.")
        return {}

    # Initialize the novelty info dictionary
# Initialize result structure
    trial_types = ["sample_trial", "test_trial"]
    object_ids = ["A", "B", "C", "D"]
    num_objects = len(object_ids)
    novelty_info_dict = {
        t: {"position": [None] * num_objects, "novelty": [None] * num_objects}
        for t in trial_types
    }

    # Process each row in the filtered dataframe
    for _, row in df.iterrows():
        filename = row["Filename"]

        # Determine trial type from filename
        if "sample" in filename.lower():
            trial_type = "sample_trial"
        elif "test" in filename.lower():
            trial_type = "test_trial"
        else:
            warnings.warn(f"Could not determine trial type from filename: {filename}")
            continue

        # Extract object information for each object (A, B, C, D)
        object_ids = ["A", "B", "C", "D"]

        for i, object_id in enumerate(object_ids):
            # Get position and novelty information
            pos_col = f"Obj_{object_id}"
            nov_col = f"ID_{object_id}"

            position = row.get(pos_col, None)
            novelty = row.get(nov_col, None)

            # Handle NaN values (convert to None)
            novelty_info_dict[trial_type]["position"][i] = None if pd.isna(position) else position
            novelty_info_dict[trial_type]["novelty"][i] = None if pd.isna(novelty) else novelty

    return novelty_info_dict


def session_to_nwb(
    output_dir_path: Union[str, Path],
    video_file_paths: Union[FilePath, str],
    boris_file_path: Union[FilePath, str],
    boris_info_file_path: Union[FilePath, str],
    subject_metadata: dict,
    session_id: str,
    session_start_time: datetime,
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

    if "STM" in session_id or "LTM" in session_id:
        if len(video_file_paths) == 2:
            file_paths = convert_ts_to_mp4(video_file_paths)
            test_file_paths = [file_path for file_path in file_paths if "test" in file_path.name.lower()]
            sample_file_paths = [file_path for file_path in file_paths if "sample" in file_path.name.lower()]
            source_data.update(
                dict(
                    TestVideo=dict(file_paths=test_file_paths, video_name="BehavioralVideoTestTrial"),
                    SampleVideo=dict(file_paths=sample_file_paths, video_name="BehavioralVideoSampleTrial"),
                )
            )
            conversion_options.update(dict(TestVideo=dict(), SampleVideo=dict()))
        else:
            raise ValueError(
                f"{len(video_file_paths)} video files found for {subject_id}. Expected one video file for the sample trial and one for the test trial."
            )
        # Add Annotated events from BORIS output
        if boris_file_path is not None:
            all_observation_ids = get_observation_ids(boris_file_path)
            observation_ids = [
                obs_id
                for obs_id in all_observation_ids
                if str(subject_metadata["animal ID"]) in obs_id and session_id.replace("OLM_", "") in obs_id
            ]
            if not observation_ids:
                print(f"Observation ID not found in BORIS file {boris_file_path}.")
            else:
                for observation_id in observation_ids:
                    if "sample" in observation_id.lower():
                        source_data.update(
                            dict(
                                SampleObjectLocationMemoryBehavior=dict(
                                    file_path=boris_file_path, observation_id=observation_id
                                )
                            )
                        )
                        conversion_options.update(
                            dict(SampleObjectLocationMemoryBehavior=dict(table_name="SampleTrialBehavioralEvents"))
                        )
                    elif "test" in observation_id.lower():
                        source_data.update(
                            dict(
                                TestObjectLocationMemoryBehavior=dict(
                                    file_path=boris_file_path, observation_id=observation_id
                                )
                            )
                        )
                        conversion_options.update(
                            dict(TestObjectLocationMemoryBehavior=dict(table_name="TestTrialBehavioralEvents"))
                        )
                    else:
                        raise ValueError(f"Observation ID {observation_id} not recognized.")

    else:
        if len(video_file_paths) == 1:
            file_paths = convert_ts_to_mp4(video_file_paths)
            source_data.update(dict(Video=dict(file_paths=file_paths, video_name="BehavioralVideo")))
            conversion_options.update(dict(Video=dict()))
        elif len(video_file_paths) > 1:
            raise ValueError(f"Multiple video files found for {subject_id}.")

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
    metadata["Subject"]["genotype"] = subject_metadata["genotype"].upper()
    metadata["Subject"]["strain"] = subject_metadata["line"]
    sex = {"male": "M", "female": "F"}.get(subject_metadata["sex"], "U")
    metadata["Subject"].update(sex=sex)

    metadata["NWBFile"]["session_id"] = session_id
    metadata["NWBFile"]["session_description"] = metadata["SessionTypes"][session_id]["session_description"]

    # Check if session_start_time exists in metadata
    if "session_start_time" not in metadata["NWBFile"]:
        metadata["NWBFile"]["session_start_time"] = session_start_time

    if "LTM" in session_id:
        # Find and remove Marbles from devices if it exists
        for i, device in enumerate(metadata.get("Devices", [])):
            if device.get("name") == "Arena_STM":
                metadata["Devices"].pop(i)
            break
    elif "STM" in session_id:
        # Find and remove Marbles from devices if it exists
        for i, device in enumerate(metadata.get("Devices", [])):
            if device.get("name") == "Arena_LTM":
                metadata["Devices"].pop(i)
            break
    else:
        # Remove Arena_LTM and Arena_STM from devices if it exists
        for i, device in enumerate(metadata.get("Devices", [])):
            if device.get("name") == "Arena_LTM":
                metadata["Devices"].pop(i)
            elif device.get("name") == "Arena_STM":
                metadata["Devices"].pop(i)
            break

    # Add novelty information on the object position
    if boris_info_file_path is not None:
        novelty_info = get_novelty_information_for_the_object_positions(
            boris_info_file_path, str(subject_metadata["animal ID"]), session_id.replace("OLM_", "")
        )
        # Add novelty information to metadata if needed
        # This could be used to store object position and novelty information in the NWB file
        metadata["NoveltyInformation"] = novelty_info

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
    data_dir_path = Path("D:/Kind-CN-data-share/behavioural_pipeline/Object Location Memory")
    output_dir_path = Path("D:/kind_lab_conversion_nwb/object_location_memory")
    subjects_metadata_file_path = Path("D:/Kind-CN-data-share/behavioural_pipeline/RAT ID metadata Yunkai.xlsx")
    task_acronym = "OLM"
    session_ids = get_session_ids_from_excel(subjects_metadata_file_path, task_acronym)

    subjects_metadata = extract_subject_metadata_from_excel(subjects_metadata_file_path)
    subjects_metadata = get_subject_metadata_from_task(subjects_metadata, task_acronym)

    session_id = session_ids[-2]  # Test STM
    subject_metadata = subjects_metadata[132]  # subject 617Scn2a

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

    # check if boris info file exists
    analysis_folder_path = cohort_folder_path / "Analysis"
    boris_info_file_paths = list(analysis_folder_path.glob("boris_info*.xlsx"))
    if len(boris_info_file_paths) == 0:
        boris_info_file_path = None
        warnings.warn(f"No BORIS info excel file found in {analysis_folder_path}")
    else:
        boris_info_file_path = boris_info_file_paths[0]

    video_folder_path = cohort_folder_path / session_id
    if not video_folder_path.exists():
        raise FileNotFoundError(f"Folder {cohort_folder_path} does not exist")
    video_file_paths = list(video_folder_path.glob(f"*{subject_metadata['animal ID']}*"))

    session_start_times = []
    for video_file_path in video_file_paths:
        session_start_times.append(parse_datetime_from_filename(video_file_path.name))

    session_start_time = min(session_start_times)

    stub_test = False
    overwrite = True

    if "Test" in session_id:
        session_id = f"{session_id.split(' ')[1]}"

    session_to_nwb(
        output_dir_path=output_dir_path,
        video_file_paths=video_file_paths,
        boris_file_path=boris_file_path,
        boris_info_file_path=boris_info_file_path,
        subject_metadata=subject_metadata,
        session_id=f"{task_acronym}_{session_id}",
        session_start_time=session_start_time,
        stub_test=stub_test,
        overwrite=overwrite,
    )
