from pathlib import Path
from typing import Union
from zoneinfo import ZoneInfo

from pydantic import FilePath

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
    video_file_path: Union[FilePath, str],
    session_id: str,
    subject_metadata: dict,
    overwrite: bool = False,
):
    """
    Convert a session of auditory fear conditioning task to NWB format.

    Parameters
    ----------
    output_dir_path : Union[str, Path]
        The folder path where the NWB file will be saved.
    video_file_path: Union[FilePath, str]
        The path to the video file to be converted.
    session_id: str
        The session ID to be used in the metadata.
    subject_metadata: dict
        The metadata for the subject, including animal ID and cohort ID.
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
    video_file_paths = convert_ts_to_mp4(video_file_paths=[video_file_path])
    # TODO: can there be multiple video files per session?
    source_data.update(dict(Video=dict(file_paths=video_file_paths, video_name="BehavioralVideo")))
    conversion_options.update(dict(Video=dict()))

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

    # TODO: get session start time from video name
    session_start_time = None
    session_start_time = session_start_time.replace(tzinfo=ZoneInfo("Europe/London"))
    metadata["NWBFile"].update(session_start_time=session_start_time)

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
    data_dir_path = Path("/Users/weian/data/Prey Capture")
    output_dir_path = data_dir_path / "nwbfiles"

    subjects_metadata_file_path = Path("/Users/weian/data/RAT ID metadata Yunkai copy - updated 12.2.25.xlsx")
    task_acronym = "PC"
    session_ids = get_session_ids_from_excel(
        subjects_metadata_file_path=subjects_metadata_file_path,
        task_acronym=task_acronym,
    )

    subjects_metadata = extract_subject_metadata_from_excel(subjects_metadata_file_path)
    subjects_metadata = get_subject_metadata_from_task(subjects_metadata, task_acronym)

    session_id = session_ids[0]  # 1_HabD1
    subject_metadata = subjects_metadata[0]  # subject 408_Arid1b(3)

    cohort_folder_path = data_dir_path / subject_metadata["line"] / f"{subject_metadata['cohort ID']}_{task_acronym}"
    if not cohort_folder_path.exists():
        raise FileNotFoundError(f"Folder {cohort_folder_path} does not exist")

    video_folder_path = cohort_folder_path / session_id

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

    stub_test = False
    # Whether to overwrite the NWB file if it already exists
    overwrite = True

    session_to_nwb(
        output_dir_path=output_dir_path,
        video_file_path=video_file_path,
        session_id=f"{task_acronym}_{session_id}",
        subject_metadata=subject_metadata,
        overwrite=overwrite,
    )
