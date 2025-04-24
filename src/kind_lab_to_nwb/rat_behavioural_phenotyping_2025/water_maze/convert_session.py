from pathlib import Path
from typing import List, Optional, Union

import natsort
import pandas as pd
from pynwb import NWBHDF5IO

from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.utils import (
    extract_subject_metadata_from_excel,
    get_session_ids_from_excel,
    get_subject_metadata_from_task,
)
from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.water_maze import (
    WaterMazeNWBConverter,
    add_trials_to_nwbfile,
)
from neuroconv.tools.nwb_helpers import configure_and_write_nwbfile
from neuroconv.utils import dict_deep_update, load_dict_from_file


def session_to_nwb(
    output_dir_path: Union[str, Path],
    video_file_paths: List[Union[str, Path]],
    video_timestamps: pd.Series,
    session_id: str,
    subject_metadata: dict,
    trials: Optional[pd.DataFrame] = None,
    overwrite: bool = False,
):
    """
    Convert a session of auditory fear conditioning task to NWB format.

    Parameters
    ----------
    output_dir_path : Union[str, Path]
        The folder path where the NWB file will be saved.
    video_file_paths: List[Union[str, Path]]
        The list of paths to the video files (.avi) to be converted.
        The number of video files should be 4 that corresponds to the number of swims during the day.
    video_timestamps: pd.Series
        The timestamps of the videos in the format of pandas Series.
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
    if trials is not None:
        day_num = int(trials["Day"].iloc[0])
        nwbfile_path = output_dir_path / f"sub-{subject_id}_ses-{session_id}_D{day_num}.nwb"

    source_data = dict()
    conversion_options = dict()

    # Add Behavioral Video
    sorted_video_file_paths = natsort.natsorted(video_file_paths)
    if len(sorted_video_file_paths) != 4:
        raise ValueError("Expected 4 video files for the session.")
    for video_index, video_file_path in enumerate(sorted_video_file_paths):
        video_name = f"BehavioralVideoTrial{video_index + 1}"
        source_data.update({f"VideoTrial{video_index + 1}": dict(file_paths=[video_file_path], video_name=video_name)})

    converter = WaterMazeNWBConverter(source_data=source_data, verbose=True)

    # Update starting time of videos
    # TBD
    for i in range(1, len(video_timestamps)):
        video_starting_time = (video_timestamps.iloc[i] - video_timestamps.iloc[0]).total_seconds()
        converter.data_interface_objects[f"VideoTrial{i + 1}"]._starting_time = video_starting_time

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

    session_start_time = video_timestamps.iloc[0]
    metadata["NWBFile"]["session_start_time"] = session_start_time.tz_localize("Europe/London")

    # Run conversion
    nwbfile = converter.create_nwbfile(metadata=metadata, conversion_options=conversion_options)
    add_trials_to_nwbfile(nwbfile=nwbfile, trials=trials)
    configure_and_write_nwbfile(
        nwbfile=nwbfile,
        nwbfile_path=nwbfile_path,
    )
    # converter.run_conversion(
    #     metadata=metadata,
    #     nwbfile_path=nwbfile_path,
    #     conversion_options=conversion_options,
    #     overwrite=overwrite,
    # )

    with NWBHDF5IO(nwbfile_path, mode="r") as io:
        nwbfile_in = io.read()
        print(nwbfile_in.acquisition)
        print(nwbfile_in.trials[:])


if __name__ == "__main__":

    # Parameters for conversion
    data_dir_path = Path("/Users/weian/data/Water Maze")
    output_dir_path = data_dir_path / "nwbfiles"

    subjects_metadata_file_path = Path("/Users/weian/data/RAT ID metadata Yunkai copy - updated 12.2.25.xlsx")
    task_acronym = "WM"
    session_ids = get_session_ids_from_excel(
        subjects_metadata_file_path=subjects_metadata_file_path,
        task_acronym=task_acronym,
    )

    subjects_metadata = extract_subject_metadata_from_excel(subjects_metadata_file_path)
    subjects_metadata = get_subject_metadata_from_task(subjects_metadata, task_acronym)

    session_id = session_ids[1]  # 2_Reference
    subject_metadata = subjects_metadata[0]  # subject 302_Arid1b(2)

    cohort_folder_path = data_dir_path / subject_metadata["line"] / f"{subject_metadata['cohort ID']}_{task_acronym}"
    if not cohort_folder_path.exists():
        raise FileNotFoundError(f"Folder {cohort_folder_path} does not exist")

    video_folder_path = cohort_folder_path / session_id

    if not video_folder_path.exists():
        raise FileNotFoundError(f"Folder {cohort_folder_path} does not exist")
    video_file_paths = natsort.natsorted(video_folder_path.glob(f"*{subject_metadata['animal ID']}*.avi"))

    if not len(video_file_paths):
        raise FileNotFoundError(
            f"No video files found in for animal ID {subject_metadata['animal ID']} in '{video_folder_path}'."
        )

    analysis_csv_file_path = None
    if session_id == "2_Reference":
        analysis_csv_file_path = cohort_folder_path / "Reference_analysis.csv"
    elif session_id == "3_Reversal":
        analysis_csv_file_path = cohort_folder_path / "Reversal_analysis.csv"

    # Whether to overwrite the NWB file if it already exists
    overwrite = True

    if analysis_csv_file_path is not None:
        df = pd.read_csv(analysis_csv_file_path)
        filtered_data = df[df["Animal"].str.contains(str(subject_metadata["animal ID"]), na=False)]
        reduced_data = filtered_data.reindex(columns=("Date", "Time"))
        filtered_data_index = range(len(reduced_data))
        filtered_data.index = filtered_data_index
        if filtered_data.empty:
            raise ValueError(f"No rows found in the CSV file for '{subject_metadata['animal ID']}'.")
        for session_date, session_data in filtered_data.groupby("Date"):
            video_file_paths_per_session = [video_file_paths[i] for i in session_data.index]

            datetime_strings = session_data["Date"] + " " + session_data["Time"]
            video_timestamps = pd.to_datetime(datetime_strings, dayfirst=True)
            day_num = int(session_data["Day"].iloc[0])
            session_to_nwb(
                output_dir_path=output_dir_path,
                video_file_paths=video_file_paths_per_session,
                video_timestamps=video_timestamps,
                session_id=f"{task_acronym}_{session_id}",
                subject_metadata=subject_metadata,
                trials=session_data,
                overwrite=overwrite,
            )

    # session_to_nwb(
    #     output_dir_path=output_dir_path,
    #     video_file_paths=video_file_paths,
    #     session_id=f"{task_acronym}_{session_id}",
    #     subject_metadata=subject_metadata,
    #     analysis_csv_file_path=analysis_csv_file_path,
    #     overwrite=overwrite,
    # )
