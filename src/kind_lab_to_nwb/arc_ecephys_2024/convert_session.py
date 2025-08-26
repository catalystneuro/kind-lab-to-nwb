"""Primary script to run to convert an entire session for of data using the NWBConverter."""

import datetime
from pathlib import Path
from typing import Union
from zoneinfo import ZoneInfo
import numpy as np

from pynwb import NWBHDF5IO

from neuroconv.utils import (
    dict_deep_update,
    load_dict_from_file,
)
from neuroconv.tools.nwb_helpers import (
    get_default_nwbfile_metadata,
    make_nwbfile_from_metadata,
)
from neuroconv.tools.path_expansion import LocalPathExpander

from kind_lab_to_nwb.arc_ecephys_2024.utils import (
    add_behavioral_video,
    get_channels_info_from_subject_id,
    add_electrical_series,
    add_behavioral_events,
    add_behavioral_signals,
    compute_time_offset,
    get_first_CS_time,
    get_first_CS_video_frame,
)
import pandas as pd


def session_to_nwb(
    data_dir_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    path_expander_metadata: dict,
    stub_test: bool = False,
    verbose: bool = False,
):

    data_dir_path = Path(data_dir_path)
    output_dir_path = Path(output_dir_path)
    if stub_test:
        output_dir_path = output_dir_path / "nwb_stub"
    output_dir_path.mkdir(
        parents=True,
        exist_ok=True,
    )
    subject_id = path_expander_metadata["metadata"]["Subject"]["subject_id"]
    session_id = path_expander_metadata["metadata"]["NWBFile"]["session_id"]
    nwbfile_path = output_dir_path / f"sub-{subject_id}_ses-{session_id}.nwb"

    # Get default metadata
    metadata = get_default_nwbfile_metadata()

    # Add datetime to conversion
    session_date = path_expander_metadata["metadata"]["extras"]["session_date"]
    session_time = path_expander_metadata["metadata"]["extras"]["session_time"]
    session_start_time = datetime.datetime.strptime(f"{session_date} {session_time}", "%Y-%m-%d %H-%M-%S")
    metadata["NWBFile"]["session_start_time"] = session_start_time.replace(tzinfo=ZoneInfo("Europe/London"))

    # Update default metadata with the editable in the corresponding yaml file
    editable_metadata_path = Path(__file__).parent / "metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    metadata = dict_deep_update(
        metadata,
        editable_metadata,
    )
    metadata["NWBFile"]["session_description"] = editable_metadata["Tasks"][session_id]["session_description"]

    # Update default metadata with the metadata extracted from the path expander
    metadata = dict_deep_update(
        metadata,
        path_expander_metadata["metadata"],
    )

    # Update subject genotype
    file_path = data_dir_path / "channels_details_v2.xlsx"
    df = pd.read_excel(file_path)
    subject_genotype = df[df["Folder"] == subject_id]["Genotype"].iloc[0]
    if subject_genotype == "wt":
        subject_genotype = "WT"
    elif subject_genotype == "het":
        subject_genotype = "Syngap1+/Delta-GAP"
    else:
        raise ValueError(f"Genotype {subject_genotype} not recognized")

    metadata["Subject"]["genotype"] = subject_genotype

    nwbfile = make_nwbfile_from_metadata(metadata=metadata)

    TDT_folder_path = path_expander_metadata["source_data"]["OpenEphysRecording"]["folder_path"]

    video_extensions = ["avi", "mp4", "mkv"]
    video_file_path = None
    for ext in video_extensions:
        video_files = list(data_dir_path.glob(f"{subject_id}/{session_id}/*.{ext}"))
        if video_files:
            video_file_path = video_files[0]
            break
        else:
            print(f"Warning: No video file found for subject {subject_id}, session {session_id} with extension {ext}")

    # Time alignment
    if session_id == "Recall" and video_file_path:
        CS_video_frames_file = data_dir_path / "cs_video_frames.xlsx"
        first_CS_video_frame = get_first_CS_video_frame(file_path=CS_video_frames_file, subject_id=subject_id)
        if first_CS_video_frame is None:
            time_offset = 0.0
            video_time_offset = np.nan
        else:
            first_CS_time = get_first_CS_time(folder_path=TDT_folder_path)
            time_offset = compute_time_offset(
                video_file_path=video_file_path, first_CS_time=first_CS_time, first_CS_video_frame=first_CS_video_frame
            )
            video_time_offset = time_offset if time_offset > 0 else 0.0
    else:
        time_offset = 0.0
        video_time_offset = np.nan

    # Add behavioral video
    if video_file_path:
        task_metadata = editable_metadata["Tasks"][session_id]
        add_behavioral_video(
            nwbfile=nwbfile,
            metadata=metadata,
            video_file_path=video_file_path,
            task_metadata=task_metadata,
            time_offset=video_time_offset,
        )

    # Add EEG data
    excel_file_path = data_dir_path / "channels_details_v2.xlsx"
    channels_info, probe_id = get_channels_info_from_subject_id(
        subject_id=subject_id, excel_file_path=excel_file_path, number_of_channels=16
    )

    add_electrical_series(
        nwbfile=nwbfile,
        metadata=metadata,
        channels_info=channels_info,
        probe_id=probe_id,
        folder_path=TDT_folder_path,
        stream_name="Signals CH",
        time_offset=-time_offset if time_offset < 0 else 0.0,
    )

    # Add accelerometer data
    add_behavioral_signals(
        nwbfile=nwbfile,
        metadata=metadata,
        folder_path=TDT_folder_path,
        stream_name="Signals AUX",
        time_offset=-time_offset if time_offset < 0 else 0.0,
    )

    # Add behavior events
    add_behavioral_events(
        nwbfile=nwbfile, folder_path=TDT_folder_path, starting_time=-time_offset if time_offset < 0 else 0.0
    )

    if verbose:
        print(f"Write NWB file {nwbfile_path.name}")
    with NWBHDF5IO(nwbfile_path, mode="w") as io:
        io.write(nwbfile)


if __name__ == "__main__":
    # Parameters for conversion
    # data_dir_path = Path("/media/alessandra/HD2/Kind-CN-data-share/neuronal_circuits/fear_conditionning_paradigm")
    data_dir_path = Path("D:/Kind-CN-data-share/neuronal_circuits/fear_conditionning_paradigm")
    # output_dir_path = Path("/media/alessandra/HD2/kind_lab_conversion_nwb")
    output_dir_path = Path("D:/kind_lab_conversion_nwb")

    source_data_spec = {
        "OpenEphysRecording": {
            "base_directory": data_dir_path,
            "folder_path": "{subject_id}/{session_id}/{subject_id}_{session_date}_{session_time}_{task}",
        },
    }

    # Instantiate LocalPathExpander
    path_expander = LocalPathExpander()

    # Expand paths and extract metadata
    metadata_list = path_expander.expand_paths(source_data_spec)

    stub_test = False
    for id in [31, 36, 40, 44, 48, 52, 56, 60, 64, 68, 73, 78, 83, 88, 93, 98]:
        session_to_nwb(
            data_dir_path=data_dir_path,
            output_dir_path=output_dir_path,
            path_expander_metadata=metadata_list[id],
            stub_test=stub_test,
        )
