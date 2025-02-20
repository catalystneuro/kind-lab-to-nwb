"""Primary script to run to convert an entire session for of data using the NWBConverter."""

import datetime
from pathlib import Path
from typing import Union
from zoneinfo import ZoneInfo

from neuroconv.utils import (
    dict_deep_update,
    load_dict_from_file,
)
from neuroconv.tools.path_expansion import LocalPathExpander

from nwbconverter import ArcEcephys2024NWBConverter
import pandas as pd


def session_to_nwb(
    data_dir_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    path_expander_metadata: dict,
    stub_test: bool = False,
    overwrite: bool = False,
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
    nwbfile_path = output_dir_path / f"sub_{subject_id}-ses{session_id}.nwb"

    source_data = dict()
    conversion_options = dict()

    # Add Recording
    recordings_folder_path = path_expander_metadata["source_data"]["OpenEphysRecording"]["folder_path"]
    source_data.update(dict(OpenEphysRecording=dict(folder_path=recordings_folder_path, stream_name="Signals CH")))
    conversion_options.update(dict(OpenEphysRecording=dict(stub_test=stub_test)))
    # TODO :  Add EEG and LFP with spyglass compatibility

    # TODO
    # Add openephys events
    # Add accelometer data

    # Add Video
    video_file_path = next(data_dir_path.glob(f"{subject_id}/{session_id}/*.avi"))
    source_data.update(dict(Video=dict(file_paths=[video_file_path])))
    conversion_options.update(dict(Video=dict(stub_test=stub_test)))

    # Instantiate converter
    converter = ArcEcephys2024NWBConverter(source_data=source_data)

    # Add datetime to conversion
    metadata = converter.get_metadata()
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
    metadata["NWBFile"]["session_description"] = editable_metadata["SessionType"][session_id]["session_description"]

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
    if subject_genotype == "het":
        subject_genotype = "Syngap+/∆-GAP"
    else:
        raise ValueError(f"Genotype {subject_genotype} not recognized")

    metadata["Subject"]["genotype"] = subject_genotype

    # Run conversion
    converter.run_conversion(
        metadata=metadata,
        nwbfile_path=nwbfile_path,
        conversion_options=conversion_options,
        overwrite=overwrite,
    )


if __name__ == "__main__":

    # Parameters for conversion
    data_dir_path = Path("D:/Kind-CN-data-share/neuronal_circuits/fear_conditionning_paradigm")
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

    stub_test = True
    overwrite = True

    session_to_nwb(
        data_dir_path=data_dir_path,
        output_dir_path=output_dir_path,
        path_expander_metadata=metadata_list[140],
        stub_test=stub_test,
        overwrite=overwrite,
    )
