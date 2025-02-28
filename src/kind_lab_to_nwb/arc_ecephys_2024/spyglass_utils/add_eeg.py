"""
Create a mock NWB file with spyglass-compatible behavior data for testing purposes.

This file also contains ephys data, since it is necessary for the behavior data to be compatible with spyglass.
"""

from spikeinterface.extractors.neoextractors import OpenEphysLegacyRecordingExtractor

from pydantic import DirectoryPath, FilePath
from typing import Optional
import numpy as np
import pandas as pd

from pynwb import NWBFile
from pynwb.ecephys import ElectricalSeries

from ndx_franklab_novela import DataAcqDevice, Probe, Shank, ShanksElectrode, NwbElectrodeGroup


def get_channels_info_from_subject_id(subject_id: str, excel_file_path: FilePath, number_of_channels: int = 16) -> dict:
    """
    Get channels information from an excel file based on the subject ID.

    Parameters
    ----------
    subject_id : str
        The subject ID corresponding to the "Folder" column in the excel file.
    excel_file_path : FilePath
        The path to the excel file containing channel information.

    Returns
    -------
    dict
        A dict containing channels information for the specified subject.
        Includes channel locations and bad channel indicators.
    """
    # Read the Excel file containing channel information
    df = pd.read_excel(excel_file_path)

    # Filter rows for the specific subject_id
    subject_df = df[df["Folder"] == subject_id]

    if subject_df.empty:
        raise ValueError(f"Subject ID '{subject_id}' not found in the Excel file")

    # Extract channel information
    channels_info = {}
    for channel_id in range(number_of_channels):
        if channel_id in subject_df.columns:
            value = subject_df[channel_id].iloc[0]
            bad_channel = value == "bad"
            location = "Bad channel with no location" if bad_channel else value

            channels_info[channel_id] = {"location": location, "bad_channel": bad_channel}
        else:
            raise ValueError(f"Channel {channel_id} not found in the Excel file")
    return channels_info


def add_eeg(
    nwbfile: NWBFile,
    metadata,
    channels_info: dict,
    folder_path: DirectoryPath,
    stream_name: Optional[str] = "Signals CH",
    block_index: Optional[int] = None,
) -> None:
    data_acq_device = DataAcqDevice(**metadata["Devices"]["DataAcqDevice"])
    nwbfile.add_device(data_acq_device)

    electrode = ShanksElectrode(**metadata["Ecephys"]["ShanksElectrode"])
    shanks_electrodes = [electrode]
    shank = Shank(**metadata["Ecephys"]["Shank"], shanks_electrodes=shanks_electrodes)
    probe = Probe(**metadata["Ecephys"]["Probe"], shanks=[shank])
    nwbfile.add_device(probe)

    # add to electrical series
    electrode_group = NwbElectrodeGroup(**metadata["Ecephys"]["NwbElectrodeGroup"], device=probe)
    nwbfile.add_electrode_group(electrode_group)

    extra_cols = [
        "probe_shank",
        "probe_electrode",
        "bad_channel",
        "ref_elect_id",
    ]
    for col in extra_cols:
        nwbfile.add_electrode_column(name=col, description=f"description for {col}")

    extractor = OpenEphysLegacyRecordingExtractor(
        folder_path=folder_path,
        stream_name=stream_name,
        block_index=block_index,
    )

    # channels_xyz = extractor.get_channel_locations(axes="xyz")

    for ch, info in channels_info.items():
        nwbfile.add_electrode(
            location=info["location"],  # convert to standard naming
            group=electrode_group,
            probe_shank=1,
            probe_electrode=1,
            bad_channel=info["bad_channel"],
            ref_elect_id=ch,
            x=0.0,  # channels_xyz[ch, 0],
            y=0.0,  # channels_xyz[ch, 1],
            z=0.0,  # channels_xyz[ch, 2],
        )

    electrodes = nwbfile.electrodes.create_region(
        name="electrodes", region=list(range(len(channels_info.items()))), description="electrodes"
    )

    electrical_series = extractor.get_traces()
    time_info = extractor.get_time_info()
    rate = time_info["sampling_frequency"]
    starting_time = time_info["t_start"]
    # conversion = extractor.get_property("gain_to_uV")
    # offset = extractor.get_property("offset_to_uV")
    eeg = ElectricalSeries(
        name="eeg_series", data=electrical_series, electrodes=electrodes, rate=rate, starting_time=starting_time
    )
    nwbfile.add_acquisition(eeg)
