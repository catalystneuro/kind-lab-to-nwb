"""
Utility functions to add EEG data from OpenEphys recordings to NWB files.

This module handles extraction of EEG data from OpenEphys files, including channel information
management and electrical series creation in the NWB format.
"""

from pydantic import DirectoryPath, FilePath
from typing import Optional
import numpy as np
import pandas as pd

from pynwb import NWBFile
from pynwb.ecephys import ElectricalSeries, LFP
from neuroconv.tools.spikeinterface.spikeinterface import _report_variable_offset

from spikeinterface.extractors.neoextractors import OpenEphysLegacyRecordingExtractor

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
            location = "unknown" if bad_channel else value

            channels_info[channel_id] = {"location": location, "bad_channel": bad_channel}
        else:
            raise ValueError(f"Channel {channel_id} not found in the Excel file")
    return channels_info


def add_electrical_series(
    nwbfile: NWBFile,
    metadata,
    channels_info: dict,
    folder_path: DirectoryPath,
    stream_name: Optional[str] = "Signals CH",
    block_index: Optional[int] = None,
) -> None:

    data_acq_device = DataAcqDevice(**metadata["Devices"]["DataAcqDevice"])
    nwbfile.add_device(data_acq_device)

    extractor = OpenEphysLegacyRecordingExtractor(
        folder_path=folder_path,
        stream_name=stream_name,
        block_index=block_index,
    )
    channel_names = extractor.get_property("channel_names")

    shanks_electrodes = []
    for ch, info in channels_info.items():
        electrode = ShanksElectrode(name=str(ch), rel_x=0.0, rel_y=0.0, rel_z=0.0)
        shanks_electrodes.append(electrode)

    shank = Shank(**metadata["Ecephys"]["Shank"], shanks_electrodes=shanks_electrodes)
    probe = Probe(**metadata["Ecephys"]["Probe"], shanks=[shank])
    nwbfile.add_device(probe)

    # add to electrical series
    electrode_group = NwbElectrodeGroup(**metadata["Ecephys"]["NwbElectrodeGroup"], device=probe)
    nwbfile.add_electrode_group(electrode_group)

    extra_cols = [
        "channel_name",
        "probe_shank",
        "probe_electrode",
        "bad_channel",
        "ref_elect_id",
    ]
    for col in extra_cols:
        nwbfile.add_electrode_column(name=col, description=f"description for {col}")

    for ch, info in channels_info.items():
        nwbfile.add_electrode(
            location=info["location"],  # convert to standard naming
            group=electrode_group,
            channel_name=channel_names[ch],
            probe_shank=0,
            probe_electrode=ch,
            bad_channel=info["bad_channel"],
            ref_elect_id=ch,
            # x=0.0,
            # y=0.0,
            # z=0.0,
        )

    electrodes = nwbfile.electrodes.create_region(
        name="electrodes",
        region=list(range(len(channels_info.items()))),
        description="electrodes table region",
    )

    electrical_series = extractor.get_traces()
    time_info = extractor.get_time_info()
    rate = time_info["sampling_frequency"]
    starting_time = time_info["t_start"]

    channel_conversion = extractor.get_channel_gains()
    channel_offsets = extractor.get_channel_offsets()

    unique_channel_conversion = np.unique(channel_conversion)
    unique_channel_conversion = unique_channel_conversion[0] if len(unique_channel_conversion) == 1 else None

    unique_offset = np.unique(channel_offsets)
    if unique_offset.size > 1:
        channel_ids = extractor.get_channel_ids()
        # This prints a user friendly error where the user is provided with a map from offset to channels
        _report_variable_offset(channel_offsets, channel_ids)
    unique_offset = unique_offset[0] if unique_offset[0] is not None else 0

    micro_to_volts_conversion_factor = 1e-6
    if unique_channel_conversion is None:
        conversion = micro_to_volts_conversion_factor
        channel_conversion = channel_conversion
    elif unique_channel_conversion is not None:
        conversion = unique_channel_conversion * micro_to_volts_conversion_factor
        channel_conversion = None

    if extractor.is_filtered():
        electrical_series_name = "lfp_series"
    else:
        electrical_series_name = "eeg_series"

    electrical_series = ElectricalSeries(
        name=electrical_series_name,
        data=electrical_series,
        electrodes=electrodes,
        rate=rate,
        starting_time=starting_time,
        channel_conversion=channel_conversion,
        conversion=conversion,
    )
    if extractor.is_filtered():
        lfp = LFP(electrical_series=electrical_series)
        ecephys_module = nwbfile.create_processing_module(name="ecephys", description="ecephys module")
        ecephys_module.add(lfp)
    else:
        nwbfile.add_acquisition(electrical_series)
