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


def get_channels_info_from_subject_id(
    subject_id: str, excel_file_path: FilePath, number_of_channels: int = 16
) -> tuple:
    """
    Get channels information and probe ID from an excel file based on the subject ID.

    Parameters
    ----------
    subject_id : str
        The subject ID corresponding to the "Folder" column in the excel file.
    excel_file_path : FilePath
        The path to the excel file containing channel information.
    number_of_channels : int, optional
        The number of channels to extract information for. Default is 16.

    Returns
    -------
    tuple
        A tuple containing:
        - dict: Channel information with channel IDs as keys and dicts with 'location' and 'bad_channel' as values
        - probe_id: The index of the subject in the dataframe, used as probe_id
    """
    # Read the Excel file containing channel information
    df = pd.read_excel(excel_file_path)

    # Filter row for the specific subject_id
    subject_df = df[df["Folder"] == subject_id]

    # Get row index for the subject_id to use it as probe_id
    probe_id = subject_df.index[0]

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
    return channels_info, probe_id


def add_electrical_series(
    nwbfile: NWBFile,
    metadata,
    channels_info: dict,
    probe_id: int,
    folder_path: DirectoryPath,
    stream_name: Optional[str] = "Signals CH",
    block_index: Optional[int] = None,
) -> None:

    data_acq_device = DataAcqDevice(**metadata["Devices"]["DataAcqDevice"])
    nwbfile.add_device(data_acq_device)

    extra_cols = [
        "channel_name",
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
    channel_names = extractor.get_property("channel_names")

    for ch, info in channels_info.items():
        if "EEG" in info["location"]:
            probe_shank = 0
        else:
            probe_shank = 1

    shanks = []
    for ch, info in channels_info.items():
        location = info["location"]
        if "EEG" in location:
            eeg_electrode = ShanksElectrode(name=location, rel_x=0.0, rel_y=0.0, rel_z=0.0)
            shank = Shank(**metadata["Ecephys"][location], shanks_electrodes=eeg_electrode)
            shanks.append(shank)
        else:
            lfp_electrode_left = ShanksElectrode(name=location, rel_x=0.0, rel_y=0.0, rel_z=0.0)
            lfp_electrode_right = ShanksElectrode(name=location, rel_x=0.0, rel_y=0.0, rel_z=0.0)
            shank = Shank(
                **metadata["Ecephys"][location.split("_")[0]],
                shanks_electrodes=[lfp_electrode_left, lfp_electrode_right],
            )
            shanks.append(shank)
        nwbfile.add_electrode(
            location=info["location"],  # convert to standard naming
            group=electrode_group,
            channel_name=channel_names[ch],
            probe_shank=probe_shank,
            probe_electrode=ch,
            bad_channel=info["bad_channel"],
            ref_elect_id=ch,
            # x=0.0,
            # y=0.0,
            # z=0.0,
        )

    probe = Probe(**metadata["Ecephys"]["Probe"], probe_id=probe_id, probe_description=str(probe_id), shanks=shanks)
    nwbfile.add_device(probe)

    # Add ElectrodeGroup
    electrode_group = NwbElectrodeGroup(**metadata["Ecephys"]["NwbElectrodeGroup"], device=probe)
    nwbfile.add_electrode_group(electrode_group)

    # Add to electrical series
    traces = extractor.get_traces()
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

    lfp_channel_ids = [ch for ch, info in channels_info.items() if "EEG" not in info["location"]]
    if len(lfp_channel_ids) > 0:
        lfp_electrodes = nwbfile.electrodes.create_region(
            name="electrodes",
            region=lfp_channel_ids,
            description="lfp electrodes table region",
        )
        lfp_traces = traces[:, lfp_channel_ids]
        lfp_electrical_series = ElectricalSeries(
            name="lfp_series",
            data=lfp_traces,
            electrodes=lfp_electrodes,
            rate=rate,
            starting_time=starting_time,
            channel_conversion=channel_conversion[lfp_channel_ids] if channel_conversion is not None else None,
            conversion=conversion,
        )
        lfp = LFP(electrical_series=lfp_electrical_series)
        ecephys_module = nwbfile.create_processing_module(name="ecephys", description="ecephys module")
        ecephys_module.add(lfp)

    eeg_channel_ids = [ch for ch, info in channels_info.items() if "EEG" in info["location"]]
    if len(eeg_channel_ids) > 0:
        eeg_electrodes = nwbfile.electrodes.create_region(
            name="electrodes",
            region=eeg_channel_ids,
            description="eeg electrodes table region",
        )
        eeg_traces = traces[:, eeg_channel_ids]
        eeg_electrical_series = ElectricalSeries(
            name="eeg_series",
            data=eeg_traces,
            electrodes=eeg_electrodes,
            rate=rate,
            starting_time=starting_time,
            channel_conversion=channel_conversion[eeg_channel_ids] if channel_conversion is not None else None,
            conversion=conversion,
        )
        nwbfile.add_acquisition(eeg_electrical_series)
