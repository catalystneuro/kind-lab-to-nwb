"""
Utilities for adding behavioral data to NWB files.
This module provides functions to add behavioral events and signals
from Open Ephys recordings to NWB files.
Functions:
    add_behavioral_events: Add behavioral events from Open Ephys recordings to an NWB file.
    add_behavioral_signals: Add behavioral signals (e.g., accelerometer data) from Open Ephys
        recordings to an NWB file.
"""

from pydantic import DirectoryPath
from typing import Optional

from pynwb import NWBFile, TimeSeries
from pynwb.behavior import BehavioralEvents, BehavioralTimeSeries
from spikeinterface.extractors.neoextractors import OpenEphysLegacyRecordingExtractor

from ndx_franklab_novela import DataAcqDevice

from open_ephys.analysis import Session
import numpy as np


def add_behavioral_events(nwbfile: NWBFile, folder_path) -> None:
    session = Session(folder_path)
    recording = session.recordings[0]
    event_table = recording.events

    # Constants and setup
    rate = 2000.0  # Hz
    starting_time = 0.0  # seconds

    # Create behavioral events container
    behavioral_events = BehavioralEvents(name="behavioral_events")

    # Process each channel
    for channel, events in event_table.groupby("channel"):
        events = events.sort_values("timestamp")
        max_timestamp = int(events["timestamp"].max() + 1)
        unique_states = set(events["state"].unique())
        if all(state == 0 for state in unique_states):
            ValueError(f"Channel {channel} has no events")

        signal = np.zeros(max_timestamp)
        for _, row in events.iterrows():
            timestamp = int(row["timestamp"])
            signal[timestamp:] = row["state"]

        behavioral_events.add_timeseries(
            TimeSeries(
                name=f"ttl_channel_{channel}",  # TODO add a more specific name and description for the TTL signal
                data=signal,
                rate=rate,
                starting_time=starting_time,
                unit="n.a.",
                description=f"TTL signal from channel {channel}",
            )
        )

    # Check if behavior module already exists in the NWBFile
    behavior_module = None
    if "behavior" in nwbfile.processing:
        behavior_module = nwbfile.processing["behavior"]
    else:
        behavior_module = nwbfile.create_processing_module(name="behavior", description="behavior module")
    # Add behavioral events to the behavior module
    behavior_module.add(behavioral_events)


def add_behavioral_signals(
    nwbfile: NWBFile,
    metadata,
    folder_path: DirectoryPath,
    stream_name: Optional[str] = "Signals AUX",
    block_index: Optional[int] = None,
) -> None:
    # Check if DataAcqDevice already exists in the NWBFile
    device_name = metadata["Devices"]["DataAcqDevice"].get("name")
    if device_name in nwbfile.devices:
        data_acq_device = nwbfile.devices[device_name]
    else:
        data_acq_device = DataAcqDevice(**metadata["Devices"]["DataAcqDevice"])
        nwbfile.add_device(data_acq_device)

    extractor = OpenEphysLegacyRecordingExtractor(
        folder_path=folder_path,
        stream_name=stream_name,
        block_index=block_index,
    )

    time_series = extractor.get_traces()
    time_info = extractor.get_time_info()
    rate = time_info["sampling_frequency"]
    starting_time = time_info["t_start"]

    # Create behavioral events container
    analog_timeseries = BehavioralEvents(name="analog")
    analog_timeseries.create_timeseries(
        name="analog",
        description="AccelerometerXComponent AccelerometerYComponent AccelerometerZComponent",
        unit="volts",
        data=time_series,
        starting_time=starting_time,
        rate=rate,
    )
    # Check if behavior module already exists in the NWBFile
    behavior_module = None
    if "behavior" in nwbfile.processing:
        behavior_module = nwbfile.processing["behavior"]
    else:
        behavior_module = nwbfile.create_processing_module(name="behavior", description="behavior module")

    behavior_module.add(analog_timeseries)
