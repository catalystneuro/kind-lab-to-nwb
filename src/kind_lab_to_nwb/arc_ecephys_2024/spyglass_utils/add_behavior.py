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

from pynwb import NWBFile
from pynwb.behavior import BehavioralEvents, BehavioralTimeSeries
from spikeinterface.extractors.neoextractors import OpenEphysLegacyRecordingExtractor

from ndx_franklab_novela import DataAcqDevice

from open_ephys.analysis import Session


def add_behavioral_events(nwbfile: NWBFile, folder_path) -> None:
    session = Session(folder_path)
    recording = session.recordings[0]
    event_table = recording.events
    print(event_table)
    # behavioral_events = BehavioralEvents(name="behavioral_events", time_series=time_series)
    # behavior_module = nwbfile.create_processing_module(name="behavior", description="behavior module")
    # behavior_module.add(behavioral_events)


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

    behavioral_timeseries = BehavioralTimeSeries(name="behavioral_timeseries")
    behavioral_timeseries.create_timeseries(
        name="accelerometer_signal",
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

    behavior_module.add(behavioral_timeseries)
