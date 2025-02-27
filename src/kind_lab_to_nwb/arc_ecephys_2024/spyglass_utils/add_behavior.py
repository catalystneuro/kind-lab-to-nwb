"""
Create a mock NWB file with spyglass-compatible behavior data for testing purposes.

This file also contains ephys data, since it is necessary for the behavior data to be compatible with spyglass.
"""

from typing import Optional
import numpy as np

from pynwb import NWBFile
from pynwb.behavior import BehavioralEvents

from ndx_franklab_novela import DataAcqDevice, Probe, Shank, ShanksElectrode, NwbElectrodeGroup


def add_behavioral_events(nwbfile: NWBFile, metadata) -> None:
    # TODO extract time_series from OpenEphysEventExtractor
    behavioral_events = BehavioralEvents(name="behavioral_events", time_series=time_series)
    behavior_module = nwbfile.create_processing_module(name="behavior", description="behavior module")
    behavior_module.add(behavioral_events)


def add_behavioral_signals(nwbfile: NWBFile, metadata) -> None:
    # TODO extract time_series from OpenEphysRecordingInterface
    behavioral_events = BehavioralEvents(name="behavioral_signals", time_series=time_series)
    behavior_module = nwbfile.create_processing_module(name="behavior", description="behavior module")
    behavior_module.add(behavioral_events)
