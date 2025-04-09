"""
Create a mock NWB file with spyglass-compatible behavior data for testing purposes.

This file also contains ephys data, since it is necessary for the behavior data to be compatible with spyglass.
"""

from pynwb.testing.mock.file import mock_NWBFile
from pynwb.testing.mock.behavior import mock_TimeSeries
from pynwb.behavior import BehavioralEvents
from pynwb import NWBHDF5IO
import numpy as np
from pathlib import Path

"""
Create a mock NWB file with spyglass-compatible behavior data for testing purposes.

This file also contains ephys data, since it is necessary for the behavior data to be compatible with spyglass.
"""

from pynwb.testing.mock.file import mock_NWBFile
from pynwb.testing.mock.behavior import mock_TimeSeries
from pynwb.behavior import BehavioralEvents
from pynwb import NWBHDF5IO
import numpy as np
from pathlib import Path
from kind_lab_to_nwb.arc_ecephys_2024.spyglass_mock import add_ephys
from mock_ephys_nwbfile import add_ephys


def add_behavioral_events(nwbfile):
    time_series = mock_TimeSeries(name="behavioral_events_series", timestamps=np.arange(20), data=np.ones((20, 1)))
    behavioral_events = BehavioralEvents(name="behavioral_events", time_series=time_series)
    behavior_module = nwbfile.create_processing_module(name="behavior", description="behavior module")
    behavior_module.add(behavioral_events)


def main():
    nwbfile = mock_NWBFile(identifier="identifier", session_description="session_description")
    add_behavioral_events(nwbfile)

    # add ephys data to make spyglass happy
    add_ephys(nwbfile)

    nwbfile_path = Path("/media/alessandra/HD2/kind_lab_conversion_nwb/Spyglass/raw/mock_behavior.nwb")
    if nwbfile_path.exists():
        nwbfile_path.unlink()
    with NWBHDF5IO(nwbfile_path, "w") as io:
        io.write(nwbfile)
    print(f"mock behavior NWB file successfully written at {nwbfile_path}")


if __name__ == "__main__":
    main()
