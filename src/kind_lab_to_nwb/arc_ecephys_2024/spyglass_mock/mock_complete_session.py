"""
Create a mock NWB file with spyglass-compatible behavior data for testing purposes.

This file also contains ephys data, since it is necessary for the behavior data to be compatible with spyglass.
"""

from pynwb.testing.mock.file import mock_NWBFile
from pynwb import NWBHDF5IO
from pathlib import Path
from mock_ephys_nwbfile import add_ephys
from mock_behavior_nwbfile import add_behavioral_events
from mock_video_nwbfile import add_video, add_epoch


def main():
    nwbfile = mock_NWBFile(identifier="identifier", session_description="session_description")
    add_ephys(nwbfile)
    add_behavioral_events(nwbfile)
    add_video(nwbfile)
    add_epoch(nwbfile)

    nwbfile_path = Path("/media/alessandra/HD2/kind_lab_conversion_nwb/Spyglass/raw/mock_complete_session.nwb")
    if nwbfile_path.exists():
        nwbfile_path.unlink()
    with NWBHDF5IO(nwbfile_path, "w") as io:
        io.write(nwbfile)
    print(f"mock behavior NWB file successfully written at {nwbfile_path}")


if __name__ == "__main__":
    main()
