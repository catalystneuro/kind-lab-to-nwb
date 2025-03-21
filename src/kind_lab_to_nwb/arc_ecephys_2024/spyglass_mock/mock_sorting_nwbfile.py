"""Create a mock NWB file with spyglass-compatible ephys data for testing purposes."""

from pynwb.testing.mock.file import mock_NWBFile
from pynwb.testing.mock.ecephys import mock_Units
from ndx_franklab_novela import DataAcqDevice, CameraDevice, Probe, Shank, ShanksElectrode, NwbElectrodeGroup
from pynwb import NWBHDF5IO
import numpy as np
from pathlib import Path


def main():
    nwbfile = mock_NWBFile(identifier="my_identifier", session_description="my_session_description")
    columns = ["custom_label", "custom_quantification"]
    for column in columns:
        nwbfile.add_unit_column(name=column, description=f"description for {column}")
    num_units = 10
    for unit in range(num_units):
        spike_times = sorted(np.random.rand(unit + 1))
        label = "even unit" if unit % 2 == 0 else "odd unit"
        quantification = np.random.rand()
        nwbfile.add_unit(spike_times=spike_times, custom_label=label, custom_quantification=quantification)

    # add processing module to make spyglass happy
    nwbfile.create_processing_module(name="behavior", description="dummy behavior module")

    nwbfile_path = Path("/media/alessandra/HD2/kind_lab_conversion_nwb/Spyglass/raw/mock_sorting.nwb")
    if nwbfile_path.exists():
        nwbfile_path.unlink()
    with NWBHDF5IO(nwbfile_path, "w") as io:
        io.write(nwbfile)
    print(f"mock sorting NWB file successfully written at {nwbfile_path}")


if __name__ == "__main__":
    main()
