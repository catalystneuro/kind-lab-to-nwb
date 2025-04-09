"""Create a mock NWB file with spyglass-compatible ephys data for testing purposes."""

from pynwb.testing.mock.file import mock_NWBFile
from pynwb.testing.mock.ecephys import mock_ElectricalSeries
from ndx_franklab_novela import DataAcqDevice, Probe, Shank, ShanksElectrode, NwbElectrodeGroup
from pynwb import NWBHDF5IO
import numpy as np
from pathlib import Path


def add_ephys(nwbfile):
    data_acq_device = DataAcqDevice(name="data_acq", system="system", amplifier="amplifier", adc_circuit="adc_circuit")
    nwbfile.add_device(data_acq_device)

    electrode = ShanksElectrode(name="0", rel_x=0.0, rel_y=0.0, rel_z=0.0)
    shanks_electrodes = [electrode]
    shank = Shank(name="0", shanks_electrodes=shanks_electrodes)
    probe = Probe(
        name="probe",
        id=0,
        probe_type="probe_type",
        units="units",
        description="description",
        probe_description="0",
        contact_side_numbering=False,
        contact_size=1.0,
        shanks=[shank],
    )
    nwbfile.add_device(probe)

    # add to electrical series
    electrode_group = NwbElectrodeGroup(
        name="electrode_group",
        description="group_description",
        location="location",
        device=probe,
        targeted_location="targeted_location",
        targeted_x=0.0,
        targeted_y=0.0,
        targeted_z=0.0,
        units="mm",
    )
    nwbfile.add_electrode_group(electrode_group)
    extra_cols = [
        "probe_shank",
        "probe_electrode",
        "bad_channel",
        "ref_elect_id",
    ]
    for col in extra_cols:
        nwbfile.add_electrode_column(name=col, description=f"description for {col}")
    nwbfile.add_electrode(
        location="location",
        group=electrode_group,
        probe_shank=0,
        probe_electrode=0,
        bad_channel=False,
        ref_elect_id=0,
        x=0.0,
        y=0.0,
        z=0.0,
    )
    electrodes = nwbfile.electrodes.create_region(name="electrodes", region=[0], description="electrodes")
    mock_ElectricalSeries(
        name="electrical_series",
        electrodes=electrodes,
        nwbfile=nwbfile,
        timestamps=np.arange(20),
        data=np.ones((20, 1)),
    )


if __name__ == "__main__":
    nwbfile = mock_NWBFile(identifier="identifier", session_description="session_description")
    add_ephys(nwbfile)
    # add processing module to make spyglass happy
    nwbfile.create_processing_module(name="behavior", description="dummy behavior module")

    nwbfile_path = Path("/media/alessandra/HD2/kind_lab_conversion_nwb/Spyglass/raw/mock_ephys.nwb")
    if nwbfile_path.exists():
        nwbfile_path.unlink()
    with NWBHDF5IO(nwbfile_path, "w") as io:
        io.write(nwbfile)
    print(f"mock video NWB file successfully written at {nwbfile_path}")
