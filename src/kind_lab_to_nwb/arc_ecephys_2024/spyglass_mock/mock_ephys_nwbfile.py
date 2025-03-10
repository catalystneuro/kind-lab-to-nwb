"""Create a mock NWB file with spyglass-compatible ephys data for testing purposes."""

from pynwb.testing.mock.file import mock_NWBFile
from pynwb.testing.mock.ecephys import mock_ElectricalSeries
from ndx_franklab_novela import DataAcqDevice, CameraDevice, Probe, Shank, ShanksElectrode, NwbElectrodeGroup
from pynwb import NWBHDF5IO
import numpy as np
from pathlib import Path


def main():
    nwbfile = mock_NWBFile(
        identifier="my_identifier", session_description="my_session_description", session_id="my_session_id"
    )

    data_acq_device = DataAcqDevice(
        name="my_data_acq", system="my_system", amplifier="my_amplifier", adc_circuit="my_adc_circuit"
    )
    camera_device = CameraDevice(
        name="Camera 1",
        meters_per_pixel=1.0,
        model="my_model",
        lens="my_lens",
        camera_name="my_camera_name",
    )
    nwbfile.add_device(data_acq_device)
    nwbfile.add_device(camera_device)

    electrode = ShanksElectrode(name="1", rel_x=0.0, rel_y=0.0, rel_z=0.0)
    shanks_electrodes = [electrode]
    shank = Shank(name="1", shanks_electrodes=shanks_electrodes)
    probe = Probe(
        name="my_probe",
        id=0,
        probe_type="my_probe_type",
        units="my_units",
        probe_description="my_probe_description",
        contact_side_numbering=False,
        contact_size=1.0,
        shanks=[shank],
    )
    nwbfile.add_device(probe)

    # add to electrical series
    electrode_group = NwbElectrodeGroup(
        name="my_electrode_group",
        description="my_description",
        location="my_location",
        device=probe,
        targeted_location="my_targeted_location",
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
        location="my_location",
        group=electrode_group,
        probe_shank=1,
        probe_electrode=1,
        bad_channel=False,
        ref_elect_id=0,
        x=0.0,
        y=0.0,
        z=0.0,
    )
    electrodes = nwbfile.electrodes.create_region(name="electrodes", region=[0], description="electrodes")
    mock_ElectricalSeries(electrodes=electrodes, nwbfile=nwbfile, data=np.ones((10, 1)))

    # add processing module to make spyglass happy
    nwbfile.create_processing_module(name="behavior", description="dummy behavior module")

    nwbfile_path = Path("/media/alessandra/HD2/kind_lab_conversion_nwb/Spyglass/raw/mock_ephys.nwb")
    if nwbfile_path.exists():
        nwbfile_path.unlink()
    with NWBHDF5IO(nwbfile_path, "w") as io:
        io.write(nwbfile)
    print(f"mock ephys NWB file successfully written at {nwbfile_path}")


if __name__ == "__main__":
    main()
