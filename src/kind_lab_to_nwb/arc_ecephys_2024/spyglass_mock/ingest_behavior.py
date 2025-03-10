"""Ingest mock behavior data from an NWB file into a spyglass database."""

import numpy as np
from pynwb import NWBHDF5IO
import datajoint as dj
from pathlib import Path

dj_local_conf_path = "/Users/pauladkisson/Documents/CatalystNeuro/JadhavConv/jadhav-lab-to-nwb/src/jadhav_lab_to_nwb/spyglass_mock/dj_local_conf.json"
dj.config.load(dj_local_conf_path)  # load config for database connection info

# spyglass.common has the most frequently used tables
import spyglass.common as sgc  # this import connects to the database

# spyglass.data_import has tools for inserting NWB files into the database
import spyglass.data_import as sgi
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename


def main():
    nwbfile_path = Path("/Volumes/T7/CatalystNeuro/Spyglass/raw/mock_behavior.nwb")
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)

    if sgc.Session & {"nwb_file_name": nwb_copy_file_name}:
        (sgc.Session & {"nwb_file_name": nwb_copy_file_name}).delete()
    if sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}:
        (sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}).delete()
    if sgc.DIOEvents & {"nwb_file_name": nwb_copy_file_name}:
        (sgc.DIOEvents & {"nwb_file_name": nwb_copy_file_name}).delete()

    sgi.insert_sessions(str(nwbfile_path), rollback_on_fail=True, raise_err=True)
    print("=== Session ===")
    print(sgc.Session & {"nwb_file_name": nwb_copy_file_name})
    print("=== NWB File ===")
    print(sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name})
    print("=== DIOEvents ===")
    print(sgc.DIOEvents & {"nwb_file_name": nwb_copy_file_name})

    time_series = (sgc.DIOEvents & {"nwb_file_name": nwb_copy_file_name}).fetch_nwb()[0]["dio"]
    spyglass_dio_data = np.asarray(time_series.data)
    with NWBHDF5IO(nwbfile_path, "r") as io:
        nwbfile = io.read()
        nwb_dio_data = np.asarray(
            nwbfile.processing["behavior"].data_interfaces["behavioral_events"].time_series["my_time_series"].data
        )
        np.testing.assert_array_equal(spyglass_dio_data, nwb_dio_data)


if __name__ == "__main__":
    main()
    print("Done!")
