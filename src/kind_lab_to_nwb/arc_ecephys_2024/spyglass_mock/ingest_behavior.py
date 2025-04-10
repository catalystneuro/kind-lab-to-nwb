"""Ingest mock behavior data from an NWB file into a spyglass database."""

import datajoint as dj
from pathlib import Path
import numpy as np
from pynwb import NWBHDF5IO

dj.conn(use_tls=False)

dj_local_conf_path = "/home/alessandra/CatalystNeuro/kind-lab-to-nwb/dj_local_conf.json"
dj.config.load(dj_local_conf_path)  # load config for database connection info

# spyglass.common has the most frequently used tables
import spyglass.common as sgc  # this import connects to the database

# spyglass.data_import has tools for inserting NWB files into the database
import spyglass.data_import as sgi

from spyglass.spikesorting.spikesorting_merge import (
    SpikeSortingOutput,
)  # This import is necessary for the spike sorting to be loaded properly
import spyglass.spikesorting.v1 as sgs
from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename


def test_behavior(nwbfile_path: Path):
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    time_series = (sgc.DIOEvents & {"nwb_file_name": nwb_copy_file_name}).fetch_nwb()[0]["dio"]
    spyglass_dio_data = np.asarray(time_series.data)
    with NWBHDF5IO(nwbfile_path, "r") as io:
        nwbfile = io.read()
        nwb_dio_data = np.asarray(
            nwbfile.processing["behavior"]
            .data_interfaces["behavioral_events"]
            .time_series["behavioral_events_series"]
            .data
        )
        np.testing.assert_array_equal(spyglass_dio_data, nwb_dio_data)
    print("Test passed: DIO data matches between Spyglass and NWB file.")


def main():
    nwbfile_path = Path("/media/alessandra/HD2/kind_lab_conversion_nwb/Spyglass/raw/mock_behavior.nwb")
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

    test_behavior(nwbfile_path)


if __name__ == "__main__":
    main()
    print("Done!")
