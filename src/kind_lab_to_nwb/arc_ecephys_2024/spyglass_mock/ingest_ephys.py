"""Ingest mock ephys data from an NWB file into a spyglass database."""

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


def test_ephys(nwbfile_path: Path):
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    electrical_series = (sgc.Raw & {"nwb_file_name": nwb_copy_file_name}).fetch_nwb()[0]["raw"]
    spyglass_raw_data = np.asarray(electrical_series.data[:10])
    with NWBHDF5IO(nwbfile_path, "r") as io:
        nwbfile = io.read()
        nwb_raw_data = np.asarray(nwbfile.acquisition["electrical_series"].data[:10])
    np.testing.assert_array_equal(spyglass_raw_data, nwb_raw_data)
    print("Test passed: Raw data matches between Spyglass and NWB file.")


def main():
    nwbfile_path = Path("/media/alessandra/HD2/kind_lab_conversion_nwb/Spyglass/raw/mock_ephys.nwb")
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)

    if sgc.Session & {"nwb_file_name": nwb_copy_file_name}:
        (sgc.Session & {"nwb_file_name": nwb_copy_file_name}).delete()
    if sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}:
        (sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}).delete()
    if sgc.ProbeType & {"probe_type": "probe_type"}:
        (sgc.ProbeType & {"probe_type": "probe_type"}).delete()

    sgi.insert_sessions(str(nwbfile_path), rollback_on_fail=True, raise_err=True)
    print("=== Session ===")
    print(sgc.Session & {"nwb_file_name": nwb_copy_file_name})
    print("=== NWB File ===")
    print(sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name})
    print("=== Electrode ===")
    print(sgc.Electrode & {"nwb_file_name": nwb_copy_file_name})
    print("=== Electrode Group ===")
    print(sgc.ElectrodeGroup & {"nwb_file_name": nwb_copy_file_name})
    print("=== Probe ===")
    print(sgc.Probe & {"probe_type": "probe_type"})
    print("=== Probe Shank ===")
    print(sgc.Probe.Shank & {"probe_type": "probe_type"})
    print("=== Probe Electrode ===")
    print(sgc.Probe.Electrode & {"probe_type": "probe_type"})
    print("=== Raw ===")
    print(sgc.Raw & {"nwb_file_name": nwb_copy_file_name})

    test_ephys(nwbfile_path=nwbfile_path)


if __name__ == "__main__":
    main()
    print("Done!")
