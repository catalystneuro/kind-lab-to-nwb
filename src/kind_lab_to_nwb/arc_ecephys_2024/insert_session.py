"""Ingest all data from a converted NWB file into a spyglass database."""

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


def clean_all_db():
    sgc.Session.delete()
    sgc.Nwbfile.delete()
    sgc.DIOEvents.delete()
    sgc.Electrode.delete()
    sgc.ElectrodeGroup.delete()
    sgc.Probe.delete()
    sgc.ProbeType.delete()
    sgc.Raw.delete()
    sgc.DataAcquisitionDevice.delete()
    sgc.IntervalList.delete()
    sgc.Task.delete()
    sgc.TaskEpoch.delete()
    sgc.VideoFile.delete()
    sgc.CameraDevice.delete()


def clean_db_entry(nwbfile_path):
    """
    Delete all entries related to the NWB file in the database.
    """
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    if sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}:
        (sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}).delete()
    if sgc.Session & {"nwb_file_name": nwb_copy_file_name}:
        (sgc.Session & {"nwb_file_name": nwb_copy_file_name}).delete()
    if sgc.DIOEvents & {"nwb_file_name": nwb_copy_file_name}:
        (sgc.DIOEvents & {"nwb_file_name": nwb_copy_file_name}).delete()
    if sgc.Electrode & {"nwb_file_name": nwb_copy_file_name}:
        (sgc.Electrode & {"nwb_file_name": nwb_copy_file_name}).delete()
    if sgc.ElectrodeGroup & {"nwb_file_name": nwb_copy_file_name}:
        (sgc.ElectrodeGroup & {"nwb_file_name": nwb_copy_file_name}).delete()
    probe_ids = (sgc.ElectrodeGroup & {"nwb_file_name": nwb_copy_file_name}).fetch("probe_id")
    for probe_id in probe_ids:
        if sgc.Probe & {"probe_id": probe_id}:
            (sgc.Probe & {"probe_id": probe_id}).delete()
        if sgc.ProbeType & {"probe_type": probe_id}:
            (sgc.ProbeType & {"probe_type": probe_id}).delete()
    if sgc.Raw & {"nwb_file_name": nwb_copy_file_name}:
        (sgc.Raw & {"nwb_file_name": nwb_copy_file_name}).delete()
    sgc.DataAcquisitionDevice.delete()
    if sgc.IntervalList & {"nwb_file_name": nwb_copy_file_name}:
        (sgc.IntervalList & {"nwb_file_name": nwb_copy_file_name}).delete()
    sgc.Task.delete()
    sgc.TaskEpoch.delete()
    if sgc.VideoFile & {"nwb_file_name": nwb_copy_file_name}:
        (sgc.VideoFile & {"nwb_file_name": nwb_copy_file_name}).delete()
    camera_names = (sgc.VideoFile & {"nwb_file_name": nwb_copy_file_name}).fetch("camera_name")
    for camera_name in camera_names:
        if sgc.CameraDevice & {"camera_name": camera_name}:
            (sgc.CameraDevice & {"camera_name": camera_name}).delete()
    if sgc.SensorData & {"nwb_file_name": nwb_copy_file_name}:
        (sgc.SensorData & {"nwb_file_name": nwb_copy_file_name}).delete()


def print_tables(nwbfile_path):
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    with open("tables.txt", "w") as f:
        print("=== NWB File ===", file=f)
        print(sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}, file=f)
        print("=== Session ===", file=f)
        print(sgc.Session & {"nwb_file_name": nwb_copy_file_name}, file=f)
        print("=== DIOEvents ===", file=f)
        print(sgc.DIOEvents & {"nwb_file_name": nwb_copy_file_name}, file=f)
        print("=== Electrode ===", file=f)
        print(sgc.Electrode & {"nwb_file_name": nwb_copy_file_name}, file=f)
        print("=== Electrode Group ===", file=f)
        print(sgc.ElectrodeGroup & {"nwb_file_name": nwb_copy_file_name}, file=f)
        probe_ids = (sgc.ElectrodeGroup & {"nwb_file_name": nwb_copy_file_name}).fetch("probe_id")
        print("=== Probe ===", file=f)
        print(sgc.Probe & [{"probe_id": probe_id} for probe_id in probe_ids], file=f)
        print("=== Probe Shank ===", file=f)
        print(sgc.Probe.Shank & [{"probe_id": probe_id} for probe_id in probe_ids], file=f)
        print("=== Probe Electrode ===", file=f)
        print(sgc.Probe.Electrode & [{"probe_id": probe_id} for probe_id in probe_ids], file=f)
        print("=== Raw ===", file=f)
        print(sgc.Raw & {"nwb_file_name": nwb_copy_file_name}, file=f)
        print("=== DataAcquisitionDevice ===", file=f)
        print(sgc.DataAcquisitionDevice & {"nwb_file_name": nwb_copy_file_name}, file=f)
        print("=== IntervalList ===", file=f)
        print(sgc.IntervalList() & {"nwb_file_name": nwb_copy_file_name}, file=f)
        print("=== Task ===", file=f)
        print(sgc.Task(), file=f)
        print("=== VideoFile ===", file=f)
        print(sgc.VideoFile & {"nwb_file_name": nwb_copy_file_name}, file=f)
        camera_names = (sgc.VideoFile & {"nwb_file_name": nwb_copy_file_name}).fetch("camera_name")
        print("=== CameraDevice ===", file=f)
        print(sgc.CameraDevice & [{"camera_name": camera_name} for camera_name in camera_names], file=f)
        print("=== Task Epoch ===", file=f)
        print(sgc.TaskEpoch(), file=f)
        print("=== SensorData ===", file=f)
        print(sgc.SensorData(), file=f)


def main():
    nwbfile_path = Path(
        "/media/alessandra/HD2/kind_lab_conversion_nwb/Spyglass/raw/sub_Rat_1717-ses_Baseline_tone_flash_hab.nwb"
    )

    clean_db_entry(nwbfile_path)

    sgi.insert_sessions(str(nwbfile_path), rollback_on_fail=True, raise_err=True)

    print_tables(nwbfile_path)


if __name__ == "__main__":
    # clean_all_db()
    main()
