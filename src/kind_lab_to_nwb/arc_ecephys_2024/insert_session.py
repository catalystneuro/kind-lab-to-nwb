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

# LFP Imports
import spyglass.lfp as sglfp
from spyglass.utils.nwb_helper_fn import estimate_sampling_rate
from pynwb.ecephys import ElectricalSeries, LFP


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


def insert_lfp(nwbfile_path: Path):
    """
    Insert LFP data from an NWB file into a spyglass database.

    Parameters
    ----------
    nwbfile_path : Path
        The path to the NWB file to insert.
    """
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    lfp_file_name = sgc.AnalysisNwbfile().create(nwb_copy_file_name)
    analysis_file_abspath = sgc.AnalysisNwbfile().get_abs_path(lfp_file_name)

    raw_io = NWBHDF5IO(nwbfile_path, "r")
    raw_nwbfile = raw_io.read()
    lfp_eseries = raw_nwbfile.processing["ecephys"]["LFP"].electrical_series["lfp_series"]
    eseries_kwargs = {
        "data": lfp_eseries.data,
        # "timestamps": lfp_eseries.timestamps,
        "rate": lfp_eseries.rate,
        "starting_time": lfp_eseries.starting_time,
        "description": lfp_eseries.description,
    }

    # Create dynamic table region and electrode series, write/close file
    analysis_io = NWBHDF5IO(path=analysis_file_abspath, mode="a", load_namespaces=True)
    analysis_nwbfile = analysis_io.read()

    # get the indices of the electrodes in the electrode table
    electrodes_table = analysis_nwbfile.electrodes.to_dataframe()
    # filter the electrodes table to only include the electrodes used for LFP
    electrodes_table = electrodes_table[electrodes_table["group_name"].str.contains("LFP")]
    lfp_electrode_indices = electrodes_table.index.tolist()

    electrode_table_region = analysis_nwbfile.create_electrode_table_region(
        lfp_electrode_indices, "filtered electrode table"
    )
    eseries_kwargs["name"] = "filtered data"
    eseries_kwargs["electrodes"] = electrode_table_region
    es = ElectricalSeries(**eseries_kwargs)
    lfp_object_id = es.object_id
    ecephys_module = analysis_nwbfile.create_processing_module(name="ecephys", description="ecephys module")
    ecephys_module.add(LFP(electrical_series=es))
    analysis_io.write(analysis_nwbfile, link_data=False)
    analysis_io.close()

    sgc.AnalysisNwbfile().add(nwb_copy_file_name, lfp_file_name)

    lfp_electrode_group_name = "lfp_electrode_group"
    sglfp.lfp_electrode.LFPElectrodeGroup.create_lfp_electrode_group(
        nwb_file_name=nwb_copy_file_name,
        group_name=lfp_electrode_group_name,
        electrode_list=lfp_electrode_indices,
    )
    # lfp_sampling_rate = estimate_sampling_rate(eseries_kwargs["timestamps"][:1_000_000])
    lfp_sampling_rate = eseries_kwargs["rate"]
    key = {
        "nwb_file_name": nwb_copy_file_name,
        "lfp_electrode_group_name": lfp_electrode_group_name,
        "interval_list_name": "raw data valid times",
        "lfp_sampling_rate": lfp_sampling_rate,
        "lfp_object_id": lfp_object_id,
        "analysis_file_name": lfp_file_name,
    }
    sglfp.ImportedLFP.insert1(key, allow_direct_insert=True)
    sglfp.lfp_merge.LFPOutput.insert1(key, allow_direct_insert=True)

    raw_io.close()


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
        print("=== ImportedLFP ===", file=f)
        print(sglfp.ImportedLFP(), file=f)


def main():
    nwbfile_path = Path(
        "/media/alessandra/HD2/kind_lab_conversion_nwb/Spyglass/raw/sub_Rat_1717-ses_Baseline_tone_flash_hab.nwb"
    )

    clean_db_entry(nwbfile_path)

    sgi.insert_sessions(str(nwbfile_path), rollback_on_fail=True, raise_err=True)
    insert_lfp(nwbfile_path)

    print_tables(nwbfile_path)


if __name__ == "__main__":
    # clean_all_db()
    main()
