"""Ingest all data from a converted NWB file into a spyglass database."""

from pathlib import Path

import datajoint as dj
import numpy as np
from pynwb import NWBHDF5IO

dj.conn(use_tls=False)  # configurable in config using [datababse.use_tls]

dj_local_conf_path = (
    "/home/alessandra/CatalystNeuro/kind-lab-to-nwb/dj_local_conf.json"
)
dj.config.load(dj_local_conf_path)  # load config for database connection info

# to prevent needing to load before import, consider saving the config
dj.config.save_global()  # default loaded for anywhere on the system
dj.config.save_local()  # loaded when running a script from this directory

from pynwb.ecephys import LFP, ElectricalSeries

# spyglass.common has the most frequently used tables
import spyglass.common as sgc  # this import connects to the database

# spyglass.data_import has tools for inserting NWB files into the database
import spyglass.data_import as sgi

# LFP Imports
import spyglass.lfp as sglfp
import spyglass.spikesorting.v1 as sgs
from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
from spyglass.spikesorting.spikesorting_merge import (
    SpikeSortingOutput,
)  # This import is necessary for the spike sorting to be loaded properly
from spyglass.utils.nwb_helper_fn import (
    estimate_sampling_rate,
    get_nwb_copy_filename,
)


def clean_all_db():
    # Don't need to delete children/descendants of other deletes
    # to see full table names, run Nwbfile.descendants()
    # to see familiar names, use datajoint.utils.to_camel_case(t.split('_')[1])
    # to see populated immediate children, run Nwbfile.children(as_objects=True)

    # sgc.Session.delete()  # Descendant of Nwbfile
    sgc.Nwbfile.delete()
    # sgc.DIOEvents.delete() # Descendant of Nwbfile
    # sgc.Electrode.delete()  # Descendant of Nwbfile
    # sgc.ElectrodeGroup.delete()  # Descendant of Nwbfile
    sgc.Probe.delete()
    sgc.ProbeType.delete()
    # sgc.Raw.delete()  # Descendant of Nwbfile
    sgc.DataAcquisitionDevice.delete()
    sgc.IntervalList.delete()
    sgc.Task.delete()
    # sgc.TaskEpoch.delete()  # Descendant of Nwbfile
    # sgc.VideoFile.delete()  # Descendant of Nwbfile
    sgc.CameraDevice.delete()


# To fully nuke a database, see `drop_schemas` here:
# https://github.com/CBroz1/datajoint-utilities/tree/main/datajoint_utilities/dj_search#schema-search


def clean_db_entry(nwbfile_path):
    """
    Delete all entries related to the NWB file in the database.

    1. Removing deletes for all tables that are descendants of Nwbfile.
    2. Deleting on an empty table will not raise an error, just a warning.
    3. Fetching `as_dict=True` returns a list of dicts, which can be used as
         an 'OR' query for deletion.
    """
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    nwb_dict = {"nwb_file_name": nwb_copy_file_name}
    (sgc.Nwbfile & nwb_dict).delete()
    # Removed deletes for all tables that are descendants of Nwbfile
    probe_ids = (sgc.ElectrodeGroup & nwb_dict).fetch("probe_id", as_dict=True)
    # as_dict=True returns a list of dicts, so we need to extract the probe_id
    (sgc.ProbeType & probe_ids).delete()
    sgc.DataAcquisitionDevice.delete()
    (sgc.IntervalList & nwb_dict).delete()
    sgc.Task.delete()
    camera_names = (sgc.VideoFile & nwb_dict).fetch("camera_name", as_dict=True)
    (sgc.CameraDevice & camera_names).delete()
    (sgc.SensorData & nwb_dict).delete()


def insert_lfp(nwbfile_path: Path):
    """
    Insert LFP data from an NWB file into a spyglass database.

    CB: Why not use `ImportedLFP.populate`?

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
    lfp_eseries = raw_nwbfile.processing["ecephys"]["LFP"].electrical_series[
        "lfp_series"
    ]
    eseries_kwargs = {
        "data": lfp_eseries.data,
        "rate": lfp_eseries.rate,
        "starting_time": lfp_eseries.starting_time,
        "description": lfp_eseries.description,
    }

    # Create dynamic table region and electrode series, write/close file
    analysis_io = NWBHDF5IO(
        path=analysis_file_abspath, mode="a", load_namespaces=True
    )
    analysis_nwbfile = analysis_io.read()

    # get the indices of the electrodes in the electrode table
    electrodes_table = analysis_nwbfile.electrodes.to_dataframe()
    # filter the electrodes table to only include the electrodes used for LFP
    electrodes_table = electrodes_table[
        electrodes_table["group_name"].str.contains("LFP")
    ]
    lfp_electrode_indices = electrodes_table.index.tolist()

    electrode_table_region = analysis_nwbfile.create_electrode_table_region(
        lfp_electrode_indices, "filtered electrode table"
    )
    eseries_kwargs["name"] = "filtered data"
    eseries_kwargs["electrodes"] = electrode_table_region
    es = ElectricalSeries(**eseries_kwargs)
    lfp_object_id = es.object_id
    ecephys_module = analysis_nwbfile.create_processing_module(
        name="ecephys", description="ecephys module"
    )
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


def log_table(table, restriction=True):
    """Return a formatted header string for a table."""
    resticted_tbl = table & restriction
    return f"=== {table.__name__} ===\n{resticted_tbl}\n"


def print_tables(nwbfile_path):
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    nwb_dict = {"nwb_file_name": nwb_copy_file_name}
    probe_ids = (sgc.ElectrodeGroup & nwb_dict).fetch("probe_id", as_dict=True)
    camera_names = (sgc.VideoFile & nwb_dict).fetch("camera_name", as_dict=True)
    table_list = [  # list of tuples with (table, restriction)
        (sgc.Nwbfile, nwb_dict),
        (sgc.Session, nwb_dict),
        (sgc.DIOEvents, nwb_dict),
        (sgc.Electrode, nwb_dict),
        (sgc.ElectrodeGroup, nwb_dict),
        (sgc.Probe, probe_ids),
        (sgc.Probe.Shank, probe_ids),
        (sgc.Probe.Electrode, probe_ids),
        (sgc.Raw, nwb_dict),
        (sgc.DataAcquisitionDevice, nwb_dict),
        (sgc.IntervalList, nwb_dict),
        (sgc.Task, True),
        (sgc.VideoFile, nwb_dict),
        (sgc.CameraDevice, camera_names),
        (sgc.TaskEpoch, True),
        (sgc.SensorData, nwb_dict),
        (sglfp.ImportedLFP, True),
    ]
    with open("tables.txt", "w") as f:
        for table, restriction in table_list:
            print(log_table(table, restriction), file=f)


# Alternatively, to see all entries associated with a given upstream entry...
# from spyglass.utils.dj_graph import RestrGraph
#
# rg = RestrGraph(
#     seed_table=sgc.Nwbfile,  # Any table
#     leaves=dict(
#         table_name=sgc.Nwbfile.full_table_name,  # Node to search from
#         restriction='nwb_file_name="this-file_.nwb"',  # must be a string restr
#     ),
#     direction="down",  # 'down' for descendants, 'up' for ancestors
#     verbose=True,  # Log output to see connections
#     cascade=True,  # Auto-run cascade process
# )
# rg.restr_ft  # list of all tables connected to the restricted leaf


def main():
    nwbfile_path = Path(
        "/media/alessandra/HD2/kind_lab_conversion_nwb/Spyglass/raw/sub_Rat_1717-ses_Baseline_tone_flash_hab.nwb"
    )

    clean_db_entry(nwbfile_path)

    sgi.insert_sessions(
        str(nwbfile_path), rollback_on_fail=True, raise_err=True
    )
    insert_lfp(nwbfile_path)

    print_tables(nwbfile_path)


if __name__ == "__main__":
    # clean_all_db()
    main()
