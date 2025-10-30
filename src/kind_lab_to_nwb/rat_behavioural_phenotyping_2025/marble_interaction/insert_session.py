"""Ingest all data from a converted NWB file into a spyglass database."""

from pathlib import Path

import datajoint as dj

dj.conn(use_tls=False)  # configurable in config using [datababse.use_tls]

dj_local_conf_path = "/home/alessandra/CatalystNeuro/kind-lab-to-nwb/dj_local_conf.json"
dj.config.load(dj_local_conf_path)  # load config for database connection info

# to prevent needing to load before import, consider saving the config
dj.config.save_global()  # default loaded for anywhere on the system
dj.config.save_local()  # loaded when running a script from this directory


# spyglass.common has the most frequently used tables
import spyglass.common as sgc  # this import connects to the database

# spyglass.data_import has tools for inserting NWB files into the database
import spyglass.data_import as sgi

from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename, get_nwb_file

from kind_lab_to_nwb.spyglass_utils import insert_annotated_events, clean_db_entry, AnnotatedEvents


def log_table(table, restriction=True):
    """Return a formatted header string for a table."""
    resticted_tbl = table & restriction
    return f"=== {table.__name__} ===\n{resticted_tbl}\n"


def print_tables(nwbfile_path):
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    nwb_dict = {"nwb_file_name": nwb_copy_file_name}
    camera_names = (sgc.VideoFile & nwb_dict).fetch("camera_name", as_dict=True)
    table_list = [  # list of tuples with (table, restriction)
        (sgc.Nwbfile, nwb_dict),
        (sgc.Session, nwb_dict),
        (AnnotatedEvents, nwb_dict),
        (sgc.Raw, nwb_dict),
        (sgc.IntervalList, nwb_dict),
        (sgc.Task, True),
        (sgc.VideoFile, nwb_dict),
        (sgc.CameraDevice, camera_names),
        (sgc.TaskEpoch, True),
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
    nwb_file_name = "sub-408-Arid1b(3)_ses-MI-Test_image.nwb"
    nwbfile_path = Path("/media/alessandra/HD2/kind_lab_conversion_nwb/Spyglass/raw") / nwb_file_name
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    nwb_dict = dict(nwb_file_name=nwb_copy_file_name)

    clean_db_entry(nwbfile_path)

    sgi.insert_sessions(str(nwbfile_path), rollback_on_fail=True, raise_err=True)

    nwbf = get_nwb_file(nwbfile_path)

    # Insert annotated events data
    events = AnnotatedEvents()
    if not events & nwb_dict:
        events.insert_from_nwbfile(nwb_copy_file_name, nwbf)

    # Fetch actions DataFrame
    annotated_events_df = events.fetch1_dataframe("annotated_events")
    print(annotated_events_df.head())

    print_tables(nwbfile_path)


if __name__ == "__main__":
    # clean_all_db()
    main()
