"""Ingest all data from a converted NWB file into a spyglass database."""

# %%
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

from kind_lab_to_nwb.spyglass_utils import (
    clean_db_entry,
    AnnotatedEvents,
    print_tables,
)


nwb_file_name = "sub-1072-Grin2b(6)_ses-OR-HabD1.nwb"
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
# %%
# Alternatively, to see all entries associated with a given upstream entry...
from spyglass.utils.dj_graph import RestrGraph

rg = RestrGraph(
    seed_table=sgc.Nwbfile,  # Any table
    leaves=dict(
        table_name=sgc.Nwbfile.full_table_name,  # Node to search from
        restriction='nwb_file_name="sub-1072-Grin2b(6)_ses-OR-HabD1_.nwb"',  # must be a string restr
    ),
    direction="down",  # 'down' for descendants, 'up' for ancestors
    verbose=True,  # Log output to see connections
    cascade=True,  # Auto-run cascade process
)
rg.restr_ft  # list of all tables connected to the restricted leaf

# %%
