"""SpyGlass extension for annotated events data.

This module provides a custom DataJoint table that extends SpyGlass functionality
to store annotated events data from behavioral experiments. It extracts event timing,
descriptions, labels and durations from NWB files and stores them in a structured
database format for analysis and querying.
"""

import datajoint as dj
import pynwb

from spyglass.utils import SpyglassMixin
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.utils.nwb_helper_fn import get_nwb_file

schema = dj.schema("annotated_events")


@schema
class AnnotatedEvents(SpyglassMixin, dj.Manual):
    """Custom SpyGlass table for storing annotated events data.

    This table extends the standard SpyGlass table to include annotated events data
    in behavioral experiments.

    The table adds annotated events-specific attributes, enabling queries that link task parameters with behavioral events details.
    """

    definition = """
    # Annotated events data
    -> Nwbfile
    ---
    annotated_events_object_id=NULL: varchar(40)  # Object ID for annotated events table in NWB file
    """
    _nwb_table = Nwbfile

    def insert_from_nwbfile(
        self, nwb_file_name: str, nwbf: pynwb.NWBFile, data_interface_name: str = "AnnotatedBehavioralEvents"
    ):
        """Insert annotated events from an NWB file.

        Parameters
        ----------
        nwb_file_name : str
            The name of the NWB file.
        nwbf : pynwb.NWBFile
            The source NWB file object.

        Notes
        -----
        This method expects the NWB file to contain a 'events' processing module
        with annotated events tables that include 'event_times', 'event_description','label', and
        'duration' columns.
        """
        nwb_dict = dict(nwb_file_name=nwb_file_name)

        self_insert = nwb_dict.copy()
        if data_interface_name not in nwbf.processing["behavior"].data_interfaces:
            print(f"No {data_interface_name} found in {nwb_file_name}. Skipping.")
            return

        annotated_events = nwbf.processing["behavior"][data_interface_name]
        annotated_events_object_id = annotated_events.object_id
        self_insert["annotated_events_object_id"] = annotated_events_object_id
        self.insert1(self_insert)

    def fetch1_dataframe(self, table_name: str):
        """Fetch a DataFrame for a specific table name."""
        if table_name not in ["annotated_events"]:
            raise ValueError(f"Invalid table name: {table_name}")

        _ = self.ensure_single_entry()
        return self.fetch_nwb()[0][table_name]
