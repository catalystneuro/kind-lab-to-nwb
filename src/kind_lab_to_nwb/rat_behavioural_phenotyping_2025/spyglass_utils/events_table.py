"""SpyGlass extension for annotated events data.

This module provides a custom DataJoint table that extends SpyGlass functionality
to store annotated events data from behavioral experiments. It extracts event timing,
descriptions, labels and durations from NWB files and stores them in a structured
database format for analysis and querying.
"""

import datajoint as dj
from spyglass.utils import SpyglassMixin
from spyglass.common.common_task import DIOEvents
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.utils.nwb_helper_fn import get_nwb_file

schema = dj.schema("annotated_events")


@schema
class AnnotatedEvents(SpyglassMixin, dj.Imported):
    """Custom SpyGlass table for storing annotated events data.

    This table extends the standard SpyGlass DIOEvents table to include annotated events data
    in behavioral experiments.

    The table inherits the primary key from the DIOEvents table and adds annotated events-specific
    attributes, enabling queries that link task parameters with behavioral events details.
    """

    definition = """
    -> DIOEvents # Inherit primary key from DIOEvents
    duration : varchar(32) # string of max length 32
    event_times : varchar(32) # string of max length 32
    event_description : varchar(32) # string of max length 32
    label : varchar(32) # string of max length 32
    """

    def make(self, key):
        """Extract and populate annotated events data from NWB file.

        Reads events information from the NWB file's processing module and extracts
        annotated events data.

        Parameters
        ----------
        key : dict
            Dictionary containing the primary key, must include 'nwb_file_name'.

        Notes
        -----
        This method expects the NWB file to contain a 'events' processing module
        with annotated events tables that include 'event_times', 'event_description','label', and
        'duration' columns.
        """

        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile().get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)

        # Create a list of dictionaries to insert
        inserts = []
        for name, events_table in nwbf.processing["events"].data_interfaces.items():
            events_table = events_table.to_dataframe()
            for _, row in events_table.iterrows():
                duration = row["duration"]
                event_times = row["event_times"]
                event_description = row["event_description"]
                label = row["label"]

                for led_name, led_position in zip(event_description, label):
                    inserts.append(
                        {
                            "duration": duration,
                            "led_name": led_name,
                            "event_times": event_times,
                            "led_position": led_position,
                        }
                    )
        self.insert(inserts, allow_direct_insert=True, skip_duplicates=True)
