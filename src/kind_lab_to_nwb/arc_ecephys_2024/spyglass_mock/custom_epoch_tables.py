import datajoint as dj
from spyglass.utils import SpyglassMixin
from spyglass.common.common_task import TaskEpoch
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.utils.nwb_helper_fn import get_nwb_file

schema = dj.schema("custom_epoch_tables")


@schema
class TaskLEDs(SpyglassMixin, dj.Imported):
    """Table to accompany spyglass.common.TaskEpoch with extra information about LEDs used in tasks."""

    definition = """
    -> TaskEpoch # Inherit primary key from TaskEpoch
    ---
    custom_data_string : varchar(32) # string of max length 32
    """

    def make(self, key):
        """Populate TaskLEDs from the epoch table in the NWB file."""

        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile().get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)

        epochs = nwbf.epochs.to_dataframe()

        # Create a list of dictionaries to insert
        epoch_inserts = []
        for _, epoch_data in epochs.iterrows():
            epoch = int(epoch_data.tags[0])
            epoch_inserts.append(
                {
                    "nwb_file_name": nwb_file_name,
                    "epoch": epoch,
                    "custom_data_string": epoch_data.custom_data_string,
                }
            )
        self.insert(epoch_inserts, allow_direct_insert=True, skip_duplicates=True)
