# spyglass.common has the most frequently used tables
import spyglass.common as sgc  # this import connects to the database

from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

from kind_lab_to_nwb.spyglass_utils import AnnotatedEvents


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
