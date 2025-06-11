import spyglass.common as sgc  # this import connects to the database
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename


def clean_all_db():
    # Don't need to delete children/descendants of other deletes
    # to see full table names, run Nwbfile.descendants()
    # to see familiar names, use datajoint.utils.to_camel_case(t.split('_')[1])
    # to see populated immediate children, run Nwbfile.children(as_objects=True)

    sgc.Nwbfile.delete()
    sgc.Probe.delete()
    sgc.ProbeType.delete()
    sgc.DataAcquisitionDevice.delete()
    sgc.IntervalList.delete()
    sgc.Task.delete()
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
