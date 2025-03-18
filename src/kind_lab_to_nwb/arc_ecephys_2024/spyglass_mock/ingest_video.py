"""Ingest mock video data from an NWB file into a spyglass database."""

import datajoint as dj
from pathlib import Path

dj.conn(use_tls=False)
dj_local_conf_path = "/home/alessandra/CatalystNeuro/kind-lab-to-nwb/dj_local_conf.json"
dj.config.load(dj_local_conf_path)  # load config for database connection info

# spyglass.common has the most frequently used tables
import spyglass.common as sgc  # this import connects to the database

# spyglass.data_import has tools for inserting NWB files into the database
import spyglass.data_import as sgi
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename


def main():
    nwbfile_path = Path("/media/alessandra/HD2/kind_lab_conversion_nwb/Spyglass/raw/mock_video.nwb")
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)

    if sgc.Session & {"nwb_file_name": nwb_copy_file_name}:
        (sgc.Session & {"nwb_file_name": nwb_copy_file_name}).delete()
    if sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}:
        (sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}).delete()

    sgi.insert_sessions(str(nwbfile_path), rollback_on_fail=True, raise_err=True)
    print("=== Session ===")
    print(sgc.Session & {"nwb_file_name": nwb_copy_file_name})
    print("=== NWB File ===")
    print(sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name})
    print("=== Camera Device ===")
    print(sgc.CameraDevice & {"camera_name": "my_camera_name"})
    print("=== Video File ===")
    print(sgc.VideoFile & {"camera_name": "my_camera_name"})


if __name__ == "__main__":
    main()
    print("Done!")
