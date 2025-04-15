"""Primary script to run to convert an entire session for of data using the NWBConverter."""
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import FilePath

from kind_lab_to_nwb.rat_behavioural_phenotyping_2025.auditory_fear_conditioning import (
    AuditoryFearConditioningNWBConverter,
)
from neuroconv.utils import dict_deep_update, load_dict_from_file


def _convert_ffii_to_avi(
    folder_path: Union[str, Path], convert_ffii_repo_path: Optional[Union[str, Path]] = None, frame_rate: int = 15
) -> None:
    """
    Convert ffii files in a directory to avi format using the convert-ffii tool.
    This utility function requires ffmpeg (https://ffmpeg.org/download.html) to be installed on the system prior to use.

    Notes
    -----
    The video conversion script is from https://github.com/jf-lab/convert-ffii.git
    The converted avi files will be saved in the same directory as the input files.

    This utility function checks if the convert-ffii repository is available,
    clones it if needed, and runs the conversion script on the specified directory.

    Parameters
    ----------
    folder_path : Union[str, Path]
        Path to the directory containing the ffii files to be converted
    convert_ffii_repo_path : Optional[Union[str, Path]], optional
        Path where the convert-ffii repository should be or will be cloned to.
        If None, will use the current working directory.
    frame_rate : int, optional
        Frame rate to use for the video conversion, by default 15 Hz

    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Input directory '{folder_path}' does not exist.")

    convert_ffii_repo_path = Path(convert_ffii_repo_path) if convert_ffii_repo_path else None
    if convert_ffii_repo_path is None:
        convert_ffii_repo_path = Path.cwd() / "convert-ffii"

    if not convert_ffii_repo_path.exists():
        print(f"Cloning convert-ffii repository to {str(convert_ffii_repo_path)}...")
        subprocess.run(
            ["git", "clone", "https://github.com/jf-lab/convert-ffii.git", str(convert_ffii_repo_path)], check=True
        )
    else:
        print(f"Using existing convert-ffii repository at '{str(convert_ffii_repo_path)}'.")

    # Run the conversion script
    ffii_to_avi_conversion_script = convert_ffii_repo_path / "ffii2avi_recursive.py"
    if not ffii_to_avi_conversion_script.exists():
        raise FileNotFoundError(f"Conversion script not found at '{ffii_to_avi_conversion_script}'.")

    subprocess.run(
        ["python", str(ffii_to_avi_conversion_script), str(folder_path), "--fps", str(frame_rate)], check=True
    )


def session_to_nwb(
    nwbfile_path: Union[str, Path],
    video_file_path: Union[FilePath, str],
    freeze_log_file_path: Union[FilePath, str],
    session_id: str,
    overwrite: bool = False,
):
    """
    Convert a session of auditory fear conditioning task to NWB format.

    Parameters
    ----------
    nwbfile_path : Union[str, Path]
        The path to the NWB file to be created.
    video_file_path: Union[FilePath, str]
        The path to the video file to be converted.
    freeze_log_file_path: Union[FilePath, str]
        The path to the freeze log file.
    """
    nwbfile_path = Path(nwbfile_path)
    nwbfile_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_data = dict()
    conversion_options = dict()

    # Add Behavioral Video
    source_data.update(dict(Video=dict(file_paths=[video_file_path])))
    conversion_options.update(dict(Video=dict()))

    converter = AuditoryFearConditioningNWBConverter(source_data=source_data, verbose=True)

    metadata = converter.get_metadata()
    # Update default metadata with the editable in the corresponding yaml file
    editable_metadata_path = Path(__file__).parent / "metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    metadata = dict_deep_update(
        metadata,
        editable_metadata,
    )

    metadata["NWBFile"]["session_id"] = session_id
    metadata["NWBFile"]["session_description"] = metadata["SessionTypes"][session_id]["session_description"]

    # Read freeze log file
    freeze_log = pd.read_excel(freeze_log_file_path, header=None)
    # TODO: replace with the actual column names
    row = freeze_log.values[0]
    # Combine into a single datetime object
    session_start_time = datetime.combine(row[-2].date(), row[-1])
    session_start_time = session_start_time.replace(tzinfo=ZoneInfo("Europe/London"))
    metadata["NWBFile"].update(session_start_time=session_start_time)

    # Run conversion
    converter.run_conversion(
        metadata=metadata,
        nwbfile_path=nwbfile_path,
        conversion_options=conversion_options,
        overwrite=overwrite,
    )


if __name__ == "__main__":

    # Parameters for conversion
    nwbfile_path = "/Users/weian/data/1_HabD1/Box3_Arid1b(3)_HabD1_408.nwb"

    # TODO: move this to convert_sessions.py
    # _convert_ffii_to_avi(
    #     folder_path="/Users/weian/data/1_HabD1",
    #     convert_ffii_repo_path=None,
    #     frame_rate=15, # TODO: confirm frame rate
    # )

    # Path to the video file (.avi)
    video_file_path = "/Users/weian/data/1_HabD1/Box3_Arid1b(3)_HabD1_408.avi"
    # Path to the excel file containing metadata
    freeze_log_file_path = "/Users/weian/data/1_HabD1/Freeze_Log.xls"

    # TODO: read from excel (see Marble Interaction for reference)
    session_id = "1_HabD1"

    stub_test = False
    overwrite = True

    session_to_nwb(
        nwbfile_path=nwbfile_path,
        video_file_path=video_file_path,
        freeze_log_file_path=freeze_log_file_path,
        session_id=session_id,
        overwrite=overwrite,
    )
