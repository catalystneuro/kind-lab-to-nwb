"""Insert annotated events data into SpyGlass database.

This module provides utilities for inserting custom annotated events data
into the SpyGlass database using the AnnotatedEvents extension table.
"""

from pathlib import Path
from typing import Union

# General Spyglass Imports
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename, get_nwb_file

from kind_lab_to_nwb.spyglass_utils import AnnotatedEvents


def insert_annotated_events(nwbfile_path: Union[str, Path]) -> None:
    """Insert annotated events data into SpyGlass database.

    Processes annotated behavioral events data using the custom AnnotatedEvents
    extension table. This handles experiment-specific behavioral markers
    and task events that are not part of standard SpyGlass tables.

    Parameters
    ----------
    nwbfile_path : Union[str, Path]
        Path to the NWB file containing annotated events data.
    """
    # Convert to Path object for consistent handling
    nwbfile_path = Path(nwbfile_path)
    nwbf = get_nwb_file(nwbfile_path)
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    AnnotatedEvents().insert_from_nwbfile(nwb_copy_file_name, nwbf)
