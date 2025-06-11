from pathlib import Path
import sys

# General Spyglass Imports
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

# Custom Table Imports
sys.path.append(
    "C:/Users/Utente/CatalystNeuro/kind-lab-to-nwb/src/kind_lab_to_nwb/rat_behavioural_phenotyping_2025/spyglass_utils/events_table.py"
)
from events_table import AnnotatedEvents


def insert_task(nwbfile_path: Path):
    """Insert custom task LED data into SpyGlass database.

    Processes task-specific LED behavioral data using the custom TaskLEDs
    extension table. This handles experiment-specific behavioral markers
    and task events that are not part of standard SpyGlass tables.

    Parameters
    ----------
    nwbfile_path : Path
        Path to the NWB file containing task and behavioral data.
    """
    nwb_copy_file_name = get_nwb_copy_filename(nwbfile_path.name)
    AnnotatedEvents().make(key={"nwb_file_name": nwb_copy_file_name})
