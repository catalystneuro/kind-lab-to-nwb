"""Example script to convert an Object Location Memory session to NWB format."""

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from .nwbconverter import ObjectLocationMemoryNWBConverter
import yaml


def session_to_nwb(
    video_path: str,
    behavior_path: str,
    output_dir: str,
    subject_id: str,
    session_id: str,
    metadata_path: str = None,
    stub_test: bool = False,
):
    """
    Convert an Object Location Memory session to NWB format.

    Parameters
    ----------
    video_path : str
        Path to video recording file (.ts or .mkv)
    behavior_path : str
        Path to BORIS behavioral scoring file (.boris or .xls)
    output_dir : str
        Directory where NWB file will be saved
    subject_id : str
        ID of the subject
    session_id : str
        ID of the session
    metadata_path : str, optional
        Path to YAML file with metadata
    stub_test : bool, default: False
        If True, truncates data for quick testing
    """
    # Initialize source data
    source_data = dict(
        Video=dict(file_path=video_path),
        BehavioralScoring=dict(file_path=behavior_path),
    )

    # Initialize converter
    converter = ObjectLocationMemoryNWBConverter(source_data)

    # Get metadata schema
    metadata_schema = converter.get_metadata_schema()

    # Get default metadata
    metadata = converter.get_metadata()

    # Load and update metadata from file if provided
    if metadata_path:
        with open(metadata_path, "r") as f:
            metadata_from_file = yaml.safe_load(f)
        metadata.update(metadata_from_file)

    # Add session start time if not in metadata
    if "session_start_time" not in metadata["NWBFile"]:
        metadata["NWBFile"].update(session_start_time=datetime.now(ZoneInfo("UTC")))

    # Update subject ID and session ID
    metadata["NWBFile"].update(session_id=session_id, subject_id=subject_id)

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run conversion
    converter.run_conversion(
        nwbfile_path=str(output_dir / f"{subject_id}_{session_id}.nwb"),
        metadata=metadata,
        stub_test=stub_test,
    )


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    import sys

    # Get command line arguments
    if len(sys.argv) < 6:
        print(
            "Usage: python convert_session.py "
            "video_path behavior_path output_dir subject_id session_id [metadata_path]"
        )
        sys.exit(1)

    video_path = sys.argv[1]
    behavior_path = sys.argv[2]
    output_dir = sys.argv[3]
    subject_id = sys.argv[4]
    session_id = sys.argv[5]
    metadata_path = sys.argv[6] if len(sys.argv) > 6 else None

    # Convert session
    session_to_nwb(
        video_path=video_path,
        behavior_path=behavior_path,
        output_dir=output_dir,
        subject_id=subject_id,
        session_id=session_id,
        metadata_path=metadata_path,
    )
