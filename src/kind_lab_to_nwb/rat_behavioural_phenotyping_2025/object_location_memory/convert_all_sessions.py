"""Script to convert all Object Location Memory sessions in a directory."""

from pathlib import Path
import yaml
from .convert_session import session_to_nwb


def convert_all_sessions(
    data_dir: str,
    output_dir: str,
    metadata_path: str = None,
    stub_test: bool = False,
):
    """
    Convert all Object Location Memory sessions in a directory to NWB format.

    Parameters
    ----------
    data_dir : str
        Path to directory containing all session data
    output_dir : str
        Directory where NWB files will be saved
    metadata_path : str, optional
        Path to YAML file with metadata
    stub_test : bool, default: False
        If True, truncates data for quick testing
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_files = list(data_dir.glob("**/*.ts")) + list(data_dir.glob("**/*.mkv"))

    for video_path in video_files:
        # Extract subject and session info from video filename
        # Assuming filename format: sub-{subject_id}_ses-{session_id}.*
        parts = video_path.stem.split("_")
        subject_id = parts[0].replace("sub-", "")
        session_id = parts[1].replace("ses-", "")

        # Look for corresponding behavior file
        behavior_path = None
        potential_behavior_files = [
            video_path.parent / f"{video_path.stem}.boris",
            video_path.parent / f"{video_path.stem}.xls",
            video_path.parent / f"{video_path.stem}.xlsx",
        ]
        for path in potential_behavior_files:
            if path.exists():
                behavior_path = path
                break

        if behavior_path is None:
            print(f"Warning: No behavior file found for {video_path}")
            continue

        try:
            # Convert session
            session_to_nwb(
                video_path=str(video_path),
                behavior_path=str(behavior_path),
                output_dir=str(output_dir),
                subject_id=subject_id,
                session_id=session_id,
                metadata_path=metadata_path,
                stub_test=stub_test,
            )
            print(f"Successfully converted session {subject_id}_{session_id}")
        except Exception as e:
            print(f"Error converting session {subject_id}_{session_id}: {str(e)}")


if __name__ == "__main__":
    import sys

    # Get command line arguments
    if len(sys.argv) < 3:
        print("Usage: python convert_all_sessions.py data_dir output_dir [metadata_path]")
        sys.exit(1)

    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    metadata_path = sys.argv[3] if len(sys.argv) > 3 else None

    # Convert all sessions
    convert_all_sessions(
        data_dir=data_dir,
        output_dir=output_dir,
        metadata_path=metadata_path,
    )
