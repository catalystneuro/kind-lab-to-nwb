from pydantic import FilePath
import pandas as pd


def extract_subject_metadata_from_excel(subjects_metadata_file_path: FilePath) -> dict:
    """Extract subject metadata from an excel sheet for all subjects.

    Parameters
    ----------
    subjects_metadata_file_path : Path
        Path to the Excel file containing subject metadata

    Returns
    -------
    dict
        Dictionary containing the metadata for all subjects, with keys as 'RAT ID_Line'
    """

    # Read the Excel file
    df = pd.read_excel(subjects_metadata_file_path)

    # Create a dictionary to store all subjects metadata
    all_subjects_metadata = {}

    # Iterate through each row in the dataframe
    for _, row in df.iterrows():
        # Create unique key combining RAT ID and Line
        subject_key = f"{row['animal ID']}_{row['line']}"

        # Convert row to dict
        metadata = row.to_dict()
        del metadata["animal ID"]

        all_subjects_metadata[subject_key] = metadata

    return all_subjects_metadata
