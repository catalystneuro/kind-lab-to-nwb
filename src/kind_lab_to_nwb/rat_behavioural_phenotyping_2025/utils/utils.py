from pydantic import FilePath
from typing import List
import pandas as pd


def extract_subject_metadata_from_excel(subjects_metadata_file_path: FilePath) -> List[dict]:
    """Extract subject metadata from an excel sheet for all subjects.

    Parameters
    ----------
    subjects_metadata_file_path : Path
        Path to the Excel file containing subject metadata

    Returns
    -------
    List[dict]
        List of dictionaries containing the metadata for each subject
    """

    # Read the Excel file
    df = pd.read_excel(subjects_metadata_file_path)
    # Remove rows with all NaN values
    df = df.dropna(how="all")

    # Convert dataframe to list of dictionaries
    subjects_metadata = df.to_dict("records")

    # For the field "tasks conducted", convert comma-separated string to list of strings
    for subject in subjects_metadata:
        if "tasks conducted" in subject and isinstance(subject["tasks conducted"], str):
            subject["tasks conducted"] = [task.strip() for task in subject["tasks conducted"].split(",")]

    # For the fiels "animal ID" and "cohort ID" convert to int
    for subject in subjects_metadata:
        subject["animal ID"] = int(subject["animal ID"])
        subject["cohort no."] = int(subject["cohort no."])

    return subjects_metadata


def get_subject_metadata_from_task(subjects_metadata: List[dict], task_acronym: str) -> List[dict]:
    """Get subject metadata for subjects that have conducted a given task.

    Parameters
    ----------
    subjects_metadata : List[dict]
        List of dictionaries containing the metadata for each subject
    task_acronym : str
        The task acronym

    Returns
    -------
    List[dict]
        List of dictionaries containing the metadata for each subject that has conducted the given task
    """

    return [
        subject
        for subject in subjects_metadata
        if "tasks conducted" in subject and task_acronym in subject["tasks conducted"]
    ]


def get_session_ids_from_excel(subjects_metadata_file_path: FilePath, task_acronym: str) -> List[str]:
    # Read the Excel file, specifically the "Task acronyms & structure" sheet
    df = pd.read_excel(subjects_metadata_file_path, sheet_name="Task acronyms & structure")

    # Find the row corresponding to the given task_acronym
    task_row = df[df.iloc[:, 0] == task_acronym]

    if task_row.empty:
        raise ValueError(f"Task acronym '{task_acronym}' not found in Excel file")

    # Get all columns except the first three (acronym, name, pattern)
    session_ids = task_row.iloc[0, 3:].dropna().tolist()

    # Remove trailing whitespace from each session ID
    session_ids = [session_id.rstrip() for session_id in session_ids]
    return session_ids
