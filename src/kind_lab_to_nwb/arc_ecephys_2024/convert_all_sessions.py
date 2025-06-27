"""Primary script to run to convert all sessions in a dataset using session_to_nwb."""

import traceback
from pathlib import Path
from pprint import pformat
from typing import Union
from tqdm import tqdm

from neuroconv.tools.path_expansion import LocalPathExpander

from kind_lab_to_nwb.arc_ecephys_2024.convert_session import (
    session_to_nwb,
)


def dataset_to_nwb(
    *,
    data_dir_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    verbose: bool = False,
):
    """Convert the entire dataset to NWB.

    Parameters
    ----------
    data_dir_path : Union[str, Path]
        The path to the directory containing the raw data.
    output_dir_path : Union[str, Path]
        The path to the directory where the NWB files will be saved.
    verbose : bool, optional
        Whether to print verbose output, by default True
    """
    data_dir_path = Path(data_dir_path)
    output_dir_path = Path(output_dir_path)
    session_to_nwb_kwargs_per_session = get_session_to_nwb_kwargs_per_session(
        data_dir_path=data_dir_path,
    )

    if verbose:
        print(f"Found {len(session_to_nwb_kwargs_per_session)} sessions to convert")

    for session_to_nwb_kwargs in tqdm(session_to_nwb_kwargs_per_session, desc="Converting sessions"):
        session_to_nwb_kwargs["output_dir_path"] = output_dir_path
        session_to_nwb_kwargs["verbose"] = verbose

        # Create meaningful error file name using subject and session info
        subject_id = session_to_nwb_kwargs["path_expander_metadata"]["metadata"]["Subject"]["subject_id"]
        session_id = session_to_nwb_kwargs["path_expander_metadata"]["metadata"]["NWBFile"]["session_id"]
        exception_file_path = output_dir_path / f"ERROR_sub_{subject_id}-ses_{session_id}.txt"

        safe_session_to_nwb(
            session_to_nwb_kwargs=session_to_nwb_kwargs,
            exception_file_path=exception_file_path,
        )


def safe_session_to_nwb(
    *,
    session_to_nwb_kwargs: dict,
    exception_file_path: Union[Path, str],
):
    """Convert a session to NWB while handling any errors by recording error messages to the exception_file_path.

    Parameters
    ----------
    session_to_nwb_kwargs : dict
        The arguments for session_to_nwb.
    exception_file_path : Path
        The path to the file where the exception messages will be saved.
    """
    exception_file_path = Path(exception_file_path)
    try:
        session_to_nwb(**session_to_nwb_kwargs)
    except Exception as e:
        with open(
            exception_file_path,
            mode="w",
        ) as f:
            f.write(f"session_to_nwb_kwargs: \n {pformat(session_to_nwb_kwargs)}\n\n")
            f.write(traceback.format_exc())


def get_session_to_nwb_kwargs_per_session(
    *,
    data_dir_path: Union[str, Path],
):
    """Get the kwargs for session_to_nwb for each session in the dataset.

    Parameters
    ----------
    data_dir_path : Union[str, Path]
        The path to the directory containing the raw data.

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries containing the kwargs for session_to_nwb for each session.
    """
    data_dir_path = Path(data_dir_path)

    # Define the source data specification pattern used to find sessions
    source_data_spec = {
        "OpenEphysRecording": {
            "base_directory": data_dir_path,
            "folder_path": "{subject_id}/{session_id}/{subject_id}_{session_date}_{session_time}_{task}",
        },
    }

    # Instantiate LocalPathExpander
    path_expander = LocalPathExpander()

    # Expand paths and extract metadata
    metadata_list = path_expander.expand_paths(source_data_spec)

    # Create kwargs list for each session
    session_to_nwb_kwargs_per_session = []
    for path_expander_metadata in metadata_list:
        session_kwargs = {
            "data_dir_path": data_dir_path,
            "path_expander_metadata": path_expander_metadata,
        }
        session_to_nwb_kwargs_per_session.append(session_kwargs)

    return session_to_nwb_kwargs_per_session


if __name__ == "__main__":

    # Parameters for conversion
    # data_dir_path = Path("/media/alessandra/HD2/Kind-CN-data-share/neuronal_circuits/fear_conditionning_paradigm")
    data_dir_path = Path("D:/Kind-CN-data-share/neuronal_circuits/fear_conditionning_paradigm")
    # output_dir_path = Path("/media/alessandra/HD2/kind_lab_conversion_nwb")
    output_dir_path = Path("D:/kind_lab_conversion_nwb")
    verbose = False

    dataset_to_nwb(
        data_dir_path=data_dir_path,
        output_dir_path=output_dir_path,
        verbose=verbose,
    )
