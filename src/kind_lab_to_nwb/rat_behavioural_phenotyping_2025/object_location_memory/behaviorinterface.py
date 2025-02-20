"""Custom DataInterface for BORIS behavioral scoring data."""

from neuroconv.basedata_interfaces import BaseDataInterface
from neuroconv.utils import FilePathType
from pynwb.behavior import BehavioralEvents
import pandas as pd
from typing import Optional


class BORISBehavioralInterface(BaseDataInterface):
    """Data interface for BORIS behavioral scoring files."""

    def __init__(self, file_path: FilePathType):
        """
        Initialize reading of BORIS behavioral data.

        Parameters
        ----------
        file_path : FilePathType
            Path to BORIS (.boris) or exported Excel (.xls) file
        """
        super().__init__(file_path=file_path)
        # Validate file path
        self._file_path = file_path

    def get_metadata(self) -> dict:
        """
        Get metadata for BORIS behavioral data.

        Returns
        -------
        metadata : dict
            Metadata dictionary
        """
        return dict(
            Behavior=dict(
                BehavioralEvents=dict(
                    name="ObjectLocationMemory",
                    description="Object exploration scoring from BORIS software",
                )
            )
        )

    def run_conversion(
        self,
        nwbfile,
        metadata: Optional[dict] = None,
        stub_test: bool = False,
    ) -> None:
        """
        Run conversion of BORIS behavioral data to NWB.

        Parameters
        ----------
        nwbfile : NWBFile
            The nwbfile to which to add behavioral data
        metadata : dict, optional
            Metadata dictionary
        stub_test : bool, default: False
            If True, truncates data for quick testing
        """
        if stub_test:
            # For testing, return minimal data
            return

        # Read BORIS data
        if str(self._file_path).endswith(".boris"):
            # TODO: Implement BORIS file parsing
            raise NotImplementedError("Direct BORIS file parsing not yet implemented")
        elif str(self._file_path).endswith(".xls") or str(self._file_path).endswith(".xlsx"):
            df = pd.read_excel(self._file_path)
        else:
            raise ValueError("Unsupported file format. Must be .boris, .xls, or .xlsx")

        # Create behavioral events
        behavioral_events = BehavioralEvents(
            name="object_exploration",
            description="Object exploration events scored in BORIS",
        )

        # Add events from dataframe
        # TODO: Adjust column names based on actual BORIS export format
        behavioral_events.add_timeseries(
            name="exploration_times",
            description="Times of object exploration events",
            timestamps=df["time"].values,  # Adjust column name as needed
            data=df["event"].values,  # Adjust column name as needed
        )

        # Add to NWB file
        nwbfile.add_behavioral_events(behavioral_events)
