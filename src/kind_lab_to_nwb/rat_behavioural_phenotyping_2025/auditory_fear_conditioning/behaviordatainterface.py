from pathlib import Path
from typing import Union

import pandas as pd
from pynwb import NWBFile

from neuroconv import BaseDataInterface


class AuditoryFearConditioningBehavioralInterface(BaseDataInterface):
    """Adds trials from freezing behavior data."""

    keywords = ["behavior", "trials"]

    def __init__(self, file_path: Union[str, Path], identifier: str):
        super().__init__(file_path=file_path)
        self.identifier = identifier

    def read_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.source_data["file_path"])
        df.dropna(axis=1, how="all", inplace=True)

        return df

    def add_to_nwbfile(self, nwbfile: NWBFile, **kwargs) -> None:
        """
        Add trials to the NWB file.

        Parameters
        ----------
        nwbfile : NWBFile
            The NWB file to add the trials to.

        """
        # Read the data from the file
        data = self.read_data()

        threshold_column_idx = next(
            (i for i, col in enumerate(data.columns) if data[col].astype(str).str.contains("Threshold").any()), None
        )
        if threshold_column_idx is None:
            raise ValueError("No row with 'Threshold' found in the CSV file.")
        # Based on the example CSVs we have, the time bins end before the threshold column
        trial_column_end = threshold_column_idx - 1

        # trial onset values ("Onset" is the first row)
        trial_start_times = data.values[0][1:trial_column_end]
        # trial durations ("Duration" is the second row)
        durations = data.values[1][1:trial_column_end]
        # trial stop times
        trial_stop_times = trial_start_times + durations

        filtered_df = data.loc[data["% freeze"].isin([self.identifier])]
        if filtered_df.empty:
            raise ValueError(f"No rows found in the CSV file for '{self.identifier}'.")
        freeze_times = filtered_df.values[0][1:trial_column_end]
        threshold = filtered_df.values[0][threshold_column_idx]
        bout_duration = filtered_df.values[0][threshold_column_idx + 1]
        protocol = filtered_df.values[0][threshold_column_idx + 2]

        nwbfile.add_trial_column(name="percentage_of_time_spent_freezing", description="%time freezing")
        nwbfile.add_trial_column(
            name="threshold",
            description="The threshold for freezing detection. The threshold was set to 0 for all habituation trials. For conditioning and recall trials, a variable threshold was determined for each individual rat based on their activity or movement.",
        )
        nwbfile.add_trial_column(name="bout_duration", description="The duration of the tone in seconds.")
        nwbfile.add_trial_column(name="protocol", description="The protocol used for the trial.")

        for trial_index, (trial_start_time, trial_stop_time) in enumerate(zip(trial_start_times, trial_stop_times)):
            nwbfile.add_trial(
                start_time=trial_start_time,
                stop_time=trial_stop_time,
                percentage_of_time_spent_freezing=freeze_times[trial_index],
                threshold=threshold,
                bout_duration=bout_duration,
                protocol=protocol,
            )
