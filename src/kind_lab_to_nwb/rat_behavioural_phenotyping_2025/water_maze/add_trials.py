from pathlib import Path
from typing import Union

import pandas as pd
from pynwb import NWBFile


def add_trials_to_nwbfile(
    nwbfile: NWBFile,
    trials: pd.DataFrame,
):

    datetime_strings = trials["Date"] + " " + trials["Time"]
    trials["timestamps"] = pd.to_datetime(datetime_strings, dayfirst=True)
    trials["start_time"] = (trials["timestamps"] - trials["timestamps"].iloc[0]).dt.total_seconds()

    standard_columns = {
        "Platform": ("platform", "Platform location"),
        "Time to platform": ("time_to_platform", "Time taken to reach the platform in seconds"),
        "Trial duration": ("trial_duration", "Total duration of the trial in seconds"),
        "Distance travelled (cm)": ("distance_travelled_cm", "Distance travelled in centimeters"),
        "Average speed": ("average_speed", "Average swimming speed in cm/s"),
        "% time near walls": ("percent_time_near_walls", "Percentage of time spent near walls"),
        "Platform Quadrant": ("platform_quadrant", "Quadrant containing the platform (1=NE, 2=NW, 3=SW, 4=SE)", "int"),
    }

    # Add the standard columns to the NWB file
    for standard_column_name, (name, description, *_) in standard_columns.items():
        if standard_column_name in trials.columns:
            nwbfile.add_trial_column(name=name, description=description)

    # Determine the type of columns present in the data
    has_quadrant_data = "Quadrant time (sec)" in trials.columns
    # The zone time column name is sometimes different
    zone_time_column_name_substring = "Zone (% time)"
    zone_time_column_name = next((col for col in trials.columns if zone_time_column_name_substring in col), None)
    has_zone_data = zone_time_column_name is not None

    # Add columns for quadrant times (in seconds)
    quadrants = ["NE", "NW", "SW", "SE"]
    if has_quadrant_data:
        # Find the column index for "Quadrant time (sec)"
        quadrant_time_sec_idx = trials.columns.get_loc("Quadrant time (sec)")
        # Find the column index for "Quadrant time (%)"
        quadrant_time_percent_idx = trials.columns.get_loc("Quadrant time (%)")
        # Find the column index for "Time to platform"
        time_to_platform_idx = trials.columns.get_loc("Time to platform.1")

        for quadrant in quadrants:
            nwbfile.add_trial_column(
                name=f"time_in_{quadrant}_quadrant", description=f"Time spent in {quadrant} quadrant in seconds"
            )
            nwbfile.add_trial_column(
                name=f"percent_time_in_{quadrant}_quadrant",
                description=f"Percentage of time spent in {quadrant} quadrant",
            )
            nwbfile.add_trial_column(
                name=f"time_to_platform_{quadrant}", description=f"Time to platform when in {quadrant} quadrant"
            )

    # Add columns for zone data if present
    zones = ["NE", "NE_A", "SE", "SE_A", "SW", "SW_A", "NW", "NW_A"]
    if has_zone_data:
        zone_idx = trials.columns.get_loc(zone_time_column_name)

        for zone in zones:
            nwbfile.add_trial_column(
                name=f"zone_percent_time_{zone}", description=f"Percentage of time spent in {zone} zone"
            )

    # Add trials to the NWB file
    for trial_index, row in trials.reset_index(drop=True).iterrows():
        trial_data = {}

        # Add standard columns
        for csv_col, (nwb_col, _, *optional_type) in standard_columns.items():
            if csv_col in row.index:
                value = row[csv_col]
                # Convert to integer if specified
                if optional_type and optional_type[0] == "int":
                    value = int(float(value))  # Convert through float to handle potential decimal values
                trial_data[nwb_col] = value

        if has_quadrant_data:
            for i, quadrant in enumerate(quadrants):
                # Time in quadrants (seconds)
                trial_data[f"time_in_{quadrant}_quadrant"] = float(row.iloc[quadrant_time_sec_idx + i])

                # Percentage time in quadrants
                trial_data[f"percent_time_in_{quadrant}_quadrant"] = float(row.iloc[quadrant_time_percent_idx + i])

                # Time to platform in each quadrant
                trial_data[f"time_to_platform_{quadrant}"] = float(row.iloc[time_to_platform_idx + i])
        if has_zone_data:
            for i, zone in enumerate(zones):
                trial_data[f"zone_percent_time_{zone}"] = float(row.iloc[zone_idx + i])

        # Add the trial with all its data
        image_series_name = f"BehavioralVideoTrial{trial_index + 1}"
        assert image_series_name in nwbfile.acquisition, f"Image series '{image_series_name}' not found in NWB file."
        image_series = nwbfile.acquisition[image_series_name]

        nwbfile.add_trial(
            start_time=row["start_time"],
            stop_time=row["start_time"] + row["Trial duration"],
            **trial_data,
            timeseries=image_series,
        )
