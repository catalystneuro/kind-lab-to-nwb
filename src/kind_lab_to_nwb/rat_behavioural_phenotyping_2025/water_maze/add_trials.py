import pandas as pd
from pynwb import NWBFile


def add_trials_to_nwbfile(
    nwbfile: NWBFile,
    trials: pd.DataFrame,
):

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
    for _, (name, description, *optional_type) in standard_columns.items():
        nwbfile.add_trial_column(name=name, description=description)

    # Add columns for quadrant times (in seconds)
    for quadrant in ["NE", "NW", "SW", "SE"]:
        nwbfile.add_trial_column(
            name=f"time_in_{quadrant}_quadrant", description=f"Time spent in {quadrant} quadrant in seconds"
        )
        nwbfile.add_trial_column(
            name=f"percent_time_in_{quadrant}_quadrant", description=f"Percentage of time spent in {quadrant} quadrant"
        )
        nwbfile.add_trial_column(
            name=f"time_to_platform_{quadrant}", description=f"Time to platform when in {quadrant} quadrant"
        )

    datetime_strings = trials["Date"] + " " + trials["Time"]
    video_timestamps = pd.to_datetime(datetime_strings, dayfirst=True)
    video_starting_times = [0.0]
    for i in range(1, len(video_timestamps)):
        video_starting_time = (video_timestamps.iloc[i] - video_timestamps.iloc[0]).total_seconds()
        video_starting_times.append(video_starting_time)

    image_series_names = list(nwbfile.acquisition.keys())
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

        # Add quadrant data - using column positions since they repeat names
        quadrants = ["NE", "NW", "SW", "SE"]
        # Time in quadrants (seconds) - columns 12-15
        for i, quadrant in enumerate(quadrants):
            trial_data[f"time_in_{quadrant}_quadrant"] = float(row.iloc[12 + i])

        # Percentage time in quadrants - columns 16-19
        for i, quadrant in enumerate(quadrants):
            trial_data[f"percent_time_in_{quadrant}_quadrant"] = float(row.iloc[16 + i])

        # Time to platform in each quadrant - columns 20-23
        for i, quadrant in enumerate(quadrants):
            trial_data[f"time_to_platform_{quadrant}"] = float(row.iloc[20 + i])

        # Add the trial with all its data
        nwbfile.add_trial(
            start_time=video_starting_times[trial_index],
            stop_time=video_starting_times[trial_index] + row["Trial duration"],
            **trial_data,
            timeseries=nwbfile.acquisition[image_series_names[trial_index]],
        )
