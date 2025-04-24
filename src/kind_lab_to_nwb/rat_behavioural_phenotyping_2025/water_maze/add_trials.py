import pandas as pd
from pynwb import NWBFile


def add_trials_to_nwbfile(
    nwbfile: NWBFile,
    trials: pd.DataFrame,
):

    column_name_mapping = {
        "Platform": "platform",
        "Time to platform": "time_to_platform",
        "Distance travelled (cm)": "distance_travelled_in_cm",
        "Trial duration": "trial_duration",
        "Average speed": "average_speed",
        "% time near walls": "percentage_time_near_walls",
    }
    nwbfile.add_trial_column(name="platform", description="The location of the platform in the maze.")
    nwbfile.add_trial_column(
        name="time_to_platform", description="The duration of the swim to the platform in seconds."
    )
    nwbfile.add_trial_column(name="distance_travelled_in_cm", description="The distance travelled in cm.")
    nwbfile.add_trial_column(name="trial_duration", description="The duration of the trial in seconds.")
    nwbfile.add_trial_column(name="average_speed", description="The average speed in cm/s.")
    nwbfile.add_trial_column(name="percentage_time_near_walls", description="% time near walls")
    #    todo : add more columns
    nwbfile.add_trial_column(name="image_series", description="The reference for the video file.")

    datetime_strings = trials["Date"] + " " + trials["Time"]
    video_timestamps = pd.to_datetime(datetime_strings, dayfirst=True)
    video_starting_times = [0.0]
    for i in range(1, len(video_timestamps)):
        video_starting_time = (video_timestamps.iloc[i] - video_timestamps.iloc[0]).total_seconds()
        video_starting_times.append(video_starting_time)

    image_series_names = list(nwbfile.acquisition.keys())
    for trial_index, row in trials.reset_index(drop=True).iterrows():
        trial_data = {column_name_mapping[col]: row[col] for col in column_name_mapping.keys() if col in row.index}

        nwbfile.add_trial(
            start_time=video_starting_times[trial_index],
            stop_time=video_starting_times[trial_index] + row["Trial duration"],
            **trial_data,
            image_series=nwbfile.acquisition[image_series_names[trial_index]],  # TBD
        )
