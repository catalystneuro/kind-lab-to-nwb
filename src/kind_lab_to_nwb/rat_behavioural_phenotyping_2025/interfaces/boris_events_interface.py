"""Custom DataInterface for BORIS behavioral scoring data for marble interaction task."""

import numpy as np
from typing import List, Dict, Tuple
import json
from datetime import datetime
from ndx_events import EventsTable, MeaningsTable, CategoricalVectorData, TimestampVectorData
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.utils import DeepDict
from pydantic import FilePath


def get_observation_ids(file_path: FilePath) -> List[str]:
    with open(file_path, "r") as f:
        boris_output = json.load(f)
    observation_ids = [obs_key for obs_key in boris_output["observations"].keys()]
    return np.unique(observation_ids)


def get_matching_observation_id(file_path: FilePath, pattern: str) -> Dict:
    observation_ids = get_observation_ids(file_path=file_path)
    matching_observation_ids = [obs_id for obs_id in observation_ids if pattern in obs_id]
    if matching_observation_ids > 1:
        raise ValueError(f"Multiple observation_ids found for pattern {pattern}. Found {matching_observation_ids}")
    elif len(matching_observation_ids) == 0:
        raise ValueError(f"No observation_id found for pattern {pattern}")
    else:
        return matching_observation_ids[0]


class BORISBehavioralEventsInterface(BaseDataInterface):
    """Adds data for annotated behavioral data."""

    keywords = ["behavior", "behavioral events"]

    def __init__(self, file_path: FilePath, observation_id: float, verbose: bool = False):
        # This should load the data lazily and prepare variables you need

        self.file_path = file_path
        self.verbose = verbose
        observation_ids = get_observation_ids(file_path)
        if observation_id not in observation_ids:
            raise ValueError(
                f"observation_id {observation_id} not found in {file_path}."
                f"{file_path} contains the following observation_ids: {observation_ids}"
            )
        else:
            self.observation_id = observation_id
        self._starting_time = None
        self._data = None

    def get_metadata(self) -> DeepDict:
        metadata = super().get_metadata()

        session_start_time = self._get_session_start_time()
        if session_start_time is not None:
            metadata["NWBFile"].update(session_start_time=session_start_time)

        return metadata

    def get_event_types(self) -> List[Dict]:
        """Get behavioral event type.

        Returns
        -------
        list
            List of dictionaries containing the type of behvaioral event stored in the BORIS output file
        """
        with open(self.file_path, "r") as f:
            boris_output = json.load(f)
        event_types = []
        event_description = []
        for idx, event in boris_output["behaviors_conf"].items():
            event_types.append(event["code"])
            event_description.append(event["description"])
        return event_types, event_description

    def is_state_event(self, event_type: str) -> bool:
        with open(self.file_path, "r") as f:
            boris_output = json.load(f)
        for idx, event in boris_output["behaviors_conf"].items():
            if event["code"] == event_type and event["type"] == "State event":
                return True
            elif event["code"] == event_type and event["type"] == "Point event":
                return False
            else:
                raise ValueError(f"Could not determine is the event {event_type} is a state or point event.")

    def _get_data_from_observation_id(self) -> List[Dict]:

        with open(self.file_path, "r") as f:
            boris_output = json.load(f)
        self._data = boris_output["observations"][self.observation_id]
        return self._data

    def get_starting_time(self) -> float:
        if self._data is None:
            self._data = self._get_data_from_observation_id()
        self._starting_time = self._data["time offset"]

    def get_timestamps(self, event_type: str) -> Tuple[List[float], List[float]]:
        # Extract start and stop times of the events
        if self._data is None:
            self._data = self._get_data_from_observation_id()
        events = self._data["events"]
        event_types, event_descriptions = self.get_event_types()
        if event_type not in event_types:
            raise ValueError(
                f"Event type {event_type} not found in the BORIS output file. "
                f"Available event types are {event_types}"
            )
        if self._starting_time is None:
            self.get_starting_time()

        if self.is_state_event(event_type):  # state event
            timestamps = [event[0] + self._starting_time for event in events if event[2] == event_type]
            # the timestamps with even indexes will be the start of the event and the timestamps with odd indexes will be the end of the event
            # save event_timestamps as a list of the the timestamps of the start of the event
            event_timestamps = timestamps[::2]
            # save event_durations as a list of the durations of the events by subtracting the start of the event from the end of the event
            event_durations = [timestamps[n + 1] - timestamp for n, timestamp in enumerate(timestamps) if n % 2 == 0]
        else:  # point event
            event_timestamps = [event[0] + self._starting_time for event in events if event[2] == event_type]
            event_durations = [np.nan for event in events if event[2] == event_type]

        return event_timestamps, event_durations

    def _get_session_start_time(self) -> float:
        """Get the start time of the session.

        Returns
        -------
        float
            Start time of the session
        """
        if self._data is None:
            self._data = self._get_data_from_observation_id()
        session_start_time = self._data["date"]
        session_start_time = datetime.strptime(session_start_time, "%Y-%m-%dT%H:%M:%S")
        return session_start_time

    def get_event_frame_index(self, event_type: str) -> List[int]:
        # Extract frame index of the events
        if self._data is None:
            self._data = self._get_data_from_observation_id()
        events = self._data["events"]
        frame_index = [event[3] for event in events if event[2] == event_type]
        return frame_index

    def set_aligned_starting_time(self, aligned_start_time) -> None:
        self._starting_time = aligned_start_time

    def get_behavioral_video_fps(self) -> float:
        if self._data is None:
            self._data = self._get_data_from_observation_id()
        for video_filepath, fps in self._data["media_info"]["fps"].items():
            return float(fps)  # TODO check if fps is the same for all videos

    def add_to_nwbfile(self, nwbfile, metadata, **conversion_options):
        event_type_meanings_table = MeaningsTable(
            name="EventTypeMeanings", description="Meanings of behavioral event types"
        )
        event_types, event_descriptions = self.get_event_types()
        for event_type, event_description in zip(event_types, event_descriptions):
            event_type_meanings_table.add_row(value=event_type, meaning=event_description)
        event_type_column = CategoricalVectorData(
            name="behavioral_event", description="Type of behavioral event", meanings=event_type_meanings_table
        )
        # Add behavioral events to nwbfile
        behavioral_events = EventsTable(
            name="BehavioralEvents",
            description="Behavioral events from BORIS output",
            columns=[event_type_column],
            meanings_tables=[event_type_meanings_table],
        )

        for event_type in event_types:
            timestamps, durations = self.get_timestamps(event_type=event_type)
            for timestamp, duration in zip(timestamps, durations):
                behavioral_events.add_row(timestamp=timestamp, duration=duration, behavioral_event=event_type)

        behavioral_events.timestamp.resolution = 1 / self.get_behavioral_video_fps()
        behavioral_events.duration.resolution = 1 / self.get_behavioral_video_fps()

        nwbfile.add_events_table(behavioral_events)

        return nwbfile
