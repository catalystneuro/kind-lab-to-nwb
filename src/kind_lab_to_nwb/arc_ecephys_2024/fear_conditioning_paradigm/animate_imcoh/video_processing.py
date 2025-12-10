"""
    LFP/EEG Analysis Pipeline for Fear Conditioning Paradigm
    Copyright (C) 2023 Dr Paul Rignanese, Kind Lab, University of Edinburgh

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import json
import subprocess

import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

matplotlib.use('TkAgg')


def get_frame_timestamps(video_path):

    timestamps = []

    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
        # Run ffprobe to get frame-level information
        command = [
            'ffprobe', '-loglevel', 'quiet',
            '-show_frames', '-print_format', 'json',
            video_path
        ]

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Parse the JSON output
        if result.returncode == 0:
            probe_data = json.loads(result.stdout)
            frames = probe_data.get('frames', [])
            for frame in frames:
                if frame.get('media_type') == 'video':  # Filter for video frames
                    pts_time = frame.get('best_effort_timestamp_time', 'N/A')
                    if pts_time != 'N/A':
                        timestamps.append(float(pts_time))
        else:
            print("Error running ffprobe:", result.stderr.decode())

    return timestamps


# Step 2: Find frame corresponding to specific start time
def find_start_frame(timestamps, start_time_sec):
    return next(i for i, t in enumerate(timestamps) if t >= start_time_sec)


# Step 3: Display video using matplotlib
def display_video(video_path, start_frame):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fig, ax = plt.subplots()
    ret, frame = cap.read()
    im = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def update(frame_idx):
        ret, frame = cap.read()
        if ret:
            im.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return im,

    ani = FuncAnimation(fig, update, interval=50, blit=True)
    plt.axis('off')
    plt.show()

    cap.release()

# Main execution
# video_path = '/home/paul/Desktop/hdd/SGAP_ephys/Rat_698/Recall/Rat_698_recall.avi'
# timestamps = get_frame_timestamps(video_path)
# start_time_sec = 5 * 60  # 5 minutes and 40 seconds
# start_frame = find_start_frame(timestamps, start_time_sec)
#
# display_video(video_path, start_frame)
