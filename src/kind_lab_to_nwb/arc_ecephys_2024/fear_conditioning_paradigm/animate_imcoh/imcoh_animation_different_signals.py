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

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from scipy.signal import hilbert
from tqdm import tqdm

matplotlib.use('TkAgg')


# Polar Grid Class for dynamic updates
class PolarGrid:
    def __init__(self, ax, num_circles=3, max_radius=1, center=(0, 0), color='k'):
        self.ax = ax
        self.num_circles = num_circles
        self.max_radius = max_radius
        self.center = center
        self.color = color
        self.circles = []
        self.radial_lines = []
        self._draw_grid()

    def _draw_grid(self):
        x_center, y_center = self.center

        # Draw concentric circles
        for i in range(1, self.num_circles + 1):
            radius = (i / self.num_circles) * self.max_radius
            circle = plt.Circle((x_center, y_center), radius, color=self.color, fill=False, linestyle='--', alpha=0.5)
            self.ax.add_patch(circle)
            self.circles.append(circle)

        # Draw radial lines every 45 degrees
        angles = np.deg2rad(np.arange(0, 360, 45))
        for angle in angles:
            x_end = x_center + self.max_radius * np.cos(angle)
            y_end = y_center + self.max_radius * np.sin(angle)
            line, = self.ax.plot([x_center, x_end], [y_center, y_end], color=self.color, linestyle='--', alpha=0.5)
            self.radial_lines.append(line)

    def set_center(self, new_center):
        self.center = new_center
        x_center, y_center = new_center

        # Update circle centers
        for circle in self.circles:
            circle.set_center((x_center, y_center))

        # Update radial lines
        angles = np.deg2rad(np.arange(0, 360, 45))
        for line, angle in zip(self.radial_lines, angles):
            x_end = x_center + self.max_radius * np.cos(angle)
            y_end = y_center + self.max_radius * np.sin(angle)
            line.set_data([x_center, x_end], [y_center, y_end])


def animate_imcoh(raw_signal_1, raw_signal_2, signal_1, signal_2, color_area_1, color_area_2, background_color,
                  motion_trace,
                  motion_threshold, freezings, sample_rate, animal_id, cs, cs_sample_index,
                  video_data, video_path):
    if background_color == 'white':
        motion_threshold_color = 'orange'
        t_line_color = 'k'
        motion_line_color = 'k'
        polar_grid_color = 'k'
        angle_ax_lines_color = 'k'
    elif background_color == 'k':
        motion_threshold_color = 'orange'
        t_line_color = 'white'
        motion_line_color = 'white'
        plt.style.use('dark_background')
        polar_grid_color = 'white'
        angle_ax_lines_color = 'white'

    sns.set_context("notebook", font_scale=1.25)
    global video_frame_timestamps, frame_index

    signal_1 = np.asarray(signal_1) * 60
    signal_2 = np.asarray(signal_2) * 60

    video_start_sample = cs_sample_index - 40000
    video_start_time = video_start_sample/sample_rate
    video_frame_timestamps = video_data['frames_timestamps']
    # video_frame_timestamps = {k: v  for k, v in video_frame_timestamps.items()}

    closest_frame = min(video_frame_timestamps, key=lambda k: abs(video_frame_timestamps[k] - video_start_time))

    if len(signal_1) != len(signal_2):
        raise ValueError("The two signals must have the same length.")

    samples = np.arange(len(signal_1))

    analytic_signal1 = hilbert(signal_1)
    analytic_signal2 = hilbert(signal_2)

    amp1 = np.abs(analytic_signal1)  # Instantaneous amplitude of signal1
    amp2 = np.abs(analytic_signal2)  # Instantaneous amplitude of signal2

    phase1 = np.unwrap(np.angle(analytic_signal1))  # Instantaneous phase of signal1
    phase2 = np.unwrap(np.angle(analytic_signal2))  # Instantaneous phase of signal2

    cap = cv2.VideoCapture(video_path)
    frame_index = closest_frame  # Start at the first frame of the video
    #
    # plt.plot([i for i in list(video_data['frames_timestamps'].values())], video_data['blue_intensities'].values)
    # plt.axvline(video_data['delay'], color='white')
    #
    # plt.axvline(video_frame_timestamps[frame_index], color='g')
    #
    # plt.axvline(video_start_time, color='r')
    #
    # plt.plot(np.arange(0, len(signal1)) / sample_rate + video_start_time, signal1)
    # plt.plot(np.arange(0, len(signal1)) / sample_rate + video_start_time, signal2)

    print(closest_frame)
    # Move to the starting frame (frame 380) before the animation starts
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index-1) #opencv frames start at 1
    # Create the image object for displaying video frames in video_ax (initialize once)
    ret, frame = cap.read()  # Read the first synced frame to initialize
    if not ret:
        raise ValueError(f"Failed to read frame {frame_index} from the video.")

    fig = plt.figure(figsize=(20, 9))  # wider than tall for a horizontal layout
    # fig.set_constrained_layout(True)   # optionally use constrained layout (don’t combine with tight_layout)

    # Grid: 6 rows x 12 cols
    # - Left 8 columns: video (full height)
    # - Right 4 columns: top = filtered (3 cols) + angle (1 col); bottom = motion (all 4 cols)
    # shape: 6 rows x 12 cols
    filtered_traces_ax = plt.subplot2grid(shape=(10, 12), loc=(0, 0), rowspan=5, colspan=9)
    motion_ax = plt.subplot2grid(shape=(10, 12), loc=(5, 0), rowspan=5, colspan=9)
    angle_ax = plt.subplot2grid(shape=(10, 12), loc=(0, 7), rowspan=5, colspan=5, projection='polar')

    # video_ax = plt.subplot2grid(shape=(10, 12), loc=(5, 7), rowspan=5, colspan=5)

    # Now manually extend video_ax down to the bottom of the figure
    # pos = video_ax.get_position()  # [x0, y0, width, height] in figure coords
    # video_ax.set_position([pos.x0, 0.02, pos.width/1.14, pos.height/1.14 + pos.y0])  # flush bottom

    # Optional: also flush to the right edge (no right margin for video_ax)
    # video_ax.set_position([pos.x0, 0.0, 1.0 - pos.x0, pos.height + pos.y0])
    plt.tight_layout()  # use this only if you did NOT set constrained_layout=True on the figure

    video_ax = fig.add_axes([0.61, 0.025, 0.3, 0.47], label='video_ax')
    video_ax.set_in_layout(False)  # exclude from tight_layout/constrained_layout adjustments

    # Optional: also flush the video to the right edge of the figure
    # video_ax.set_position([video_left, video_bottom, 1.0 - video_left, video_height])

    # plt.show()
    start_angle = np.deg2rad(45)  # 90 degrees
    end_angle = np.deg2rad(315)  # 0 degrees

    # Create a curved arrow using FancyArrowPatch
    arrow = FancyArrowPatch(
        (start_angle, 0.5),  # Start point (angle, radius)
        (end_angle, 0.5),  # End point (angle, radius)
        arrowstyle='->,head_width=5,head_length=5',
        color='grey',
        lw=3,
        # linestyle='--',
        # alpha=0.5,
        connectionstyle="arc3,rad=-0.5",  # Curvature of the arrow
        transform=angle_ax.transData,  # Use data coordinates
    )

    # Add the arrow to the plot
    angle_ax.add_patch(arrow)

    angle_ax.set_xticks([])
    angle_ax.set_yticks([])

    video_ax.axis('off')

    phase_diff_arrow = angle_ax.annotate('', xy=(0, 1), xytext=(0, 0),
                                         arrowprops=dict(facecolor=color_area_2, edgecolor=color_area_2, width=3,
                                                         headwidth=10))

    angle_ax.annotate('', xy=(0, 1), xytext=(0, 0),
                      arrowprops=dict(facecolor=color_area_1, edgecolor=color_area_1, width=3, headwidth=10))

    angle_ax.set_ylim(0, 1)  # Set the radius limit
    angle_ax.set_yticklabels([])  # Hide radial labels
    angle_ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))  # Set angle ticks every 45°
    angle_ax.grid(color=angle_ax_lines_color, linestyle='--', alpha=0.3)

    filtered_traces_ax.set_xlim(0, 1000)  # Initial x-limits (in samples)
    filtered_traces_ax.set_ylim(-2000, 2000)  # Fixed y-limits

    motion_ax.set_ylim(0, 0.01)  # Fixed y-limits

    filtered_line_1, = filtered_traces_ax.plot([], [], label="Signal 1", color=color_area_1, linewidth=2)
    filtered_line_2, = filtered_traces_ax.plot([], [], label="Signal 2", color=color_area_2, linewidth=2)
    t_line_filtered_ax, = filtered_traces_ax.plot([], [], color=t_line_color, linestyle="--", alpha=0.5)

    # raw_line_1, = motion_ax.plot([], [], label="Signal 1", color="orange", alpha=0.7)
    # raw_line_2, = motion_ax.plot([], [], label="Signal 2", color="purple", alpha=0.7)
    motion_line, = motion_ax.plot([], [], label="Motion", color=motion_line_color, linewidth=3)

    t_line_raw_ax, = motion_ax.plot([], [], color=t_line_color, linestyle="--", alpha=0.5)

    polar_grid = PolarGrid(filtered_traces_ax, num_circles=3, max_radius=2000, center=(4000, 0), color=polar_grid_color)

    filtered_hline_1, = filtered_traces_ax.plot([], [], color=color_area_1, alpha=0.8, linestyle="--", linewidth=3)
    filtered_hline_2, = filtered_traces_ax.plot([], [], color=color_area_2, alpha=0.8, linestyle="--", linewidth=3)

    filtered_hand_1, = filtered_traces_ax.plot([], [], color=color_area_1, linewidth=3)
    filtered_hand_2, = filtered_traces_ax.plot([], [], color=color_area_2, linewidth=3)

    # filtered_hand_1, = filtered_traces_ax.plot([], [], color="orange", linewidth=1.5)
    # filtered_hand_2, = filtered_traces_ax.plot([], [], color="purple", linewidth=1.5)

    for k, (onset, offset) in freezings.iterrows():
        filtered_traces_ax.axvspan(onset, offset, alpha=0.2, color='yellow')
        motion_ax.axvspan(onset, offset, alpha=0.2, color='yellow')

        xc = (onset + offset) / 2.0
        motion_ax.text(xc, 0.009, 'Freezing',
                       color='orange', fontsize=20,
                       ha='center', va='top', clip_on=True)

    filtered_traces_ax.spines['top'].set_visible(False)
    filtered_traces_ax.spines['right'].set_visible(False)
    motion_ax.spines['top'].set_visible(False)
    motion_ax.spines['right'].set_visible(False)
    motion_ax.set_xlabel('time to CS # {} onset (s)'.format(cs))
    filtered_traces_ax.set_xlabel('')
    motion_ax.axhline(motion_threshold, color=motion_threshold_color, linestyle='--', linewidth=3)
    zero_point= 40000
    tick_positions = np.arange(zero_point - 20 * sample_rate, zero_point + 21 * sample_rate, sample_rate)

    tick_labels = np.arange(-20, 21, 1)  # From -20s to +20s

    motion_ax.set_xticks(tick_positions)
    motion_ax.set_xticklabels(tick_labels)

    filtered_traces_ax.set_xticks(tick_positions)
    filtered_traces_ax.set_xticklabels([])
    filtered_traces_ax.set_yticks([-2000, 0, 2000])
    filtered_traces_ax.set_yticklabels([-2, 0, 2])

    filtered_traces_ax.set_ylabel(u'\u03B4 (3-6Hz) LFP (A.U)')
    motion_ax.set_ylabel(u'Motion (A.U)')

    motion_ax.set_yticks([0, 0.005, 0.01])
    motion_ax.set_yticklabels([0, 0.5, 1])

    hand_1_tip_points = []
    hand_2_tip_points = []

    im = video_ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))  # Use an empty frame as a placeholder initially

    plt.figtext(0.51, 0.94, "mPFC", fontsize=20, color=color_area_1, ha='left', fontweight='bold')
    plt.figtext(0.59, 0.94, "Amyg", fontsize=20, color=color_area_2, ha='left', fontweight='bold')

    # freeze_windows = freezings.to_numpy()
    # freeze_txt = motion_ax.text(0.72, 0.8, '', transform=motion_ax.transAxes,
    #                             color='orange', fontsize=20,
    #                             ha='center', va='top')

    def wrap_phase(phase):
        return (phase + 180) % 360 - 180

    def animate(animation_frame_index):
        global video_frame_timestamps, frame_index

        i = animation_frame_index  # Adjusting 'i' to map correctly to the data
        if i <= 80000:

            current_time = (i / sample_rate) + video_start_time  # Current time in seconds for animation

            xcur = samples[i]

            # if any((s <= xcur <= e) for s, e in freeze_windows):
            #     freeze_txt.set_text('Freezing')
            # else:
            #     freeze_txt.set_text('')

            if current_time >= video_frame_timestamps[frame_index]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)  # opencv frames start at 1

                ret, frame = cap.read()  # Get the next frame from the video

                brightness_factor = 50  # Adjust this value to increase/decrease brightness
                frame_bright = cv2.convertScaleAbs(frame, alpha=1, beta=brightness_factor)

                frame_rgb = cv2.cvtColor(frame_bright, cv2.COLOR_BGR2RGB)
                im.set_data(frame_rgb)
                frame_index += 1  # Increment the frame index for the video

            filtered_traces_ax.set_xlim(i - 5 * 2000, i + 6000)  # Scroll through the signal
            motion_ax.set_xlim(i - 5 * 2000, i + 6000)  # Scroll through the signal

            filtered_line_1.set_data(samples[:i], signal_1[:i])
            filtered_line_2.set_data(samples[:i], signal_2[:i])
            motion_line.set_data(samples[:i], motion_trace[:i])
            # raw_line_1.set_data(samples[:i], raw_signal_1[:i])
            # raw_line_2.set_data(samples[:i], raw_signal_2[:i])

            t_line_filtered_ax.set_data([samples[i], samples[i]], [-4000, 4000])
            t_line_raw_ax.set_data([samples[i], samples[i]], [-4000, 4000])

            polar_center = (samples[i] + 2000, 0)
            polar_grid.set_center(polar_center)

            angle1 = phase1[i]  # Instantaneous phase of signal1
            angle2 = phase2[i]  # Instantaneous phase of signal2

            hand1_x_tip = polar_center[0] + amp1[i] * np.sin(angle1)
            hand1_y_tip = amp1[i] * np.cos(angle1)

            hand2_x_tip = polar_center[0] + amp2[i] * np.sin(angle2)
            hand2_y_tip = amp2[i] * np.cos(angle2)

            filtered_hand_1.set_data([polar_center[0], polar_center[0] + amp1[i] * np.sin(angle1)],
                           [polar_center[1], amp1[i] * np.cos(angle1)])
            filtered_hand_2.set_data([polar_center[0], polar_center[0] + amp2[i] * np.sin(angle2)],
                           [polar_center[1], amp2[i] * np.cos(angle2)])

            filtered_hline_1.set_data([samples[i], polar_center[0] + amp1[i] * np.sin(angle1)],
                                      [signal_1[i], signal_1[i]])
            filtered_hline_2.set_data([samples[i], polar_center[0] + amp2[i] * np.sin(angle2)],
                                      [signal_2[i], signal_2[i]])

            phase_diff = wrap_phase(np.rad2deg(angle1 - angle2))

            phase_diff_rad = np.deg2rad(phase_diff)  # Convert to radians
            phase_diff_arrow.set_position((0, 0))  # Start at center
            phase_diff_arrow.xy = (phase_diff_rad, 1)  # Arrow points to phase difference at radius = 1

            new_point1, = filtered_traces_ax.plot(hand1_x_tip, hand1_y_tip, color=color_area_1, marker='o',
                                                  markersize=2,
                                                  alpha=1.0)
            hand_1_tip_points.append(new_point1)

            new_point2, = filtered_traces_ax.plot(hand2_x_tip, hand2_y_tip, color=color_area_2, marker='o',
                                                  markersize=2,
                                                  alpha=1.0)
            hand_2_tip_points.append(new_point2)

            for idx, point in enumerate(hand_1_tip_points[:-1]):
                point.set_alpha(max(0, 1 - (len(hand_1_tip_points) - idx) / 20))

            for idx, point in enumerate(hand_2_tip_points[:-1]):
                point.set_alpha(max(0, 1 - (len(hand_2_tip_points) - idx) / 20))

            if len(hand_1_tip_points) > 20:
                old_point1 = hand_1_tip_points.pop(0)
                old_point1.remove()

            if len(hand_2_tip_points) > 20:
                old_point2 = hand_2_tip_points.pop(0)
                old_point2.remove()

        return (filtered_line_1, filtered_line_2, motion_line, t_line_filtered_ax, t_line_raw_ax,
                *polar_grid.circles, *polar_grid.radial_lines,
                filtered_hand_1, filtered_hand_2, filtered_hline_1, filtered_hline_2, phase_diff_arrow,
                *hand_1_tip_points, *hand_2_tip_points, im)

    ani = FuncAnimation(fig, animate, frames=tqdm(range(0, len(samples), 5), desc='Rendering Frames'), interval=1,
                        blit=False)

    plt.tight_layout()
    ani.save('{}_cs_{}_imcoh_animation_{}.mp4'.format(animal_id, cs, background_color), writer='ffmpeg', fps=60)
    cap.release()
    # plt.show()
