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

import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('dark_background')

# Parameters
fs = 100  # Sampling frequency
T = 60  # Total duration in seconds
t = np.linspace(0, T, fs * T)  # Time vector
freq = 1  # Frequency of oscillation (Hz)
lag = 0.1  # Lag in seconds (for the second signal)

# Create signals
signal1 = np.sin(2 * np.pi * freq * t)
signal2 = np.sin(2 * np.pi * freq * t - 2 * np.pi * freq * lag)

# Initialize the figure
fig, ax = plt.subplots(figsize=(10, 4))

# Plot settings
ax.set_xlim(0, 1)  # Initial x-limits
ax.set_ylim(-1, 1)  # Fixed y-limits

# Lines for the signals
line1, = ax.plot([], [], label="Signal 1", color="dodgerblue")
line2, = ax.plot([], [], label="Signal 2", color="red")

# Circles for the signals (fixed positions)
circle1 = plt.Circle((1, 0), 1, color="white", fill=False)

ax.add_patch(circle1)

# Horizontal lines to follow the last point of signals
hline1, = ax.plot([], [], color="dodgerblue", alpha=0.8, linestyle="--")
hline2, = ax.plot([], [], color="red", alpha=0.8, linestyle="--")

t_hline, = ax.plot([], [], color="white", linestyle="--", alpha=0.2)
# Rotating hands
hand1, = ax.plot([], [], color="dodgerblue", linewidth=1.5)
hand2, = ax.plot([], [], color="red", linewidth=1.5)

angle_text = ax.text(0.93, 0.9, '', transform=ax.transAxes, fontsize=12, color="white")
ax.set_xticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' '])
# Add legend
# ax.legend(loc="upper right")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Time (s)')
ax.set_ylabel('signals apmlitude')


# Animation function
def animate(i):
    # Update the x-limits to follow the time point
    ax.set_xlim(t[i] - 3, t[i] + 2)

    # Update the signal lines
    line1.set_data(t[:i], signal1[:i])  # Shift signal1 upwards
    line2.set_data(t[:i], signal2[:i])  # Shift signal2 downwards

    # Update the rotating hands
    angle1 = 2 * np.pi * freq * t[i]
    angle2 = 2 * np.pi * freq * t[i] - 2 * np.pi * freq * lag

    hand1.set_data([1, 1 + np.cos(angle1)] + t[i],
                   [0, np.sin(angle1)])

    hand2.set_data([1, 1 + np.cos(angle2)] + t[i],
                   [0, np.sin(angle2)])
    # Update the horizontal lines
    hline1.set_data([t[i], 1 + np.cos(angle1) + t[i]], [signal1[i], signal1[i]])
    hline2.set_data([t[i], 1 + np.cos(angle2) + t[i]], [signal2[i], signal2[i]])
    # Calculate the angle in degrees between the two hands
    delta_angle = np.angle(np.exp(1j * (angle1 - angle2)), deg=True)

    # Update the text annotation
    angle_text.set_text(f"{delta_angle:.2f}°")
    circle1.set_center((t[i] + 1, 0))  # Modify this line to your desired movement
    t_hline.set_data([t[i], t[i]], [-1, 1])
    return line1, line2, hand1, hand2, hline1, hline2, angle_text, circle1, t_hline


# Create the animation
ani = FuncAnimation(fig, animate, frames=len(t), interval=30, blit=True)
plt.show()
