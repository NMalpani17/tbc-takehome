# The class CarSimulator is a simple 2D vehicle simulator.
# The vehicle states are:
# - x: is the position on the x axis on a xy plane
# - y: is the position on the y axis on a xy plane
# - v is the vehicle speed in the direction of travel of the vehicle
# - theta: is the angle wrt the x axis (0 rad means the vehicle
#   is parallel to the x axis, in the positive direction;
#   pi/2 rad means the vehicle is parallel
#   to the y axis, in the positive direction)
# - NOTE: all units are SI: meters (m) for distances, seconds (s) for
#   time, radians (rad) for angles...
#
# (1)
# Write the method "simulatorStep", which should update
# the vehicle states, given 3 inputs:
#  - a: commanded vehicle acceleration
#  - wheel_angle: steering angle, measured at the wheels;
#    0 rad means that the wheels are "straight" wrt the vehicle.
#    A positive value means that the vehicle is turning counterclockwise
#  - dt: duration of time after which we want to provide
#    a state update (time step)
#
# (2)
# Complete the function "main". This function should run the following simulation:
# - The vehicle starts at 0 m/s
# - The vehicle drives on a straight line and accelerates from 0 m/s to 10 m/s
#   at a constant rate of 0.4 m/s^2, then it proceeds at constant speed.
# - Once reached the speed of 10 m/s, the vehicle drives in a clockwise circle of
#   roughly 100 m radius
# - the simulation ends at 100 s
#
# (3)
# - plot the vehicle's trajectory on the xy plane
# - plot the longitudinal and lateral accelerations over time

"""Kinematic bicycle-model car simulator.

Model: kinematic bicycle model with the rear axle as the reference point,
integrated with explicit (forward) Euler:

    x'     = v * cos(theta)
    y'     = v * sin(theta)
    theta' = v / wheelbase * tan(wheel_angle)
    v'     = a

For a constant wheel angle the rear axle follows a circle of radius
wheelbase / tan(wheel_angle), so a clockwise 100 m turn needs
wheel_angle = -atan(wheelbase / 100 m). The lateral acceleration is
v * theta' = v^2 * tan(wheel_angle) / wheelbase (positive = to the left).

With constant speed and wheel angle, every Euler step has the same length
and heading change, so the turn is traced as a regular polygon (1 m sides
at 10 m/s, dt = 0.1 s). Its vertices lie on a circle of radius
v * dt / (2 * sin(theta' * dt / 2)) ~= 100.0004 m, offset by half a step
from the continuous solution, so the "roughly 100 m" circle closes on itself.

Run with: python car_simulator.py
Plots are saved to plots/trajectory.png and plots/accelerations.png.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

WHEELBASE_M = 4.0
INITIAL_SPEED_MPS = 0.0
INITIAL_HEADING_RAD = 0.0
TIME_STEP_S = 0.1
SIMULATION_DURATION_S = 100.0

ACCELERATION_MPS2 = 0.4
TARGET_SPEED_MPS = 10.0
TURN_RADIUS_M = 100.0
# Floating-point slack so a speed of 9.999999999 m/s counts as "reached".
SPEED_TOLERANCE_MPS = 1e-9

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
FIGURE_DPI = 150
LINE_WIDTH = 2.0
MARKER_SIZE = 9
GRID_ALPHA = 0.3
TRAJECTORY_COLOR = "#2a78d6"
START_COLOR = "#1baf7a"
TURN_START_COLOR = "#eb6834"
LONGITUDINAL_COLOR = "#2a78d6"
LATERAL_COLOR = "#eb6834"


class CarSimulator:
    """2D vehicle state (x, y [m], v [m/s], theta [rad]) at the rear axle."""

    def __init__(self, wheelbase: float, v0: float, theta0: float) -> None:
        """Start at the origin with speed v0 [m/s] and heading theta0 [rad].

        wheelbase is the distance between the front and rear axles [m].
        """
        if wheelbase <= 0:
            raise ValueError(f"wheelbase must be positive, got {wheelbase}")
        self.wheelbase = wheelbase
        self.x = 0.0
        self.y = 0.0
        self.v = v0
        self.theta = theta0

    # Name kept from the assignment spec (not PEP 8 snake_case).
    def simulatorStep(self, a: float, wheel_angle: float, dt: float) -> None:
        """Advance the state by dt [s] with one explicit Euler step.

        a is the commanded acceleration [m/s^2]; wheel_angle is the steering
        angle at the wheels [rad], positive turning counterclockwise.
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        # Every derivative is taken from the current state, so the updates
        # below do not depend on the order in which they are applied.
        x_rate = self.v * math.cos(self.theta)
        y_rate = self.v * math.sin(self.theta)
        theta_rate = self.v / self.wheelbase * math.tan(wheel_angle)

        self.x += x_rate * dt
        self.y += y_rate * dt
        self.theta += theta_rate * dt
        self.v += a * dt


@dataclass(frozen=True)
class SimulationLog:
    """Time series of one run; sample k holds the state at time_s[k] and
    the commands applied from that state."""

    time_s: FloatArray
    x_m: FloatArray
    y_m: FloatArray
    speed_mps: FloatArray
    heading_rad: FloatArray
    wheel_angle_rad: FloatArray
    longitudinal_accel_mps2: FloatArray
    lateral_accel_mps2: FloatArray


def lateral_acceleration(
    speed: FloatArray, wheel_angle: FloatArray, wheelbase: float
) -> FloatArray:
    """Lateral acceleration [m/s^2] of the bicycle model, positive to the left."""
    return speed**2 * np.tan(wheel_angle) / wheelbase


def compute_controls(speed: float, wheelbase: float, dt: float) -> tuple[float, float]:
    """Return (acceleration [m/s^2], wheel_angle [rad]) for the current speed.

    Accelerate in a straight line until TARGET_SPEED_MPS, then hold that
    speed in a constant clockwise turn of TURN_RADIUS_M.
    """
    if speed < TARGET_SPEED_MPS - SPEED_TOLERANCE_MPS:
        # Shrink the final step's acceleration so the speed lands exactly on
        # the target instead of overshooting it.
        acceleration = min(ACCELERATION_MPS2, (TARGET_SPEED_MPS - speed) / dt)
        return acceleration, 0.0
    # Negative wheel angle turns clockwise.
    return 0.0, -math.atan(wheelbase / TURN_RADIUS_M)


def run_simulation(
    simulator: CarSimulator, dt: float, duration: float
) -> SimulationLog:
    """Run the controller and simulator from t = 0 to duration [s] in steps of dt [s].

    The number of steps is fixed up front so floating-point error in the
    time does not add or drop a step.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")

    num_steps = round(duration / dt)
    time_s = np.arange(num_steps + 1) * dt
    x_m = np.empty_like(time_s)
    y_m = np.empty_like(time_s)
    speed_mps = np.empty_like(time_s)
    heading_rad = np.empty_like(time_s)
    wheel_angle_rad = np.empty_like(time_s)
    acceleration_mps2 = np.empty_like(time_s)

    for step in range(num_steps + 1):
        acceleration, wheel_angle = compute_controls(simulator.v, simulator.wheelbase, dt)
        x_m[step] = simulator.x
        y_m[step] = simulator.y
        speed_mps[step] = simulator.v
        heading_rad[step] = simulator.theta
        wheel_angle_rad[step] = wheel_angle
        acceleration_mps2[step] = acceleration
        if step < num_steps:
            simulator.simulatorStep(acceleration, wheel_angle, dt)

    return SimulationLog(
        time_s=time_s,
        x_m=x_m,
        y_m=y_m,
        speed_mps=speed_mps,
        heading_rad=heading_rad,
        wheel_angle_rad=wheel_angle_rad,
        longitudinal_accel_mps2=acceleration_mps2,
        lateral_accel_mps2=lateral_acceleration(
            speed_mps, wheel_angle_rad, simulator.wheelbase
        ),
    )


def find_turn_start_index(log: SimulationLog) -> int:
    """Index of the first sample with a non-zero wheel angle."""
    turning_indices = np.flatnonzero(log.wheel_angle_rad != 0.0)
    if turning_indices.size == 0:
        raise ValueError("The vehicle never started turning.")
    return int(turning_indices[0])


def plot_trajectory(log: SimulationLog, turn_start_index: int) -> Figure:
    """Plot the xy path with the start and turn-start points marked."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(log.x_m, log.y_m, color=TRAJECTORY_COLOR, linewidth=LINE_WIDTH,
            label="Rear-axle path")
    ax.plot(log.x_m[0], log.y_m[0], linestyle="none", marker="o",
            markersize=MARKER_SIZE, color=START_COLOR,
            label=f"Start (t = {log.time_s[0]:.1f} s)")
    ax.plot(log.x_m[turn_start_index], log.y_m[turn_start_index],
            linestyle="none", marker="D", markersize=MARKER_SIZE,
            color=TURN_START_COLOR,
            label=f"Turn begins (t = {log.time_s[turn_start_index]:.1f} s)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Vehicle trajectory")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_accelerations(log: SimulationLog) -> Figure:
    """Plot longitudinal and lateral acceleration against time."""
    fig, (ax_long, ax_lat) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    ax_long.plot(log.time_s, log.longitudinal_accel_mps2, color=LONGITUDINAL_COLOR,
                 linewidth=LINE_WIDTH, label="Longitudinal")
    ax_long.set_ylabel("Acceleration [m/s²]")
    ax_long.set_title("Longitudinal acceleration")

    ax_lat.plot(log.time_s, log.lateral_accel_mps2, color=LATERAL_COLOR,
                linewidth=LINE_WIDTH, label="Lateral (positive = left)")
    ax_lat.set_ylabel("Acceleration [m/s²]")
    ax_lat.set_title("Lateral acceleration")
    ax_lat.set_xlabel("Time [s]")

    for ax in (ax_long, ax_lat):
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend()
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, filename: str) -> None:
    """Save a figure as a PNG in PLOTS_DIR, creating the folder if needed."""
    PLOTS_DIR.mkdir(exist_ok=True)
    fig.savefig(PLOTS_DIR / filename, dpi=FIGURE_DPI, bbox_inches="tight")


def print_summary(log: SimulationLog, turn_start_index: int) -> None:
    """Print when and where the target speed is reached, and the cornering load."""
    print(
        f"Reached {TARGET_SPEED_MPS:.1f} m/s at t = {log.time_s[turn_start_index]:.1f} s, "
        f"position (x, y) = ({log.x_m[turn_start_index]:.2f}, "
        f"{log.y_m[turn_start_index]:.2f}) m"
    )
    print(
        f"Steady-state lateral acceleration: {log.lateral_accel_mps2[-1]:.3f} m/s² "
        "(negative = to the right)"
    )


def main() -> None:
    simulator = CarSimulator(WHEELBASE_M, INITIAL_SPEED_MPS, INITIAL_HEADING_RAD)
    log = run_simulation(simulator, TIME_STEP_S, SIMULATION_DURATION_S)
    turn_start_index = find_turn_start_index(log)

    print_summary(log, turn_start_index)
    save_figure(plot_trajectory(log, turn_start_index), "trajectory.png")
    save_figure(plot_accelerations(log), "accelerations.png")
    plt.show()


if __name__ == "__main__":
    main()
