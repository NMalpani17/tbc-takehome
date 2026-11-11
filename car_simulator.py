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


class CarSimulator():
    def __init__(self, wheelbase, v0, theta0):
        # INPUTS:
        # the wheel base is the distance between the front and the rear wheels
        self.wheelbase = wheelbase
        self.x = 0
        self.y = 0
        self.v = v0
        self.theta = theta0

    def simulatorStep(self, a, wheel_angle, dt):
        self.x = None  # TBD
        self.y = None  # TBD
        self.v = None  # TBD
        self.theta = None  # TBD (HINT: "wheelbase" needs to be used here)


def main():
     wheelbase = 4  # arbitrary 4m wheelbase
     v0 = 0
     theta0 = 0
     simulator = CarSimulator(wheelbase, v0, theta0)
     dt = 0.1  # arbitrarily set the time step to 0.1 s

main()
