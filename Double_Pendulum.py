# This should simulate the motion of a double pendulum using the
# 4th order Runge Kutta Method
# also known as RK4
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sin, cos, radians
import numpy as np
from numpy.linalg import inv
from time import time
from collections import deque

# This represents the equations of motion for the two masses on a double pendulum
# The equations have been derived using the Lagrangian method, and put into matrices such that
# Y*I = F => Y = inv(I)*F, where
# Y is a vector containing the accelerations for each mass 1 and mass 2
# I is a matrix representing the left side of the eqns of motion with the  angular accelerations factored out
# F is a vector containing the right side of the eqns of motion not containing an angular acceleration term
# All angles used in this code are taken from the down-vertical position of the pendulum, or a pendulum at rest

def G(y, t):

    omega1 = y[0]
    theta1 = y[1]
    omega2 = y[2]
    theta2 = y[3]
    i11, i12 = (m1 + m2) * L1, m2 * L2 * cos(theta1 - theta2)
    i21, i22 = L1 * cos(theta1 - theta2), L2
    I = np.array([[i11, i12],[i21, i22]])
    F1 = -m2 * L2 * (omega2**2) * sin(theta1 - theta2) - (m1 + m2) * g * sin(theta1)
    F2 = L1 * (omega1**2) * sin(theta1 - theta2) - g * sin(theta2)
    F = np.array([F1, F2])

    acceleration = inv(I).dot(F)

    return np.array([acceleration[0], omega1, acceleration[1], omega2])

# This function does the RK4 step, using time step delt, aka h in many texts
# This works by finding the slope of a diffEQ at 4 different time steps within a step delt, then averaging it,
# which is what each k represents
def RK4_step(y, time, delt):

    k1 = G(y, time)
    k2 = G(y + 0.5 * k1 * delt, time + 0.5 * delt)
    k3 = G(y + 0.5 * k2 * delt, time + 0.5 * delt)
    k4 = G(y + delt * k3, time + delt)
    return (delt / 6) * (k1 + 2*k2 + 2*k3 + k4)

# This simply calculates the x, y coordinates from the two angular positions, theta1 and theta2
def position(theta1, theta2):

    origin = [0, 0]

    x1 = np.array([origin[0], L1 * sin(theta1)])
    y1 = np.array([origin[1], -L1 * cos(theta1)])
    x2 = np.array([x1[1], x1[1] + (L2 * sin(theta2))])
    y2 = np.array([y1[1], y1[1] - (L2 * cos(theta2))])
    #print(x1, y1, x2, y2)
    return x1, y1, x2, y2


# physical variables and constants
# m1, m2 represent the point masses in kg
# L1, L2 represent the lengths of each arm in meters
# g is the gravitational constant
m1 = 1
m2 = 1
L1 = 1
L2 = 1
g = 9.81

delt = 0.025    # delt, as mentioned above in RK4 step, is the length of each time step in seconds,
                # or in our simulation, each point plotted
omega1_0 = 0    # initial angular velocity of mass 1 in rad/sec
theta1_0 = np.random.randint(360)   # picks a random initial angular between 0 and 360º
omega2_0 = 0    # initial angular velocity of mass 2 in rad/sec
theta2_0 = np.random.randint(90, 270)   # picks a random initial angular between 90 and 270º
maxTime = 1000  # amount of time to run the simulation in seconds
tracer_tail = 200   # length of the tracer used in the simulation in the form of 'dots'
print(f"Theta 1 is : {theta1_0}\N{DEGREE SIGN}", f"\nTheta 2 is : {theta2_0}\N{DEGREE SIGN}") # prints initial angles
# m_x2 and m_y2 aren't used, but the positions could be appending to as a list for later analysis or plotting
#m_x2, m_y2 = [], []
qx, qy = deque(), deque()   # defines the queues used for the tracer for more efficient adding and removing
                            # of points for the desired effect
# initial conditions
y = np.array([omega1_0, radians(theta1_0), omega2_0, radians(theta2_0)])  # [velocity, displacement]
# define the figure and set properties, titles, etc.
fig = plt.figure()
fig.patch.set_facecolor('k')
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-(L1+L2)*1.1, (L1+L2)*1.1), ylim=(-(L1+L2)*1.1, (L1+L2)*1.1))
ax.grid(False)
ax.set_facecolor('k')
ax.set_xticks([])
ax.set_yticks([])
plt.title('Double Pendulum', color='w')
time_text = ax.text(-0.1, 0.95, '', color='w', transform=ax.transAxes)

# create the L1, L2, and tracer lines that will be updated in real time
line1, = ax.plot([], [], 'o-', lw=2, color='y')
line2, = ax.plot([], [], 'o-', color='c', lw=2)
tracer = ax.scatter([], [], s=8, c=[], cmap='afmhot')

# initial function to set the plots empty for FuncAnimation
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    tracer.set_offsets([])
    time_text.set_text('')
    return line1, line2, tracer

# Calls the above functions RK4_step and position, which returns y as a vector containing angular positions and
# angular velocity for each point 1 and 2, then uses the angles to find the position in cartesian coordinates
# each position of 1 and 2 are updated to the line plots after each step
# for the tracer, the x and y queues are stored with length tracer_tail, once it hits that length, it will then
# operate under the FIFO principle using append() and popleft()

def animate(i):
    global maxTime, delt, y

    if i == maxTime/delt:
        ani.event_source.stop()
    if i < tracer_tail:
        t = i/delt
        y = y + RK4_step(y, t, delt)
        x1, y1, x2, y2 = position(y[1], y[3])
        qx.append(x2[1])
        qy.append(y2[1])
    else:
        t = i/delt
        y = y + RK4_step(y, t, delt)
        x1, y1, x2, y2 = position(y[1], y[3])
        qx.append(x2[1])
        qy.append(y2[1])
        qx.popleft()
        qy.popleft()
    line1.set_data(x1, y1)  # updates line 1 and line 2 for each pendulum arm
    line2.set_data(x2, y2)
    tracer.set_offsets(np.c_[qx, qy])   # updates the scatter plot data for the  tracer
    tracer.set_array(np.linspace(0, 1, tracer_tail))    # updates the colormap length based on the queue length
    time_text.set_text('Time = %f' % (t/1000))  # updates the time
    return line1, line2, tracer

# defines the interval time based on the length of time it takes for one step
t0 = time()
animate(0)
animate(1)
t1 = time()
interval = (1000 * delt - (t1 - t0))
# used matplotlibs FuncAnimation function to iterate through a function that updates the line and scatter plots,
# giving us the desired output
ani = FuncAnimation(fig, animate,
                    interval=interval, init_func=init)

plt.show()
