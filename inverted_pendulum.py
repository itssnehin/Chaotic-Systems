import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint

# Parameters of the pendulum
g = 9.81    # acceleration due to gravity, in m/s^2
L = 1.0     # length of the pendulum, in meters
m = 1.0     # mass of the pendulum bob, in kg
b = 0.05    # damping coefficient, in kg*m^2/s
I = m * L**2  # moment of inertia of the pendulum

# Control parameters for pendulum 1
Kp1 = 50   # proportional gain
Kd1 = 20   # derivative gain

# Control parameters for pendulum 2 (less damping to allow more oscillations)
Kp2 = 50   # proportional gain (same as pendulum 1)
Kd2 = 5    # derivative gain (reduced to allow more oscillations)

# Desired angle for both pendulums
desired_theta = np.pi  # target is the inverted position

# Control function for pendulum 1
def control1(theta, theta_dot):
    torque = Kp1 * (desired_theta - theta) - Kd1 * theta_dot
    return torque

# Control function for pendulum 2
def control2(theta, theta_dot):
    torque = Kp2 * (desired_theta - theta) - Kd2 * theta_dot
    return torque

# Differential equations of motion for pendulum 1
def deriv1(y, t):
    theta, theta_dot = y
    # Calculate the control torque
    torque = control1(theta, theta_dot)
    # Equation of motion
    theta_ddot = (torque - b * theta_dot - m * g * L * np.sin(theta)) / I
    return [theta_dot, theta_ddot]

# Differential equations of motion for pendulum 2
def deriv2(y, t):
    theta, theta_dot = y
    # Calculate the control torque
    torque = control2(theta, theta_dot)
    # Equation of motion
    theta_ddot = (torque - b * theta_dot - m * g * L * np.sin(theta)) / I
    return [theta_dot, theta_ddot]

# Time parameters
t0 = 0.0    # initial time
tf = 10.0   # final time
dt = 0.02   # time step
t = np.arange(t0, tf, dt)

# Initial conditions for pendulum 1
theta0_1 = 0.0    # initial angle (downward position)
theta_dot0_1 = 0.0  # initial angular velocity
y0_1 = [theta0_1, theta_dot0_1]

# Initial conditions for pendulum 2
theta0_2 = np.pi / 4  # initial angle (45 degrees)
theta_dot0_2 = 0.5    # initial angular velocity
y0_2 = [theta0_2, theta_dot0_2]

# Solve the differential equations for pendulum 1
solution_1 = odeint(deriv1, y0_1, t)
theta_1 = solution_1[:, 0]
theta_dot_1 = solution_1[:, 1]

# Solve the differential equations for pendulum 2
solution_2 = odeint(deriv2, y0_2, t)
theta_2 = solution_2[:, 0]
theta_dot_2 = solution_2[:, 1]

# Convert theta to be within [-pi, pi] for visualization
theta_1 = (theta_1 + np.pi) % (2 * np.pi) - np.pi
theta_2 = (theta_2 + np.pi) % (2 * np.pi) - np.pi

# Prepare for animation
x1 = L * np.sin(theta_1)
y1 = -L * np.cos(theta_1)

x2 = L * np.sin(theta_2)
y2 = -L * np.cos(theta_2)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-L - 0.5, L + 0.5)
ax.set_ylim(-L - 0.5, L + 0.5)
ax.set_aspect('equal')
ax.grid(True)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('Two Inverted Pendulums with Different Initial Conditions')

# Initialize line objects for both pendulums
line1, = ax.plot([], [], 'o-', lw=2, color='blue', label='Pendulum 1')
line2, = ax.plot([], [], 'o-', lw=2, color='red', label='Pendulum 2')
ax.legend()

# Initialize animation function
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2

# Animation function
def animate(i):
    # Pendulum 1
    thisx1 = [0, x1[i]]
    thisy1 = [0, y1[i]]
    line1.set_data(thisx1, thisy1)

    # Pendulum 2
    thisx2 = [0, x2[i]]
    thisy2 = [0, y2[i]]
    line2.set_data(thisx2, thisy2)

    return line1, line2

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=len(t), init_func=init,
                            interval=dt*1000, blit=True)

# Save the animation as a GIF
ani.save('two_inverted_pendulums_oscillation.gif', writer='pillow', fps=50)

# Display the plot (final frame)
plt.show()
