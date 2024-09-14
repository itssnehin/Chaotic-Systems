import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, PillowWriter

# Parameters
g = 9.81   # Acceleration due to gravity (m/s^2)
L1 = 1.0   # Length of first rod (m)
L2 = 1.0   # Length of second rod (m)
m1 = 1.0   # Mass of first bob (kg)
m2 = 1.0   # Mass of second bob (kg)

# Time span
t_max = 20
t_eval = np.linspace(0, t_max, 2000)

# Equations of motion
def deriv(t, y):
    theta1, omega1, theta2, omega2 = y

    delta = theta2 - theta1

    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
    den2 = (L2 / L1) * den1

    omega1_dot = (
        m2 * L1 * omega1 ** 2 * np.sin(delta) * np.cos(delta)
        + m2 * g * np.sin(theta2) * np.cos(delta)
        + m2 * L2 * omega2 ** 2 * np.sin(delta)
        - (m1 + m2) * g * np.sin(theta1)
    ) / den1

    omega2_dot = (
        -m2 * L2 * omega2 ** 2 * np.sin(delta) * np.cos(delta)
        + (m1 + m2)
        * (
            g * np.sin(theta1) * np.cos(delta)
            - L1 * omega1 ** 2 * np.sin(delta)
            - g * np.sin(theta2)
        )
    ) / den2

    return [omega1, omega1_dot, omega2, omega2_dot]

# Initial conditions for pendulum 1
theta1_0 = np.pi / 2    # Initial angle of first pendulum (radians)
theta2_0 = np.pi / 2    # Initial angle of second pendulum (radians)
omega1_0 = 0.0          # Initial angular velocity of first pendulum (rad/s)
omega2_0 = 0.0          # Initial angular velocity of second pendulum (rad/s)
y0 = [theta1_0, omega1_0, theta2_0, omega2_0]

# Initial conditions for pendulum 2 (slightly different)
delta_theta = 0.05      # Small difference in initial angle
theta1_0_2 = theta1_0 + delta_theta
theta2_0_2 = theta2_0 + delta_theta
y0_2 = [theta1_0_2, omega1_0, theta2_0_2, omega2_0]

# Solve ODE for pendulum 1
sol = solve_ivp(deriv, [0, t_max], y0, t_eval=t_eval, method='RK45')
theta1 = sol.y[0]
theta2 = sol.y[2]

# Solve ODE for pendulum 2
sol2 = solve_ivp(deriv, [0, t_max], y0_2, t_eval=t_eval, method='RK45')
theta1_2 = sol2.y[0]
theta2_2 = sol2.y[2]

# Convert to Cartesian coordinates for pendulum 1
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

# Convert to Cartesian coordinates for pendulum 2
x1_2 = L1 * np.sin(theta1_2)
y1_2 = -L1 * np.cos(theta1_2)
x2_2 = x1_2 + L2 * np.sin(theta2_2)
y2_2 = y1_2 - L2 * np.cos(theta2_2)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(- (L1 + L2 + 0.5), L1 + L2 + 0.5)
ax.set_ylim(- (L1 + L2 + 0.5), L1 + L2 + 0.5)
ax.set_aspect('equal')
ax.grid()

# Initialize lines and trajectories
line1, = ax.plot([], [], 'o-', lw=2, color='blue', label='Pendulum 1')
line2, = ax.plot([], [], 'o-', lw=2, color='green', label='Pendulum 2')
trajectory1, = ax.plot([], [], '-', lw=1, color='red', alpha=0.5)
trajectory2, = ax.plot([], [], '-', lw=1, color='orange', alpha=0.5)
ax.legend()

# Store trajectory points
traj1_x, traj1_y = [], []
traj2_x, traj2_y = [], []

# Animation function
def animate(i):
    # Pendulum 1
    thisx1 = [0, x1[i], x2[i]]
    thisy1 = [0, y1[i], y2[i]]
    line1.set_data(thisx1, thisy1)
    traj1_x.append(x2[i])
    traj1_y.append(y2[i])
    trajectory1.set_data(traj1_x, traj1_y)

    # Pendulum 2
    thisx2 = [0, x1_2[i], x2_2[i]]
    thisy2 = [0, y1_2[i], y2_2[i]]
    line2.set_data(thisx2, thisy2)
    traj2_x.append(x2_2[i])
    traj2_y.append(y2_2[i])
    trajectory2.set_data(traj2_x, traj2_y)

    return line1, line2, trajectory1, trajectory2

# Create animation
ani = FuncAnimation(fig, animate, frames=len(t_eval), interval=20, blit=True)

# Save the animation as a GIF
ani.save('double_pendulum.gif', writer=PillowWriter(fps=30))

# Display the animation
plt.show()