import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set the number of steps in the random walk
num_steps = 500

# Generate random angles and step sizes
angles = np.random.uniform(0, 2 * np.pi, num_steps)
step_sizes = np.random.exponential(scale=1.0, size=num_steps)  # Exponential distribution for step sizes

# Calculate the step increments in x and y directions
step_x = step_sizes * np.cos(angles)
step_y = step_sizes * np.sin(angles)

# Compute the cumulative sum to get the position at each step
position_x = np.cumsum(step_x)
position_y = np.cumsum(step_y)

# Initialize the plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(min(position_x) - 10, max(position_x) + 10)
ax.set_ylim(min(position_y) - 10, max(position_y) + 10)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('2D Random Walk with Variable Step Sizes')
ax.grid(True)

# Initialize line object
line, = ax.plot([], [], lw=2)

# Function to initialize the animation
def init():
    line.set_data([], [])
    return line,

# Animation function called sequentially
def animate(i):
    line.set_data(position_x[:i], position_y[:i])
    return line,

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=num_steps, init_func=init,
                                interval=20, blit=True)

# Save the animation as a GIF
ani.save('random_walk.gif', writer='pillow')

# Display the plot
plt.show()
