import os
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython import display

import operators
import timestepper


os.makedirs("./Results/AdaptiveTime", exist_ok=True)
cpu = jax.devices("cpu")[0]

N = 256
dx = float(2 / N)
dt = float(1e-7)
eps = float(1e-14)

x_coarse = jnp.linspace(-1.0, 1.0, N, endpoint=False, dtype=jnp.float64, device=cpu)

h0_coarse = 0.8 - jnp.cos(jnp.pi * x_coarse) + 0.25 * jnp.cos(2 * jnp.pi * x_coarse)

# Need to make sure h0_coarse lives on cpu
h0_coarse = jax.device_put(h0_coarse, cpu)

EDS = operators.make_EDS_adpt(dx, eps)
t, h = timestepper.adaptive_timestepping(EDS, h0_coarse, 1e-7, 0, 0.05)



# create the figure outlines
fig, ax = plt.subplots()

# create line artist
line_plotted, = ax.plot([], [])

# axis limits
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(0, 2.1)
ax.set_xlabel("$x$")
ax.set_ylabel("$h_\epsilon(x, t),\ \epsilon = $1e-14")

# time text
time_text = ax.text(
    0.618, 0.95, "$t = $0",
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment='top'
)

def AnimationFunction(frame):

    y = h[frame]
    line_plotted.set_data(x_coarse, y)

    t_curr = t[frame]          
    time_text.set_text(f"$t = ${t_curr:.3e}")

    return line_plotted, time_text

sim_results = FuncAnimation(fig, AnimationFunction, frames=len(t), blit=False)
sim_results.save("./Results/AdaptiveTime/animation_eds_coarse.gif", fps=30)
