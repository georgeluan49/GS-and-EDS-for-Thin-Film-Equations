import os
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import operators
import timestepper

os.makedirs("./Results/FixedTime", exist_ok=True)
cpu = jax.devices("cpu")[0]

N = 256
dx = float(2 / N)
dt = float(1e-7)
eps = float(1e-14)

x_coarse = jnp.linspace(-1.0, 1.0, N, endpoint=False, dtype=jnp.float64, device=cpu)

h0_coarse = 0.8 - jnp.cos(jnp.pi * x_coarse) + 0.25 * jnp.cos(2 * jnp.pi * x_coarse)

# Need to make sure h0_coarse lives on cpu
h0_coarse = jax.device_put(h0_coarse, cpu)

K = 10000
sampled_iters = {9, 99, 999, 9999}

EDS = operators.make_EDS(dx, dt, eps)
results = timestepper.fixed_timestepping(
    EDS, h0_coarse, K, sampled_iters, rtol=1e-8, atol=1e-10
)

if not results.get("success", False):
    plt.figure()
    plt.plot(x_coarse, results["h"])
    plt.title(f"Solution state before failure at t = {results["timestep"]:.3e}")
    plt.savefig("./Results/FixedTime/sim_failure_eds_coarse.png", dpi=200)
    plt.close()
    raise RuntimeError(f"Simulation failed at {results["timestep"]}")


# plotting
plt.figure()
plt.plot(x_coarse, h0_coarse, label="t = 0")

for h, k in zip(results["h_sampled"], sorted(sampled_iters)):
    plt.plot(x_coarse, h, label=f"t = {(k+1)*dt:.3e}")

plt.xlabel("$x$")
plt.ylabel(r"$h_\epsilon(x,t),\ \epsilon=10^{-14}$")
plt.legend()
plt.savefig("./Results/FixedTime/sim_results_eds_coarse.png", dpi=200)
plt.close()

plt.figure()
plt.plot(x_coarse[128:138], results["h_sampled"][-1][128:138])
plt.xlabel("$x$")
plt.ylabel(r"$h_\epsilon(x,0.001),\ \epsilon=10^{-14}$")
plt.savefig("./Results/FixedTime/eds_coarse_1e-3.png", dpi=200)
plt.close()
