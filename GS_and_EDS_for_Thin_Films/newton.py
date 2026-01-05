import jax
import jax.numpy as jnp

device = jax.devices("cpu")[0]

def newton(f, x_0, rtol=1e-8, atol=1e-10, max_iter=50):

    x = x_0
    f_jac = jax.jacfwd(f)

    Fx0 = f(x_0)
    res_scale = jnp.linalg.norm(Fx0, ord=jnp.inf)

    def newton_step(x):
        Fx = f(x)
        J  = f_jac(x)
        dx = jnp.linalg.solve(J, Fx)
        return x - dx, dx, Fx

    # Compile the Newton step
    newton_step = jax.jit(newton_step, device=device)

    for n in range(1, max_iter + 1):
        x_new, dx, Fx = newton_step(x)

        step_norm = jnp.linalg.norm(dx, ord=jnp.inf)
        x_norm    = jnp.linalg.norm(x_new, ord=jnp.inf)
        res_norm  = jnp.linalg.norm(Fx, ord=jnp.inf)

        step_ok = step_norm <= atol + rtol * jnp.maximum(1.0, x_norm)
        res_ok  = res_norm  <= atol + rtol * res_scale

        if step_ok and res_ok:
            return x_new, True, {
                "iters": n,
                "step_norm": float(step_norm),
                "res_norm": float(res_norm),
            }

        x = x_new

    return x, False, {
        "iters": max_iter,
        "step_norm": float(step_norm),
        "res_norm": float(res_norm),
    }
