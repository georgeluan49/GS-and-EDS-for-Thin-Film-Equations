from newton import newton
import jax
import jax.numpy as jnp



def fixed_timestepping(residual, h0, K, sampled_iters, rtol=1e-8, atol=1e-10):
    """
    Fixed timestep simulation
    
    Args:
        residual: Description
        h0: initial condition, (n,) jnp vector
        K: Total number of timesteps
        sampled_iters: iterations to collect the solution state from
        rtol: relative tolerance for the Newton solver
        atol: absolute tolerance for the Newton solver

    Output:
        A dict containing info on success/failure of the simulation and
        sampled solution states (if the simulation succeeds) 
    """

    h_sampled = []
    h_curr = h0

    for k in range(K):

        if k % 1000 == 0:
            print(f"Iter {k}")

        h_prev = h_curr
        def F(h):
            return residual(h, h_prev)

        success = False

        for tol_factor in (1.0, 2.0, 5.0, 10.0):
            h_next, ok, info = newton(
                F, h_curr, rtol=rtol, atol=tol_factor * atol
            )

            if ok:
                h_curr = h_next
                success = True
                break

        if not success:
            return {
                "success": False,
                "reason": "newton_failed",
                "timestep": k,
                "h": h_curr,
            }

        if k in sampled_iters:
            h_sampled.append(jnp.array(h_curr))

    return {
        "success": True,
        "h_final": h_curr,
        "h_sampled": h_sampled,
    }



def adaptive_timestepping(
    residual,              
    h0,                    
    dt0,
    tstart,
    tfinish,
    rtol=1e-8,
    atol=1e-10,
    max_dt_shrink=4,
    shrink=0.5,
    grow=1.1,
    r_hi=1e-2,
):
    """
    Adaptive timestep simulation

    Args:
        residual: residual of the difference scheme 
            with the form residual(h, h_curr, dt)
        h0: initial condition, (n,) jnp vector
        dt0: initial timestep size
        tstart: starting time
        tfinish: finishing time
        rtol: relative tolerance for the Newton solver
        atol: absolute tolerance for the Newton solver
        max_dt_shrink: maximum number of times dt can shrink
        shrink: dt <- shrink * dt
        grow: dt <- grow * dt
        r_hi: tolerance for relative temporal variation

    Outputs:
    t_hist: time discretization
    h_hist: solution states at the corresponding times
    """

    # --- initialization ---
    t_curr = float(tstart)
    h_curr = h0
    dt = float(dt0)

    t_hist = [t_curr]
    h_hist = [h_curr]

    step = 0

    # --- main loop ---
    while t_curr < tfinish:
        step += 1

        # prevent overshoot
        dt = min(dt, tfinish - t_curr)

        accepted = False
        h_new = None

        # bounded dt-retry loop
        for _ in range(max_dt_shrink):

            # define residual for this attempt
            F = lambda h: residual(h, h_curr, dt)

            # Newton solve
            h_try, success, info = newton(
                F,
                h_curr,
                rtol=rtol,
                atol=atol,
            )

            # shrink on Newton failure
            if not success:
                dt *= shrink
                continue

            # sanity check (crucial)
            if not jnp.all(jnp.isfinite(h_try)):
                dt *= shrink
                continue

            # relative change indicator
            num = jnp.linalg.norm(h_try - h_curr, ord=jnp.inf)
            den = jnp.linalg.norm(h_try, ord=jnp.inf) + 1e-15
            r = num / den

            # reject if too aggressive
            if r > r_hi:
                dt *= shrink
                continue

            # accept
            accepted = True
            h_new = h_try
            break

        if not accepted:
            raise RuntimeError(
                f"Step rejected after {max_dt_shrink} dt reductions at t={t_curr}"
            )

        # commit step
        t_curr += dt
        h_curr = h_new

        t_hist.append(t_curr)
        h_hist.append(h_curr)

        # conservative growth
        dt *= grow

    return t_hist, h_hist
