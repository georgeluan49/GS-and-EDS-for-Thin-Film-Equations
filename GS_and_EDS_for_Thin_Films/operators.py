import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# --- flux components ---

def f(h, eps):
    h = jnp.maximum(h, 0.0)
    h35 = h**3 * jnp.sqrt(h)   
    return h**4 / (eps + h35)


def a_gs(s1, s2, eps):
    s = 0.5 * (s1 + s2)
    return f(s, eps)


def a_eds(s1, s2, eps):
    h1 = jnp.maximum(s1, 0.0)
    h2 = jnp.maximum(s2, 0.0)

    Gp1 = 2.0 * jnp.sqrt(h1) - eps / (3.0 * h1**3)
    Gp2 = 2.0 * jnp.sqrt(h2) - eps / (3.0 * h2**3)

    num = h1 - h2
    den = Gp1 - Gp2

    # fallback when h1 ~ h2
    a_diag = f(h1, eps)

    return jnp.where(jnp.abs(den) > 1e-14, num / den, a_diag)



# --- Diff schemes ---

def make_GS(dx, dt, eps):

    def GS_core(h, h_curr):
        hf1 = jnp.roll(h, -1)
        hb1 = jnp.roll(h, 1)
        hb2 = jnp.roll(h, 2)

        h_xxx = (hf1 - 3*h + 3*hb1 - hb2) / dx**3
        h_xxxf1 = jnp.roll(h_xxx, -1)

        f_f12 = a_gs(h, hf1, eps) * h_xxxf1
        f_b12 = a_gs(hb1, h, eps) * h_xxx

        return h - h_curr + (dt / dx) * (f_f12 - f_b12)
    
    return jax.jit(GS_core)


def make_GS_adpt(dx, eps):

    def GS_core(h, h_curr, dt):
        hf1 = jnp.roll(h, -1)
        hb1 = jnp.roll(h, 1)
        hb2 = jnp.roll(h, 2)

        h_xxx = (hf1 - 3*h + 3*hb1 - hb2) / dx**3
        h_xxxf1 = jnp.roll(h_xxx, -1)

        f_f12 = a_gs(h, hf1, eps) * h_xxxf1
        f_b12 = a_gs(hb1, h, eps) * h_xxx

        return h - h_curr + (dt / dx) * (f_f12 - f_b12)
    
    return jax.jit(GS_core)


def make_EDS(dx, dt, eps):

    def EDS_core(h, h_curr):
        hf1 = jnp.roll(h, -1)
        hb1 = jnp.roll(h, 1)
        hb2 = jnp.roll(h, 2)

        h_xxx = (hf1 - 3*h + 3*hb1 - hb2) / dx**3
        h_xxxf1 = jnp.roll(h_xxx, -1)

        f_f12 = a_eds(h, hf1, eps) * h_xxxf1
        f_b12 = a_eds(hb1, h, eps) * h_xxx

        return h - h_curr + (dt / dx) * (f_f12 - f_b12)

    EDS = jax.jit(EDS_core)
    
    return EDS


def make_EDS_adpt(dx, eps):

    def EDS_core(h, h_curr, dt):
        hf1 = jnp.roll(h, -1)
        hb1 = jnp.roll(h, 1)
        hb2 = jnp.roll(h, 2)

        h_xxx = (hf1 - 3*h + 3*hb1 - hb2) / dx**3
        h_xxxf1 = jnp.roll(h_xxx, -1)

        f_f12 = a_eds(h, hf1, eps) * h_xxxf1
        f_b12 = a_eds(hb1, h, eps) * h_xxx

        return h - h_curr + (dt / dx) * (f_f12 - f_b12)
    
    return jax.jit(EDS_core)