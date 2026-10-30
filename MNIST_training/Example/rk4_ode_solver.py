import jax
from jax import numpy as jnp

# Runge-Kutta solver
def rk4_step(f, y, t, dt, kappa, kappa1, g, J, x_in, p_in):
    """Performs a single step of the 4th-order Runge-Kutta method."""
    k1 = f(t, y, kappa, kappa1, g, J, x_in, p_in)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1, kappa, kappa1, g, J, x_in, p_in)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2, kappa, kappa1, g, J, x_in, p_in)
    k4 = f(t + dt, y + dt * k3, kappa, kappa1, g, J, x_in, p_in)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def solve_ode(f, y0, t_span, num_steps, dt, kappa, kappa1, g, J, x_in, p_in):
    t_start, t_end = t_span
    ts = jnp.linspace(t_start, t_end, num_steps, dtype=jnp.float32)  # Corrected for inclusion of t_end

    def step(y, t):
        """Step function for JAX scan"""
        y_next = rk4_step(f, y, t, dt, kappa, kappa1, g, J, x_in, p_in)
        return y_next, y_next  # Carry (updated state), output (new state)

    # Run JAX scan for efficient iteration
    _, ys = jax.lax.scan(step, y0, ts)

    return ts, jnp.vstack([y0, ys])  # Stack initial state with results