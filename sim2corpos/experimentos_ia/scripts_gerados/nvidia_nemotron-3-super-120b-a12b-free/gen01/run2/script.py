import numpy as np
from scipy.integrate import solve_ivp
import sys

def two_body_deriv(t, y, G, m1, m2):
    x1, y1, x2, y2, vx1, vy1, vx2, vy2 = y
    dx = x2 - x1
    dy = y2 - y1
    r3 = (dx*dx + dy*dy)**1.5
    ax1 = G * m2 * dx / r3
    ay1 = G * m2 * dy / r3
    ax2 = -G * m1 * dx / r3
    ay2 = -G * m1 * dy / r3
    return [vx1, vy1, vx2, vy2, ax1, ay1, ax2, ay2]

def main():
    G = 1.0
    m1 = 1.0
    m2 = 0.1
    # initial state [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
    y0 = np.array([-0.090909, 0.0, 0.909091, 0.0,
                   0.0, -0.095346, 0.0, 0.953463], dtype=float)

    t_span = (0.0, 17.0)
    t_eval = np.arange(0, 18)  # t = 0,1,...,17

    sol = solve_ivp(two_body_deriv, t_span, y0, args=(G, m1, m2),
                    method='DOP853', t_eval=t_eval,
                    rtol=1e-12, atol=1e-14)

    if not sol.success:
        raise RuntimeError("Integration failed: " + sol.message)

    # Extract positions
    t = sol.t
    x1 = sol.y[0].copy()
    y1 = sol.y[1].copy()
    x2 = sol.y[2].copy()
    y2 = sol.y[3].copy()

    # Force exact initial values (avoid integration drift at t=0)
    x1[0], y1[0] = y0[0], y0[1]
    x2[0], y2[0] = y0[2], y0[3]

    out = sys.stdout
    out.write("t,x1,y1,x2,y2\n")
    for ti, x1i, y1i, x2i, y2i in zip(t, x1, y1, x2, y2):
        out.write(f"{ti:.12f},{x1i:.12f},{y1i:.12f},{x2i:.12f},{y2i:.12f}\n")

if __name__ == "__main__":
    main()