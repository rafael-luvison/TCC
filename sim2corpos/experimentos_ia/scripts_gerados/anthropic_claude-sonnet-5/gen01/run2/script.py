import numpy as np
from scipy.integrate import solve_ivp

G = 1.0
m1 = 1.0
m2 = 0.1

def derivs(t, state):
    x1, y1, x2, y2, vx1, vy1, vx2, vy2 = state
    dx = x2 - x1
    dy = y2 - y1
    r2 = dx*dx + dy*dy
    r3 = r2 * np.sqrt(r2)
    ax1 = G * m2 * dx / r3
    ay1 = G * m2 * dy / r3
    ax2 = -G * m1 * dx / r3
    ay2 = -G * m1 * dy / r3
    return [vx1, vy1, vx2, vy2, ax1, ay1, ax2, ay2]

x1_0, y1_0 = -0.090909, 0.0
x2_0, y2_0 = 0.909091, 0.0
vx1_0, vy1_0 = 0.0, -0.095346
vx2_0, vy2_0 = 0.0, 0.953463

state0 = [x1_0, y1_0, x2_0, y2_0, vx1_0, vy1_0, vx2_0, vy2_0]

t_eval = np.arange(0, 18, 1.0)

sol = solve_ivp(
    derivs,
    (0, 17),
    state0,
    method='DOP853',
    t_eval=t_eval,
    rtol=1e-12,
    atol=1e-12,
    max_step=0.01
)

print("t,x1,y1,x2,y2")
for i in range(len(sol.t)):
    t = sol.t[i]
    x1, y1, x2, y2 = sol.y[0, i], sol.y[1, i], sol.y[2, i], sol.y[3, i]
    print(f"{t:.6g},{x1:.6f},{y1:.6f},{x2:.6f},{y2:.6f}")