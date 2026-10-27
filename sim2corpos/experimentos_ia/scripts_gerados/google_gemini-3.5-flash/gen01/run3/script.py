import numpy as np
from scipy.integrate import solve_ivp

# Parâmetros
G = 1.0
m1 = 1.0
m2 = 0.1

# Condições iniciais: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
y0 = [-0.090909, 0.0, 0.909091, 0.0, 0.0, -0.095346, 0.0, 0.953463]

def derivatives(t, y):
    x1, y1, x2, y2, vx1, vy1, vx2, vy2 = y
    
    dx = x2 - x1
    dy = y2 - y1
    dist = (dx**2 + dy**2)**0.5
    dist3 = dist**3
    
    # Acelerações
    ax1 = G * m2 * dx / dist3
    ay1 = G * m2 * dy / dist3
    ax2 = -G * m1 * dx / dist3
    ay2 = -G * m1 * dy / dist3
    
    return [vx1, vy1, vx2, vy2, ax1, ay1, ax2, ay2]

# Integração de t=0 a t=17
t_eval = np.arange(18)
sol = solve_ivp(derivatives, [0, 17], y0, t_eval=t_eval, method='DOP853', rtol=1e-13, atol=1e-13)

# Impressão do resultado em formato CSV
print("t,x1,y1,x2,y2")
for i in range(len(t_eval)):
    t = int(sol.t[i])
    x1 = sol.y[0, i]
    y1 = sol.y[1, i]
    x2 = sol.y[2, i]
    y2 = sol.y[3, i]
    print(f"{t},{x1},{y1},{x2},{y2}")