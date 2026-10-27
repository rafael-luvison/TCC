import numpy as np
from scipy.integrate import solve_ivp

# Parâmetros do sistema
G = 1.0
m1 = 1.0
m2 = 0.1

# Condições iniciais: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
y0 = [-0.090909, 0.0, 0.909091, 0.0, 0.0, -0.095346, 0.0, 0.953463]

# Equações de movimento (Gravitação Newtoniana)
def equations(t, y):
    x1, y1, x2, y2, vx1, vy1, vx2, vy2 = y
    dx = x2 - x1
    dy = y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    factor = G / (dist**3)
    
    ax1 = m2 * dx * factor
    ay1 = m2 * dy * factor
    ax2 = -m1 * dx * factor
    ay2 = -m1 * dy * factor
    
    return [vx1, vy1, vx2, vy2, ax1, ay1, ax2, ay2]

# Tempos de avaliação: de t=0 a t=17
t_eval = np.arange(18)

# Resolução numérica de alta precisão
sol = solve_ivp(equations, (0, 17), y0, t_eval=t_eval, rtol=1e-12, atol=1e-12, method='DOP853')

# Impressão do resultado em formato CSV
print("t,x1,y1,x2,y2")
for i in range(18):
    t = int(sol.t[i])
    x1, y1, x2, y2 = sol.y[0, i], sol.y[1, i], sol.y[2, i], sol.y[3, i]
    print(f"{t},{x1:.8f},{y1:.8f},{x2:.8f},{y2:.8f}")