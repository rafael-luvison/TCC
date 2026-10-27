import numpy as np
from scipy.integrate import solve_ivp

# Parâmetros físicos
G = 1.0
m1 = 1.0
m2 = 0.1

# Condições iniciais
# Corpo 1
x1_0, y1_0 = -0.090909, 0.0
vx1_0, vy1_0 = 0.0, -0.095346

# Corpo 2
x2_0, y2_0 = 0.909091, 0.0
vx2_0, vy2_0 = 0.0, 0.953463

y0 = [x1_0, y1_0, vx1_0, vy1_0, x2_0, y2_0, vx2_0, vy2_0]

def equacoes_movimento(t, y):
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = y
    
    # Vetor de r1 para r2
    dx = x2 - x1
    dy = y2 - y1
    r = np.sqrt(dx**2 + dy**2)
    r3 = r**3
    
    # Acelerações
    ax1 = G * m2 * dx / r3
    ay1 = G * m2 * dy / r3
    ax2 = -G * m1 * dx / r3
    ay2 = -G * m1 * dy / r3
    
    return [vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2]

# Tempo de t=0 a t=17
t_span = (0.0, 17.0)
t_eval = np.arange(0, 18, dtype=float)

# Resolução numérica de alta precisão
sol = solve_ivp(
    equacoes_movimento, 
    t_span, 
    y0, 
    t_eval=t_eval, 
    method='DOP853', 
    rtol=1e-13, 
    atol=1e-13
)

# Impressão do cabeçalho CSV
print("t,x1,y1,x2,y2")

# Impressão dos dados formatados
for i in range(len(t_eval)):
    t = int(sol.t[i])
    x1 = sol.y[0, i]
    y1 = sol.y[1, i]
    x2 = sol.y[4, i]
    y2 = sol.y[5, i]
    print(f"{t},{x1:.6f},{y1:.6f},{x2:.6f},{y2:.6f}")