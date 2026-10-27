import math

def calc_accelerations(x1, y1, x2, y2, m1, m2):
    rx = x2 - x1
    ry = y2 - y1
    r_squared = rx*rx + ry*ry
    r_norm = math.sqrt(r_squared)
    inv_r3 = 1.0 / (r_norm * r_squared) if r_squared != 0 else 0.0
    
    a1x = m2 * rx * inv_r3
    a1y = m2 * ry * inv_r3
    
    a2x = -m1 * rx * inv_r3
    a2y = -m1 * ry * inv_r3
    
    return a1x, a1y, a2x, a2y

def main():
    m1 = 1.0
    m2 = 0.1
    x1, y1 = -0.090909, 0.0
    v1x, v1y = 0.0, -0.095346
    x2, y2 = 0.909091, 0.0
    v2x, v2y = 0.0, 0.953463
    
    dt = 0.001
    total_steps = 17000
    output_steps = 1000
    
    results = []
    results.append([0.0, x1, y1, x2, y2])
    
    a1x, a1y, a2x, a2y = calc_accelerations(x1, y1, x2, y2, m1, m2)
    
    for step in range(1, total_steps + 1):
        v1x += 0.5 * dt * a1x
        v1y += 0.5 * dt * a1y
        v2x += 0.5 * dt * a2x
        v2y += 0.5 * dt * a2y
        
        x1 += dt * v1x
        y1 += dt * v1y
        x2 += dt * v2x
        y2 += dt * v2y
        
        a1x_new, a1y_new, a2x_new, a2y_new = calc_accelerations(x1, y1, x2, y2, m1, m2)
        
        v1x += 0.5 * dt * a1x_new
        v1y += 0.5 * dt * a1y_new
        v2x += 0.5 * dt * a2x_new
        v2y += 0.5 * dt * a2y_new
        
        a1x, a1y, a2x, a2y = a1x_new, a1y_new, a2x_new, a2y_new
        
        if step % output_steps == 0:
            t = step * dt
            results.append([t, x1, y1, x2, y2])
    
    print("t,x1,y1,x2,y2")
    for t, x1_val, y1_val, x2_val, y2_val in results:
        print(f"{int(t)},{x1_val:.6f},{y1_val:.6f},{x2_val:.6f},{y2_val:.6f}")

if __name__ == "__main__":
    main()