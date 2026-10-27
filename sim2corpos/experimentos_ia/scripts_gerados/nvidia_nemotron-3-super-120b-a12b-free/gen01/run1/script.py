import math

def main():
    m1 = 1.0
    m2 = 0.1
    M = m1 + m2          # total mass
    r = 1.0              # distance between the bodies (from initial conditions)
    r1 = m2 / M * r      # radius of body 1 around the COM
    r2 = m1 / M * r      # radius of body 2 around the COM
    omega = math.sqrt(M) # angular speed (G = 1, r = 1)

    print("t,x1,y1,x2,y2")
    for t in range(0, 18):
        phi = omega * t
        x1 = -r1 * math.cos(phi)
        y1 = -r1 * math.sin(phi)
        x2 =  r2 * math.cos(phi)
        y2 =  r2 * math.sin(phi)
        print(f"{t},{x1:.12f},{y1:.12f},{x2:.12f},{y2:.12f}")

if __name__ == "__main__":
    main()