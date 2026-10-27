import csv, math, sys
G=1.0; m1=1.0; m2=0.1; M=m1+m2; r0=1.0
w=math.sqrt(G*M/r0**3); T=2*math.pi/w
def exact(t):
    c,s=math.cos(w*t),math.sin(w*t)
    # sinal: x2 inicia em +10/11, y2 cresce com vy2>0
    return (-m2/M*r0*c, -m2/M*r0*s, m1/M*r0*c, m1/M*r0*s)
for f in sys.argv[1:]:
    rows=list(csv.DictReader(open(f)))
    se=0; n=0; last=None
    for r in rows:
        t=float(r['t']); x1,y1,x2,y2=exact(t)
        d1=math.hypot(float(r['x1'])-x1,float(r['y1'])-y1)
        d2=math.hypot(float(r['x2'])-x2,float(r['y2'])-y2)
        se+=d1*d1+d2*d2; n+=2; last=(t,d1,d2)
    rmse=math.sqrt(se/n)
    print(f"{f}: T_exato={T:.4f}  RMSE={rmse:.3e}  erro_final corpo2={last[2]:.3e} (t={last[0]:.0f})  raio final corpo2={math.hypot(float(rows[-1]['x2']),float(rows[-1]['y2'])):.6f} (exato {m1/M*r0:.6f})")
