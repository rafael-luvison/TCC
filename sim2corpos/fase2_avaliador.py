"""
Batchmark — FASE 2: O AVALIADOR (ponto de entrada).

Roda cada script gerado na Fase 1 de forma ISOLADA (subprocesso, nunca no kernel)
e JUSTA (mesmo interpretador, threads travadas em 1, execução serial, K repetições
com mín/mediana, mesmo ground truth e mesmo parser) e coleta as métricas.

Salva TUDO com histórico: stdout/stderr de cada script, resultados por rodada, e um
log completo do que apareceu na tela, em pastas com timestamp (nada é sobrescrito).

Rode:  python fase2_avaliador.py
Confere se já há scripts; se não houver (ou se você pedir), chama a Fase 1 antes.
"""
import os
import re
import sys
import io
import json
import time
import datetime
import subprocess
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

import fase1_pedinte as f1

PASTA = f1.PASTA
MUNDO = f1.MUNDO
TIMEOUT = 60
K = 5
ENV_JUSTO = {**os.environ,
             "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
             "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "MPLBACKEND": "Agg"}


# ---- ground truth: Verlet fino embutido ---------------------------------
def _accel(p, m, G):
    d = p[1] - p[0]; r = (d @ d) ** 0.5
    a = np.empty((2, 2)); a[0] = G * m[1] * d / r ** 3; a[1] = -G * m[0] * d / r ** 3
    return a


def ground_truth(mundo, dt=1e-4):
    G = mundo["G"]; m = np.array([mundo["m1"], mundo["m2"]], float)
    p = np.array([mundo["r1"], mundo["r2"]], float); v = np.array([mundo["v1"], mundo["v2"]], float)
    n = int(round(mundo["t_final"] / dt)); cada = int(round(mundo["dt_amostra"] / dt))
    a = _accel(p, m, G)
    L = [(0.0, p[0, 0], p[0, 1], p[1, 0], p[1, 1])]
    for i in range(1, n + 1):
        p = p + v * dt + 0.5 * a * dt ** 2
        a2 = _accel(p, m, G); v = v + 0.5 * (a + a2) * dt; a = a2
        if i % cada == 0:
            L.append((i * dt, p[0, 0], p[0, 1], p[1, 0], p[1, 1]))
    return pd.DataFrame(L, columns=["t", "x1", "y1", "x2", "y2"])


GT = ground_truth(MUNDO)


def parse_csv(texto):
    def cel(l):
        l = l.strip().strip("`").strip(); sep = "|" if "|" in l else ","
        return [c.strip() for c in l.split(sep) if c.strip() != ""]
    Ls = texto.splitlines()
    i0 = next((i for i, l in enumerate(Ls) if [c.lower() for c in cel(l)] == ["t", "x1", "y1", "x2", "y2"]), None)
    corpo = Ls[i0 + 1:] if i0 is not None else Ls
    tab = []
    for l in corpo:
        c = cel(l)
        if len(c) != 5:
            if tab: break
            continue
        try: tab.append([float(x) for x in c])
        except ValueError:
            if tab: break
            continue
    return pd.DataFrame(tab, columns=["t", "x1", "y1", "x2", "y2"]) if tab else None


_WRAPPER = r'''import sys, runpy, resource, json
try:
    runpy.run_path(sys.argv[1], run_name="__main__")
finally:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak = ru.ru_maxrss/1e6 if sys.platform=="darwin" else ru.ru_maxrss/1024.0
    sys.stderr.write("\n__METRICS__ "+json.dumps({"peak_rss_mb":peak,"cpu_s":ru.ru_utime+ru.ru_stime})+"\n")
'''


def _executa(script_path, wrapper_path):
    t0 = time.perf_counter()
    try:
        r = subprocess.run([sys.executable, str(wrapper_path.resolve()), str(script_path.resolve())],
                           cwd=script_path.parent, env=ENV_JUSTO, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "wall": TIMEOUT, "stdout": "", "stderr": "(timeout)", "peak_rss_mb": None, "cpu_s": None}
    wall = time.perf_counter() - t0
    met = re.search(r"__METRICS__ (\{.*\})", r.stderr); met = json.loads(met.group(1)) if met else {}
    return {"status": "ok" if r.returncode == 0 else "crash", "wall": wall,
            "stdout": r.stdout, "stderr": r.stderr,
            "peak_rss_mb": met.get("peak_rss_mb"), "cpu_s": met.get("cpu_s")}


def acuracia(df):
    g = GT.copy(); a = df.copy(); g["tk"] = g["t"].round(6); a["tk"] = a["t"].round(6)
    m = g.merge(a, on="tk", suffixes=("_g", "_a"))
    if len(m) == 0: return None
    e1 = np.hypot(m.x1_a - m.x1_g, m.y1_a - m.y1_g); e2 = np.hypot(m.x2_a - m.x2_g, m.y2_a - m.y2_g)
    rmse = float(np.sqrt(np.mean(np.r_[e1.values ** 2, e2.values ** 2])))
    erro_final = float(np.hypot(m.x2_a - m.x2_g, m.y2_a - m.y2_g).iloc[-1])
    em = ((e1 + e2) / 2).values; tt = m.t_g.values
    slope = float(np.polyfit(tt, em, 1)[0]) if len(tt) > 1 else np.nan
    M = MUNDO["m1"] + MUNDO["m2"]
    cmx = (MUNDO["m1"] * a.x1 + MUNDO["m2"] * a.x2) / M; cmy = (MUNDO["m1"] * a.y1 + MUNDO["m2"] * a.y2) / M
    com_var = float(np.hypot(cmx.max() - cmx.min(), cmy.max() - cmy.min()))
    return {"rmse": rmse, "erro_final": erro_final, "cresc_erro": slope, "com_var": com_var, "n_linhas": len(df)}


def analise_estatica(c):
    cl = c.lower()
    if "solve_ivp" in cl or "odeint" in cl: metodo = "scipy"
    elif "verlet" in cl or "leapfrog" in cl: metodo = "verlet"
    elif "rk4" in cl or "runge" in cl: metodo = "rk4"
    elif "kepler" in cl or "anomaly" in cl or "anomalia" in cl: metodo = "analitico"
    elif "euler" in cl: metodo = "euler"
    else: metodo = "indefinido"
    return {"metodo": metodo,
            "simpletico": any(k in cl for k in ["verlet", "leapfrog", "symplectic"]),
            "adaptativo": ("solve_ivp" in cl) or ("rtol" in cl),
            "autoverifica": any(k in cl for k in ["energy", "energia", "conserv", "momento"]),
            "linhas": c.count("\n") + 1, "n_imports": len(re.findall(r"(?m)^\s*(?:import|from)\s", c))}


def camada(res, df, acc, limiar=0.1):
    if res["status"] == "timeout": return "1-timeout"
    if res["status"] == "crash": return "1-crash"
    if df is None or len(df) != MUNDO["n_linhas"]: return "2-formato"
    if acc is None or acc["rmse"] > limiar: return "3-impreciso"
    return "ok"


# ---- gestão dos scripts -------------------------------------------------
def scripts_existentes():
    return sorted(PASTA.glob("*/gen*/run*/script.py"))


def garantir_scripts():
    existentes = scripts_existentes()
    if not existentes:
        print("Nenhum script encontrado na pasta — rodando a FASE 1 para gerar...\n")
        f1.rodar_fase1()
        return
    modelos = sorted({p.parents[2].name for p in existentes})
    print(f"Já existem {len(existentes)} scripts de {len(modelos)} modelo(s) em {PASTA}/:")
    for mdl in modelos:
        gens = sorted({p.parents[1].name for p in existentes if p.parents[2].name == mdl})
        print(f"   - {mdl}  ({', '.join(gens)})")
    resp = input("\n(r) reusar os existentes   ou   (g) gerar novos com a Fase 1?  [r] > ").strip().lower()
    if resp == "g":
        f1.rodar_fase1()


# ---- o batchmark --------------------------------------------------------
def rodar_batchmark():
    wrapper = PASTA / "_wrapper.py"; wrapper.write_text(_WRAPPER)
    GT.to_csv(PASTA / "ground_truth.csv", index=False)

    run_dirs = sorted(p for p in PASTA.glob("*/gen*/run*") if p.is_dir())
    linhas = []
    for rd in run_dirs:
        modelo = rd.parents[1].name; gen = rd.parent.name; run = rd.name
        rotulo = f"{modelo}/{gen}/{run}"
        print(f"-> {rotulo} ... ", end="", flush=True)
        script = rd / "script.py"
        if not script.exists():
            linhas.append({"modelo": modelo, "gen": gen, "run": run, "camada": "0-sem_script"})
            print("sem script (geração falhou)"); continue

        res = _executa(script, wrapper)
        # salva o que o script produziu (stdout) e o erro (stderr) — histórico por run
        (rd / "saida_stdout.txt").write_text(res["stdout"] or "")
        (rd / "saida_stderr.txt").write_text(res["stderr"] or "")

        df = parse_csv(res["stdout"]) if res["status"] == "ok" else None
        acc = acuracia(df) if df is not None else None
        est = analise_estatica(script.read_text())
        meta = json.loads((rd / "meta.json").read_text()) if (rd / "meta.json").exists() else {}

        wall = [res["wall"]]
        if res["status"] == "ok":
            for _ in range(K - 1):
                wall.append(_executa(script, wrapper)["wall"])

        cam = camada(res, df, acc)
        linha = {"modelo": modelo, "gen": gen, "run": run, "camada": cam, "status": res["status"],
                 "tempo_min_s": round(min(wall), 4), "tempo_med_s": round(statistics.median(wall), 4),
                 "pico_mem_mb": round(res["peak_rss_mb"], 1) if res["peak_rss_mb"] else None,
                 "cpu_s": round(res["cpu_s"], 3) if res["cpu_s"] else None,
                 "tokens": meta.get("tokens"), "gen_s": meta.get("tempo_geracao_s"), **est, **(acc or {})}
        (rd / "avaliacao.json").write_text(json.dumps(linha, indent=2, default=str))   # histórico por run
        linhas.append(linha)
        print(f"{cam} ({linha['tempo_min_s']}s, RMSE={linha.get('rmse', '-')})")

    return pd.DataFrame(linhas)


def consistencia(rdf):
    """Agrega as N runs de cada (modelo, geração) — o eixo da consistência."""
    out = []
    chaves = ["modelo", "gen"] if "gen" in rdf else ["modelo"]
    for chave, g in rdf.groupby(chaves):
        ok = g[g["camada"] == "ok"]
        rmses = ok["rmse"].dropna().tolist() if "rmse" in ok else []
        tempos = ok["tempo_min_s"].dropna().tolist() if "tempo_min_s" in ok else []
        metodos = set(g["metodo"].dropna()) if "metodo" in g else set()
        d = dict(zip(chaves, chave if isinstance(chave, tuple) else (chave,)))
        d.update({"sucesso": f"{len(ok)}/{len(g)}", "mesmo_metodo": len(metodos) == 1,
                  "metodos": ",".join(sorted(metodos)),
                  "rmse_medio": round(np.mean(rmses), 6) if rmses else None,
                  "rmse_desvio": round(np.std(rmses), 6) if len(rmses) > 1 else None,
                  "tempo_desvio_s": round(np.std(tempos), 4) if len(tempos) > 1 else None})
        out.append(d)
    return pd.DataFrame(out)


class _Tee:
    """Escreve na tela E num buffer, para salvar o log completo da rodada."""
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s)
    def flush(self):
        for st in self.streams: st.flush()


def main():
    garantir_scripts()

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    hist = PASTA / "_historico" / ts
    hist.mkdir(parents=True, exist_ok=True)

    buffer = io.StringIO()
    original = sys.stdout
    sys.stdout = _Tee(original, buffer)      # a partir daqui, tudo impresso vai p/ o log também
    try:
        print(f"# Batchmark — rodada {ts}\n")
        resultados = rodar_batchmark()
        print("\n===== RESULTADOS (por modelo x geracao x run) =====")
        cols = ["modelo", "gen", "run", "camada", "metodo", "tempo_min_s", "pico_mem_mb", "rmse", "com_var", "tokens"]
        print(resultados[[c for c in cols if c in resultados]].to_string(index=False))
        cons = consistencia(resultados)
        print("\n===== CONSISTENCIA (as runs de cada geracao) =====")
        print(cons.to_string(index=False))
    finally:
        sys.stdout = original

    # salva histórico (com timestamp) + cópia "latest" na raiz para o analytics
    resultados.to_csv(hist / "resultados.csv", index=False)
    resultados.to_csv(PASTA / "resultados.csv", index=False)
    cons.to_csv(hist / "consistencia.csv", index=False)
    GT.to_csv(hist / "ground_truth.csv", index=False)
    (hist / "log.txt").write_text(buffer.getvalue())

    print(f"\nHistórico desta rodada salvo em: {hist.resolve()}")
    print(f"Resultados 'latest' (p/ o analytics): {(PASTA / 'resultados.csv').resolve()}")


if __name__ == "__main__":
    main()
