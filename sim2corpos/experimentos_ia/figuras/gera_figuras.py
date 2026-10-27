"""Figuras e tabela dos resultados preliminares (Seção 4 do PFCEF I).

Lê só o que já está salvo em experimentos_ia/ e escreve em pfcef1/imagens/:
  prelim-fig1-trajetorias.png   quatro painéis: API direta x2, aplicativo x2
  prelim-fig2-erro-tempo.png    erro de posição do corpo 2 no tempo, escala log
  prelim-fig3-tempo-memoria.png tempo contra memória dos 12 scripts, cor por método
  prelim-tab-desfechos.tex      tabela de desfechos por modelo e por formato

Uso:  python3 gera_figuras.py
"""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
OUT = EXP.parent.parent / "pfcef1" / "imagens"
OUT.mkdir(exist_ok=True)

# Mundo: G=1, m1=1, m2=0.1, r0=1, órbita circular. Sinais conferidos em app_tests/calc_erro.py.
G, M1, M2, R0 = 1.0, 1.0, 0.1, 1.0
M = M1 + M2
W = math.sqrt(G * M / R0**3)
T = 2 * math.pi / W


def exato(t):
    c, s = math.cos(W * t), math.sin(W * t)
    return (-M2 / M * R0 * c, -M2 / M * R0 * s, M1 / M * R0 * c, M1 / M * R0 * s)


def le_traj(caminho):
    rows = list(csv.DictReader(open(caminho)))
    return [(float(r["t"]), float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])) for r in rows]


def erro_corpo2(traj):
    out = []
    for t, x1, y1, x2, y2 in traj:
        _, _, ex2, ey2 = exato(t)
        out.append((t, math.hypot(x2 - ex2, y2 - ey2)))
    return out


def rmse(traj):
    se = n = 0
    for t, x1, y1, x2, y2 in traj:
        e1x, e1y, e2x, e2y = exato(t)
        se += (x1 - e1x) ** 2 + (y1 - e1y) ** 2 + (x2 - e2x) ** 2 + (y2 - e2y) ** 2
        n += 2
    return math.sqrt(se / n)


# ---------------------------------------------------------------- dados
CASOS = [  # (rótulo, arquivo, formato)
    ("Claude Opus 4.8, API direta (jul/2026)", EXP / "mundo_01/anthropic_claude-opus-4.8.csv", "api"),
    ("DeepSeek R1, API direta (jul/2026)", EXP / "mundo_01/deepseek_deepseek-r1.csv", "api"),
    ("Claude Haiku 4.5, aplicativo (jul/2026)", EXP / "app_tests/haiku-4.5_app.csv", "app"),
    ("Claude Opus 4.8, aplicativo (jul/2026)", EXP / "app_tests/opus-4.8_app-desktop.csv", "app"),
]
TRAJ = {rot: le_traj(arq) for rot, arq, _ in CASOS}

NOMES = {
    "anthropic_claude-sonnet-5": "Claude Sonnet 5",
    "deepseek_deepseek-r1": "DeepSeek R1",
    "google_gemini-3.5-flash": "Gemini 3.5 Flash",
    "nvidia_nemotron-3-super-120b-a12b-free": "Nemotron 3 Super 120B (aberto)",
    "google_gemma-4-26b-a4b-it-free": "Gemma 4 26B (aberto)",
    "google_gemma-4-31b-it-free": "Gemma 4 31B (aberto)",
    "nvidia_nemotron-3-nano-omni-30b-a3b-reasoning-free": "Nemotron 3 Nano 30B (aberto)",
}
METODO = {"scipy": "SciPy solve_ivp (adaptativo)", "rk4": "RK4 próprio, passo fixo", "indefinido": "próprio, não identificado"}

SCRIPTS = [r for r in csv.DictReader(open(EXP / "scripts_gerados/resultados.csv")) if r["status"] == "ok"]
for r in SCRIPTS:
    r["traj"] = le_traj(EXP / "scripts_gerados" / r["modelo"] / r["gen"] / r["run"] / "saida_stdout.txt")
    r["rmse_calc"] = rmse(r["traj"])

# ---------------------------------------------------------------- figura 1
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9})
fig, axs = plt.subplots(2, 2, figsize=(7.2, 7.2))
th = [i * 2 * math.pi / 400 for i in range(401)]
for ax, (rot, _, fmt) in zip(axs.flat, CASOS):
    tr = TRAJ[rot]
    ax.plot([M1 / M * R0 * math.cos(a) for a in th], [M1 / M * R0 * math.sin(a) for a in th], "k-", lw=1, label="órbita exata, corpo 2")
    ax.plot([-M2 / M * R0 * math.cos(a) for a in th], [-M2 / M * R0 * math.sin(a) for a in th], "k-", lw=0.6)
    ax.plot([p[3] for p in tr], [p[4] for p in tr], "o--", color="C1", ms=3.5, lw=0.8, label="corpo 2, modelo")
    ax.plot([p[1] for p in tr], [p[2] for p in tr], "s--", color="C0", ms=2.5, lw=0.6, label="corpo 1, modelo")
    ax.annotate("t=17", (tr[-1][3], tr[-1][4]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_aspect("equal")
    ax.set_title(f"{rot}\nerro quadrático médio {rmse(tr):.1e}", fontsize=8)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    lim = max(1.05, max(max(abs(p[3]), abs(p[4])) for p in tr) * 1.08)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
axs[0, 0].legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "prelim-fig1-trajetorias.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- figura 2
fig, ax = plt.subplots(figsize=(7.2, 4.6))
primeiro = True
PISO = 1e-9
for r in SCRIPTS:
    e = erro_corpo2(r["traj"])
    if max(v for _, v in e) < PISO:  # reproduz a fórmula exata, não integra
        ax.plot([t for t, _ in e][1:], [PISO] * (len(e) - 1), ":", color="0.4", lw=1,
                label="1 script que reproduz a fórmula exata (erro abaixo de $10^{-9}$)")
        continue
    ax.plot([t for t, _ in e][1:], [v for _, v in e][1:], "-", color="0.6", lw=0.8,
            label="11 scripts de agosto, 4 modelos" if primeiro else None)
    primeiro = False
ax.axhline(4.1e-6, color="0.3", lw=0.6, ls="-.")
ax.text(0.7, 4.1e-6 * 1.6, "erro das condições iniciais do enunciado, arredondadas a seis casas", fontsize=7, color="0.3")
estilo = {"api": ("C3", "-"), "app": ("C2", "--")}
for rot, _, fmt in CASOS:
    e = erro_corpo2(TRAJ[rot])
    ax.plot([t for t, _ in e][1:], [max(v, 1e-9) for _, v in e][1:], estilo[fmt][1], color=estilo[fmt][0],
            marker="o", ms=3, lw=1.2, label=rot, alpha=0.9 if fmt == "api" else 1)
# distinguir os dois de cada formato
for linha, rot in zip(ax.get_lines()[len(SCRIPTS):], [c[0] for c in CASOS]):
    if "DeepSeek" in rot or "Haiku" in rot:
        linha.set_marker("^")
ax.set_yscale("log")
ax.set_xlabel("tempo (unidade do enunciado; período = 5,99)")
ax.set_ylabel("erro de posição do corpo 2")
ax.set_xlim(0.5, 17.5)
ax.axhline(M1 / M * R0, color="k", lw=0.5, ls=":")
ax.text(17.3, M1 / M * R0 * 1.15, "raio da órbita", ha="right", fontsize=7)
ax.set_ylim(3e-10, 8)
fig.legend(fontsize=7, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(OUT / "prelim-fig2-erro-tempo.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- figura 3
fig, ax = plt.subplots(figsize=(7.2, 4.0))
cores = {"scipy": "C0", "rk4": "C1", "indefinido": "C2"}
marcas = {"anthropic_claude-sonnet-5": "o", "deepseek_deepseek-r1": "s", "google_gemini-3.5-flash": "D",
          "nvidia_nemotron-3-super-120b-a12b-free": "^"}
for r in SCRIPTS:
    ax.scatter(float(r["tempo_med_s"]), float(r["pico_mem_mb"]), c=cores[r["metodo"]], marker=marcas[r["modelo"]],
               s=55, edgecolor="k", lw=0.5, zorder=3)
nem = [r for r in SCRIPTS if r["modelo"].startswith("nvidia_nemotron-3-super")]
for r in nem:
    ax.annotate(f"Nemotron, geração {r['run'][-1]}", (float(r["tempo_med_s"]), float(r["pico_mem_mb"])),
                textcoords="offset points", xytext=(6, -10 if r["metodo"] == "scipy" else 6), fontsize=7)
ax.set_xscale("log")
ax.set_xlabel("tempo decorrido por execução (s), mediana de 3")
ax.set_ylabel("pico de memória (MB)")
from matplotlib.lines import Line2D
h1 = [Line2D([], [], marker="o", color=c, ls="", markeredgecolor="k", label=METODO[m]) for m, c in cores.items()]
h2 = [Line2D([], [], marker=mk, color="w", markeredgecolor="k", ls="", label=NOMES[m]) for m, mk in marcas.items()]
leg1 = ax.legend(handles=h1, title="método (cor)", fontsize=7, title_fontsize=7, loc="center left")
ax.add_artist(leg1)
ax.legend(handles=h2, title="modelo (marcador)", fontsize=7, title_fontsize=7, loc="lower right")
ax.set_ylim(0, 85)
fig.tight_layout()
fig.savefig(OUT / "prelim-fig3-tempo-memoria.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- tabela
status = {r["modelo"]: r for r in csv.DictReader(open(EXP / "mundo_01/status_parcial.csv"))}


def desf_api(chave):
    if chave == "anthropic/claude-sonnet-5":
        return "sem texto: resposta cortada no limite de tokens"
    if chave == "google/gemma-4-31b-it:free":
        return "sem resposta: limite de requisições"
    r = status.get(chave)
    if r is None:
        return "não solicitado"
    if r["deu_numeros"] == "sim":
        return f"tabela errada (erro {float(r['RMSE']):.2f})".replace(".", ",")
    if "provedor" in r["veredito"]:
        return "erro do provedor"
    return "sem tabela: raciocínio cortado no limite de tokens"


def desf_script(chave):
    rs = [r for r in csv.DictReader(open(EXP / "scripts_gerados/resultados.csv")) if r["modelo"] == chave]
    ok = [r for r in rs if r["status"] == "ok"]
    if not ok:
        return "sem resposta: limite de requisições"
    curto = {"scipy": "SciPy", "rk4": "RK4 próprio", "indefinido": "próprio não identificado"}
    met = [curto[r["metodo"]] for r in ok]
    met = sorted(set(met), key=met.index)
    return f"3 de 3 corretos; método: {', '.join(met)}"


LINHAS = [  # nome, chave API julho, desfecho app, chave script agosto
    ("Claude Opus 4.8", "anthropic/claude-opus-4.8", "tabela correta: RK4 próprio, passo $10^{-5}$ (erro $4{,}1\\times10^{-6}$)", None),
    ("Claude Sonnet 5", "anthropic/claude-sonnet-5", "", "anthropic_claude-sonnet-5"),
    ("Claude Haiku 4.5", None, "tabela errada: SciPy, tolerância padrão (erro 0,49)", None),
    ("DeepSeek R1", "deepseek/deepseek-r1", "", "deepseek_deepseek-r1"),
    ("Gemini 3.5 Flash", "google/gemini-3.5-flash", "", "google_gemini-3.5-flash"),
    ("GPT-5.5 Pro", "openai/gpt-5.5-pro", "", None),
    ("gpt-oss-20b (aberto)", "openai/gpt-oss-20b-free", "", None),
    ("Nemotron 3 Super 120B (aberto)", "nvidia/nemotron-3-super-120b-a12b:free", "", "nvidia_nemotron-3-super-120b-a12b-free"),
    ("Nemotron 3 Nano 30B (aberto)", None, "", "nvidia_nemotron-3-nano-omni-30b-a3b-reasoning-free"),
    ("Gemma 4 26B (aberto)", None, "", "google_gemma-4-26b-a4b-it-free"),
    ("Gemma 4 31B (aberto)", "google/gemma-4-31b-it:free", "", "google_gemma-4-31b-it-free"),
]
tex = [
    "\\begin{table}[htbp]",
    "  \\caption{Desfecho de cada modelo nos três formatos dos ensaios iniciais. API direta: tabela pedida sem programa, julho de 2026. Aplicativo: mesmo pedido pela interface do modelo com ferramenta de execução, outubro de 2026. Script: enunciado exigindo programa em Python, três gerações, agosto de 2026.}",
    "  \\begin{tabular}{L{3.0cm}L{3.4cm}L{3.4cm}L{3.4cm}}",
    "    \\toprule",
    "    \\textbf{Modelo} & \\textbf{API direta} & \\textbf{Aplicativo} & \\textbf{Script} \\\\",
    "    \\midrule",
]
for nome, kapi, app, kscr in LINHAS:
    a = desf_api(kapi) if kapi else "não solicitado"
    b = app or "não solicitado"
    c = desf_script(kscr) if kscr else "não solicitado"
    tex.append(f"    {nome} & {a} & {b} & {c} \\\\")
tex += ["    \\bottomrule", "  \\end{tabular}", "  \\label{tab:prelim-desfechos}", "  \\source{Autor}", "\\end{table}"]
(OUT / "prelim-tab-desfechos.tex").write_text("\n".join(tex) + "\n")

# ---------------------------------------------------------------- resumo no terminal
print("figuras em", OUT)
for rot, _, _ in CASOS:
    print(f"  {rot}: RMSE {rmse(TRAJ[rot]):.3e}")
for r in SCRIPTS:
    print(f"  {NOMES[r['modelo']]} {r['run']} {r['metodo']:10s} rmse {r['rmse_calc']:.2e} (csv {float(r['rmse']):.2e}) "
          f"t {float(r['tempo_med_s']):.3f}s mem {r['pico_mem_mb']} MB")
