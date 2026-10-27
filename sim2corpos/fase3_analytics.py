"""
Batchmark — FASE 3: ANALYTICS.

Lê o `resultados.csv` (o mais recente da Fase 2) e gera gráficos comparáveis.

Princípio de leitura: a UNIDADE é a "amostra" = uma execução de um script gerado
= (modelo, geração, run). Cada amostra tem um ID curto e único (ex.: nemotron-S·g1r3)
que aparece IGUAL em todos os painéis. A COR = método (legenda única). Assim você lê a
MESMA linha atravessando RMSE -> tempo -> memória e sabe exatamente qual amostra é e
qual método usou. Acompanha uma TABELA-LEGENDA MESTRE (PNG + CSV).

Rode:  python fase3_analytics.py
"""
import re
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba

import fase1_pedinte as f1

PASTA = f1.PASTA

# abreviações curtas e estáveis por modelo (fallback: último trecho do slug)
ABREV = {
    "anthropic_claude-opus-4.8": "opus", "anthropic_claude-sonnet-5": "sonnet",
    "openai_gpt-5.2": "gpt5.2", "openai_gpt-5.5-pro": "gpt5.5",
    "google_gemini-3.5-flash": "gemini", "deepseek_deepseek-r1": "deepseek",
    "nvidia_nemotron-3-super-120b-a12b-free": "nemotron-S",
    "nvidia_nemotron-3-ultra-550b-a55b-free": "nemotron-U",
    "nvidia_nemotron-3-nano-omni-30b-a3b-reasoning-free": "nemotron-N",
    "google_gemma-4-26b-a4b-it-free": "gemma-26", "google_gemma-4-31b-it-free": "gemma-31",
}
COR_METODO = {"scipy": "#1f77b4", "rk4": "#ff7f0e", "verlet": "#2ca02c",
              "analitico": "#9467bd", "euler": "#8c564b", "indefinido": "#7f7f7f"}
COR_CAMADA = {"ok": "#2ca02c", "3-impreciso": "#ff7f0e", "2-formato": "#9467bd",
              "1-crash": "#d62728", "1-timeout": "#8c564b", "0-sem_script": "#bbbbbb"}


def _abrev(slug):
    return ABREV.get(slug, slug.split("_")[-1][:12])


def _num(x):
    return re.sub(r"\D", "", str(x)).lstrip("0") or "0"


def _id(row):
    return f"{_abrev(row['modelo'])}·g{_num(row['gen'])}r{_num(row['run'])}"


def carregar(caminho=None):
    csv = Path(caminho) if caminho else PASTA / "resultados.csv"
    df = pd.read_csv(csv)
    if "gen" not in df: df["gen"] = "gen01"
    df["id"] = df.apply(_id, axis=1)
    df = df.sort_values(["modelo", "gen", "run"]).reset_index(drop=True)
    return df


def _dotplot(ax, df, coluna, titulo, log=False):
    """Um ponto por amostra (linha = ID), cor = método; anota valor e falhas."""
    y = np.arange(len(df))
    for i, row in df.iterrows():
        val = row.get(coluna)
        cor = COR_METODO.get(row.get("metodo"), "#cccccc")
        if pd.notna(val) and val != "" and row["camada"] == "ok":
            ax.plot([val], [i], "o", ms=9, color=cor, zorder=3)
            ax.annotate(f"{val:.2e}" if log else f"{val:.3g}", (val, i),
                        textcoords="offset points", xytext=(8, 0), va="center", fontsize=7)
        else:                                   # amostra que não rodou: marca a camada
            ax.text(0.02, i, row["camada"], transform=ax.get_yaxis_transform(),
                    va="center", fontsize=7, color=COR_CAMADA.get(row["camada"], "#999"), style="italic")
    if log:
        ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(df["id"], fontsize=8)
    ax.invert_yaxis(); ax.grid(axis="x", ls=":", alpha=0.5)
    ax.set_title(titulo)


def gerar(caminho_csv=None):
    csv = Path(caminho_csv) if caminho_csv else PASTA / "resultados.csv"
    if not csv.exists():
        print(f"Não achei {csv}. Rode a Fase 2 primeiro."); return
    df = carregar(csv)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    destino = PASTA / "analytics" / ts
    destino.mkdir(parents=True, exist_ok=True)
    n = len(df)

    # ---- PAINEL DE MÉTRICAS: 3 colunas, MESMA ordem de amostras (y = ID) ----
    fig, ax = plt.subplots(1, 3, figsize=(16, max(4, 0.42 * n)), sharey=True)
    _dotplot(ax[0], df, "rmse", "Acurácia — RMSE (log, ← melhor)", log=True)
    _dotplot(ax[1], df, "tempo_min_s", "Tempo mínimo (s)")
    _dotplot(ax[2], df, "pico_mem_mb", "Pico de memória (MB)")

    metodos_presentes = [m for m in COR_METODO if m in set(df["metodo"].dropna())]
    leg = [Line2D([0], [0], marker="o", ls="", color=COR_METODO[m], label=m) for m in metodos_presentes]
    fig.legend(handles=leg, title="método (cor)", loc="lower center", ncol=len(leg), fontsize=9)
    fig.suptitle(f"Batchmark — cada linha é uma amostra (modelo·geração·run) — {ts}", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    p_metricas = destino / "painel_metricas.png"
    fig.savefig(p_metricas, dpi=120); plt.close(fig)

    # ---- TABELA-LEGENDA MESTRE (PNG + CSV) ---------------------------------
    cols = ["id", "modelo", "gen", "run", "metodo", "camada", "rmse", "tempo_min_s", "pico_mem_mb", "tokens"]
    tab = df[[c for c in cols if c in df]].copy()
    tab.to_csv(destino / "tabela_mestre.csv", index=False)

    figt, axt = plt.subplots(figsize=(15, max(2.5, 0.4 * n + 1)))
    axt.axis("off")
    texto = tab.copy()
    if "modelo" in texto:                                       # evita o slug longo invadir a coluna vizinha
        texto["modelo"] = texto["modelo"].map(lambda s: s if len(str(s)) <= 20 else str(s)[:19] + "…")
    if "rmse" in texto: texto["rmse"] = texto["rmse"].map(lambda v: f"{v:.2e}" if pd.notna(v) else "—")
    for c in ["tempo_min_s", "pico_mem_mb", "tokens"]:
        if c in texto: texto[c] = texto[c].map(lambda v: f"{v:.3g}" if pd.notna(v) else "—")
    t = axt.table(cellText=texto.values, colLabels=texto.columns, cellLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.4)
    for j, col in enumerate(texto.columns):                      # colore a célula 'camada' e 'metodo'
        for i in range(len(texto)):
            cell = t[i + 1, j]
            if col == "camada" and texto.iloc[i]["camada"] in COR_CAMADA:
                cell.set_facecolor(to_rgba(COR_CAMADA[texto.iloc[i]["camada"]], 0.55))
            if col == "metodo" and texto.iloc[i]["metodo"] in COR_METODO:
                cell.set_facecolor(to_rgba(COR_METODO[texto.iloc[i]["metodo"]], 0.33))
    axt.set_title(f"Tabela-legenda mestre — {ts}", fontsize=12, pad=12)
    figt.tight_layout()
    p_tabela = destino / "tabela_mestre.png"
    figt.savefig(p_tabela, dpi=120); plt.close(figt)

    # ---- resumo ------------------------------------------------------------
    ok = df[df["camada"] == "ok"]
    resumo = [f"# Resumo do batchmark — {ts}", "",
              f"- Amostras: **{n}** | rodaram OK: **{len(ok)}**"]
    if len(ok):
        b = ok.loc[ok["rmse"].idxmin()]; resumo.append(f"- Menor RMSE: **{b['id']}** ({b['rmse']:.2e}, {b['metodo']})")
        r = ok.loc[ok["tempo_min_s"].idxmin()]; resumo.append(f"- Mais rápido: **{r['id']}** ({r['tempo_min_s']:.3f}s, {r['metodo']})")
        l = ok.loc[ok["pico_mem_mb"].idxmin()]; resumo.append(f"- Menos memória: **{l['id']}** ({l['pico_mem_mb']:.0f} MB, {l['metodo']})")
    (destino / "resumo.md").write_text("\n".join(resumo) + "\n")

    print(f"Painel de métricas: {p_metricas.resolve()}")
    print(f"Tabela mestre:      {p_tabela.resolve()}  (+ .csv)")
    print("\n".join(resumo))
    return p_metricas, p_tabela


if __name__ == "__main__":
    gerar()
