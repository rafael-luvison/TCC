"""
Batchmark — FASE 1: O PEDINTE.

Pede a cada modelo de IA escolhido um SCRIPT Python que resolve o problema de 2
corpos e imprime o CSV `t,x1,y1,x2,y2` no stdout. Pede N_REP vezes por modelo, em
requisições INDEPENDENTES (sem histórico → sem cache de conversa), para medir a
consistência do código gerado.

Uso:
    - normalmente NÃO se roda direto: a Fase 2 chama `rodar_fase1()` quando precisa;
    - mas dá para rodar sozinho:  python fase1_pedinte.py
"""
import os
import re
import json
import time
import getpass
import datetime
from pathlib import Path

import requests

# Âncora: tudo é relativo a ESTE arquivo, não ao diretório de trabalho — assim
# funciona não importa de onde você rode (raiz do TCC, sim2corpos, VS Code, etc.).
_AQUI = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

try:  # carrega o .env que está AO LADO deste arquivo
    from dotenv import load_dotenv
    load_dotenv(_AQUI / ".env")
except Exception:
    pass

URL_API = "https://openrouter.ai/api/v1"
PASTA = _AQUI / "experimentos_ia" / "scripts_gerados"   # pasta dos scripts gerados por IA
N_REP = 3                                          # gerações independentes por modelo

# Candidatos PAGOS de fronteira (edite para incluir outros). Os grátis (:free) são
# descobertos ao vivo no catálogo; ambos entram no menu de escolha.
PAGOS = [
    "anthropic/claude-opus-4.8",
    "anthropic/claude-sonnet-5",
    "openai/gpt-5.2",
    "google/gemini-3.5-flash",
    "deepseek/deepseek-r1",
]

# O "mundo" (as mesmas condições iniciais das Partes 8/10).
MUNDO = {
    "G": 1.0, "m1": 1.0, "m2": 0.1,
    "r1": (-0.090909, 0.0), "v1": (0.0, -0.095346),
    "r2": (0.909091, 0.0),  "v2": (0.0, 0.953463),
    "t_final": 17.0, "dt_amostra": 1.0, "n_linhas": 18,
}


def get_chave():
    """Chave do OpenRouter: Colab (userdata) -> variável de ambiente -> digitar."""
    ch = ""
    try:
        from google.colab import userdata
        ch = userdata.get("OPENROUTER_API_KEY")
    except Exception:
        pass
    if not ch:
        ch = os.environ.get("OPENROUTER_API_KEY", "")
    if not ch or ch.startswith("cole-sua-chave"):
        ch = getpass.getpass("Cole sua chave do OpenRouter (sk-or-...): ")
    return ch


def catalogo():
    """Lista de modelos do OpenRouter (endpoint público, não gasta chamadas)."""
    return requests.get(f"{URL_API}/models", timeout=30).json()["data"]


def escolher_modelos():
    """Mostra os modelos GRÁTIS e PAGOS disponíveis agora e deixa o usuário escolher."""
    cat = catalogo()
    ids = {m["id"] for m in cat}
    livres = sorted(m["id"] for m in cat if m["id"].endswith(":free")
                    and not any(p in m["id"] for p in ["image", "safety", "guard", "vl", "vision"]))
    pagos = [m for m in PAGOS if m in ids]        # só os candidatos que existem no catálogo

    opcoes = [(m, "grátis") for m in livres] + [(m, "pago ") for m in pagos]
    print("Modelos disponíveis agora (grátis contam no limite diário; pagos usam crédito):\n")
    for i, (m, tag) in enumerate(opcoes, 1):
        print(f"  {i:2d}. [{tag}] {m}")

    sel = input("\nQuais quer puxar? (ex: 1,3,5 | 1-4 | all) > ").strip().lower()

    def parse(s, n):
        if s in ("", "all"):
            return list(range(1, n + 1))
        idx = set()
        for p in s.replace(" ", "").split(","):
            if "-" in p:
                a, b = p.split("-"); idx.update(range(int(a), int(b) + 1))
            elif p.isdigit():
                idx.add(int(p))
        return sorted(i for i in idx if 1 <= i <= n)

    escolhidos = [opcoes[i - 1][0] for i in parse(sel, len(opcoes))]
    print("\nEscolhidos:", escolhidos)
    return escolhidos


def montar_prompt_script(m):
    return f"""Escreva um programa Python COMPLETO e autossuficiente que resolva o \
seguinte problema de dois corpos com gravitacao newtoniana em 2D, em unidades \
adimensionais com G = {m['G']}.

Corpo 1: massa m1 = {m['m1']}, posicao inicial = {m['r1']}, velocidade inicial = {m['v1']}
Corpo 2: massa m2 = {m['m2']}, posicao inicial = {m['r2']}, velocidade inicial = {m['v2']}

Requisitos do programa:
- Ao ser executado (python script.py), deve calcular as posicoes dos dois corpos de \
t=0 a t={m['t_final']:.0f} e IMPRIMIR no stdout uma tabela CSV com exatamente estas \
colunas e {m['n_linhas']} linhas de dados (t = 0, 1, 2, ..., {m['t_final']:.0f}):
t,x1,y1,x2,y2
- Use apenas a biblioteca padrao do Python e/ou numpy e scipy (nada que precise instalar).
- O metodo de resolucao e a sua escolha.
- Responda APENAS com o codigo Python, dentro de um unico bloco ```python ... ```, \
sem nenhum texto antes ou depois."""


# códigos transitórios que valem a pena tentar de novo (429 = taxa; 5xx = provedor)
RETRIAVEIS = {408, 409, 429, 500, 502, 503, 504}


def perguntar_ia(prompt, modelo, chave, max_tokens=16000, timeout=600, tentativas=5):
    """Chama o modelo (SEM temperature=0: queremos variação p/ medir consistência).

    Faz retry com backoff tanto em erros de STATUS HTTP quanto em erros que o
    OpenRouter devolve como 200 com um campo `error` (ex.: 502 upstream da NVIDIA).
    """
    headers = {"Authorization": f"Bearer {chave}", "X-Title": "TCC batchmark"}
    corpo = {"model": modelo, "messages": [{"role": "user", "content": prompt}],
             "max_tokens": max_tokens}
    for t in range(tentativas):
        try:
            r = requests.post(f"{URL_API}/chat/completions", headers=headers, json=corpo, timeout=timeout)
        except requests.RequestException as e:            # rede/timeout também é transitório
            if t < tentativas - 1:
                print(f"[rede, retry] ", end="", flush=True); time.sleep(min(60, 5 * 2 ** t)); continue
            raise

        try:
            d = r.json()
        except Exception:
            d = {}
        # o código de erro pode estar no status HTTP OU no corpo ({"error":{"code":...}})
        cod = r.status_code
        if isinstance(d.get("error"), dict) and d["error"].get("code"):
            cod = d["error"]["code"]

        if cod in RETRIAVEIS and t < tentativas - 1:
            esp = float(r.headers.get("Retry-After", min(60, 5 * 2 ** t)))
            print(f"[{cod}, retry em {esp:.0f}s] ", end="", flush=True); time.sleep(esp); continue

        r.raise_for_status()                               # erro HTTP não-retriável
        if d.get("error"):                                 # 200 com erro não-retriável
            raise RuntimeError(str(d["error"]))
        msg = d["choices"][0]["message"]
        texto = msg.get("content") or msg.get("reasoning") or ""
        if not texto.strip():
            raise RuntimeError(f"vazio (finish={d['choices'][0].get('finish_reason')})")
        return texto, d.get("usage", {})
    raise RuntimeError("excedeu as tentativas (erro transitório persistente)")


def extrair_codigo(texto):
    blocos = re.findall(r"```(?:python|py)?\s*\n(.*?)```", texto, re.S)
    return (max(blocos, key=len) if blocos else texto).strip()


def proxima_gen(slug):
    """Próximo número de geração p/ este modelo (nunca sobrescreve os anteriores)."""
    base = PASTA / slug
    if not base.exists():
        return 1
    gens = [int(p.name[3:]) for p in base.glob("gen*") if p.is_dir() and p.name[3:].isdigit()]
    return max(gens, default=0) + 1


def rodar_fase1(modelos=None):
    """Gera e salva os scripts, VERSIONADOS por geração. Se `modelos` for None, pergunta."""
    chave = get_chave()
    if modelos is None:
        modelos = escolher_modelos()
    if not modelos:
        print("Nenhum modelo escolhido — nada a fazer."); return

    PASTA.mkdir(parents=True, exist_ok=True)
    prompt = montar_prompt_script(MUNDO)
    (PASTA / "prompt.txt").write_text(prompt)
    (PASTA / "mundo.json").write_text(json.dumps(MUNDO, indent=2, default=list))
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    for modelo in modelos:
        slug = modelo.replace("/", "_").replace(":", "-")
        gen = proxima_gen(slug)                      # 1ª vez -> gen01; depois gen02, gen03...
        gtag = f"gen{gen:02d}"
        print(f"\n== {modelo}  [{gtag}] ==")
        for k in range(1, N_REP + 1):
            destino = PASTA / slug / gtag / f"run{k}"
            destino.mkdir(parents=True, exist_ok=True)
            print(f"-> {gtag}/run{k} ... ", end="", flush=True)
            try:
                t0 = time.time()
                texto, uso = perguntar_ia(prompt, modelo, chave)   # requisição nova = sem cache de conversa
                dt = time.time() - t0
                codigo = extrair_codigo(texto)
                (destino / "resposta_raw.txt").write_text(texto)
                (destino / "script.py").write_text(codigo)
                (destino / "meta.json").write_text(json.dumps({
                    "modelo": modelo, "gen": gen, "run": k, "timestamp": ts,
                    "tempo_geracao_s": round(dt, 2), "tokens": uso.get("total_tokens"),
                    "linhas_codigo": codigo.count("\n") + 1,
                }, indent=2))
                print(f"ok ({codigo.count(chr(10)) + 1} linhas, {dt:.1f}s)")
            except Exception as e:
                (destino / "erro.txt").write_text(str(e))
                print(f"FALHOU: {e}")
            time.sleep(4)   # respeita o limite de 20 req/min dos grátis
    print(f"\nScripts salvos em: {PASTA.resolve()}")


if __name__ == "__main__":
    rodar_fase1()
