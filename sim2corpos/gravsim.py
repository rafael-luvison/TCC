"""
gravsim.py — Simulador gravitacional de 2 corpos em 2D, respeitando a lei de Newton.

Filosofia deste módulo (decidida no plano do TCC):

  * O "mundo" simulado só expõe POSIÇÕES ao longo do tempo. É isso que um
    descobridor (humano ou IA) veria para, mais tarde, tentar achar a massa.
  * Grandezas derivadas (energia, momento, centro de massa) NÃO são pistas: elas
    existem só para VALIDAR que o simulador está fazendo a física certa.

Convenções de arrays:
  pos, vel : arrays de forma (2, 2)  ->  linha 0 = corpo 1, linha 1 = corpo 2,
                                        coluna 0 = x, coluna 1 = y.
  massas   : array de forma (2,)     ->  [m1, m2].

Unidades: usamos "unidades naturais" com G configurável (padrão G = 1). Isso mantém
os números em ordens de grandeza amigáveis e é justificado no notebook.
"""

import numpy as np
import pandas as pd

G_PADRAO = 1.0


# ---------------------------------------------------------------------------
# 1) FÍSICA — a lei de força de Newton (o coração do simulador)
# ---------------------------------------------------------------------------
def aceleracoes(pos, massas, G, eps):
    """Acelerações gravitacionais dos 2 corpos.

    Lei de Newton, com *softening* eps para evitar a singularidade 1/r^2 quando
    os corpos passam muito perto:

        a1 = G * m2 * (r2 - r1) / (|r2 - r1|^2 + eps^2)^(3/2)
        a2 = G * m1 * (r1 - r2) / (|r2 - r1|^2 + eps^2)^(3/2)

    Retorna array (2, 2) com a aceleração de cada corpo.
    """
    r1, r2 = pos[0], pos[1]
    d = r2 - r1                      # vetor que aponta do corpo 1 para o corpo 2
    dist2 = d @ d + eps ** 2         # distância^2 "amaciada"
    inv = dist2 ** (-1.5)            # 1 / (dist2)^(3/2)

    a = np.empty((2, 2))
    a[0] = G * massas[1] * d * inv   # corpo 1 é puxado na direção do corpo 2
    a[1] = -G * massas[0] * d * inv  # corpo 2 é puxado na direção do corpo 1
    return a


def energia(pos, vel, massas, G, eps):
    """Energia cinética, potencial e total do sistema.

    O potencial usa a MESMA distância amaciada da força, para que a energia seja
    conservada de forma consistente:

        U = -G * m1 * m2 / sqrt(|r2 - r1|^2 + eps^2)

    Retorna (E_cin, E_pot, E_tot).
    """
    e_cin = 0.5 * np.sum(massas * np.sum(vel ** 2, axis=1))
    d = pos[1] - pos[0]
    dist = np.sqrt(d @ d + eps ** 2)
    e_pot = -G * massas[0] * massas[1] / dist
    return e_cin, e_pot, e_cin + e_pot


def momento_total(vel, massas):
    """Momento linear total do sistema, vetor (px, py)."""
    return np.sum(massas[:, None] * vel, axis=0)


def centro_de_massa(pos, massas):
    """Posição do centro de massa, vetor (x, y)."""
    return np.sum(massas[:, None] * pos, axis=0) / np.sum(massas)


# ---------------------------------------------------------------------------
# 2) INTEGRADORES — como avançar o tempo
# ---------------------------------------------------------------------------
def passo_verlet(pos, vel, massas, G, eps, dt, a=None):
    """Um passo de *velocity Verlet* (integrador simpléctico).

    Reaproveita a aceleração já calculada (`a`) do passo anterior e devolve a
    nova aceleração, para gastar apenas UMA avaliação de força por passo.
    """
    if a is None:
        a = aceleracoes(pos, massas, G, eps)
    pos_novo = pos + vel * dt + 0.5 * a * dt ** 2
    a_novo = aceleracoes(pos_novo, massas, G, eps)
    vel_novo = vel + 0.5 * (a + a_novo) * dt
    return pos_novo, vel_novo, a_novo


def passo_euler(pos, vel, massas, G, eps, dt):
    """Um passo de Euler explícito.

    Existe SÓ para a demonstração didática do notebook: Euler explícito infla a
    energia com o tempo, mostrando por que preferimos o Verlet.
    """
    a = aceleracoes(pos, massas, G, eps)
    pos_novo = pos + vel * dt
    vel_novo = vel + a * dt
    return pos_novo, vel_novo


# ---------------------------------------------------------------------------
# 3) SIMULAÇÃO — roda o mundo e devolve o que é observável + diagnóstico
# ---------------------------------------------------------------------------
def _estado_inicial(config):
    massas = np.array([config["m1"], config["m2"]], dtype=float)
    pos = np.array([config["r1"], config["r2"]], dtype=float)
    vel = np.array([config["v1"], config["v2"]], dtype=float)
    return massas, pos, vel


def simular(config, metodo="verlet"):
    """Roda a simulação do "mundo" de 2 corpos.

    Integra com passo interno fino `dt`, mas só grava (amostra) os dados a cada
    `intervalo_amostragem` segundos.

    Retorna DOIS DataFrames separados de propósito:

      observavel  : colunas [t, x1, y1, x2, y2]  -> É ISTO que o mundo expõe.
      diagnostico : colunas [t, E_cin, E_pot, E_tot, px, py, com_x, com_y]
                    -> só para validar a física; não é pista para o descobridor.
    """
    G = config.get("G", G_PADRAO)
    eps = config.get("eps", 1e-3)
    dt = config["dt"]
    t_total = config["t_total"]
    intervalo = config.get("intervalo_amostragem", 1.0)

    massas, pos, vel = _estado_inicial(config)

    n_passos = int(round(t_total / dt))
    passos_por_amostra = max(1, int(round(intervalo / dt)))

    obs_linhas, diag_linhas = [], []

    def registrar(t, pos, vel):
        obs_linhas.append((t, pos[0, 0], pos[0, 1], pos[1, 0], pos[1, 1]))
        e_cin, e_pot, e_tot = energia(pos, vel, massas, G, eps)
        p = momento_total(vel, massas)
        c = centro_de_massa(pos, massas)
        diag_linhas.append((t, e_cin, e_pot, e_tot, p[0], p[1], c[0], c[1]))

    a = aceleracoes(pos, massas, G, eps)
    registrar(0.0, pos, vel)

    for i in range(1, n_passos + 1):
        if metodo == "verlet":
            pos, vel, a = passo_verlet(pos, vel, massas, G, eps, dt, a)
        elif metodo == "euler":
            pos, vel = passo_euler(pos, vel, massas, G, eps, dt)
        else:
            raise ValueError(f"metodo desconhecido: {metodo!r} (use 'verlet' ou 'euler')")

        if i % passos_por_amostra == 0:
            registrar(i * dt, pos, vel)

    observavel = pd.DataFrame(obs_linhas, columns=["t", "x1", "y1", "x2", "y2"])
    diagnostico = pd.DataFrame(
        diag_linhas,
        columns=["t", "E_cin", "E_pot", "E_tot", "px", "py", "com_x", "com_y"],
    )
    return observavel, diagnostico


# ---------------------------------------------------------------------------
# 4) CONDIÇÃO INICIAL DE CONVENIÊNCIA — órbita (quase) circular
# ---------------------------------------------------------------------------
def config_orbita_circular(m1=1.0, m2=0.1, r=1.0, G=1.0, eps=1e-3,
                           dt=1e-3, n_orbitas=3.0, intervalo_amostragem=1.0):
    """Monta um `config` de órbita circular no referencial do centro de massa.

    Os corpos ficam em lados opostos do centro de massa (na origem) e recebem
    velocidades que:
      (a) dão uma órbita relativa circular de separação `r`, com
          v_rel = sqrt(G * M / r), onde M = m1 + m2;
      (b) deixam o centro de massa em repouso (m1*v1 + m2*v2 = 0).

    `t_total` é escolhido para cobrir `n_orbitas` períodos de Kepler.
    """
    M = m1 + m2
    x1 = -(m2 / M) * r          # corpo 1 à esquerda do COM
    x2 = (m1 / M) * r           # corpo 2 à direita do COM

    v_rel = np.sqrt(G * M / r)
    v1y = -(m2 / M) * v_rel     # velocidades opostas -> COM parado
    v2y = (m1 / M) * v_rel

    T = 2 * np.pi * np.sqrt(r ** 3 / (G * M))   # período de Kepler
    return {
        "G": G, "m1": m1, "m2": m2,
        "r1": [float(x1), 0.0], "r2": [float(x2), 0.0],
        "v1": [0.0, float(v1y)], "v2": [0.0, float(v2y)],
        "eps": eps, "dt": dt,
        "t_total": float(n_orbitas * T),
        "intervalo_amostragem": intervalo_amostragem,
    }


# ---------------------------------------------------------------------------
# 5) VALIDAÇÃO — tabelas que provam que o simulador funciona
# ---------------------------------------------------------------------------
def tabela_conservacao(diagnostico):
    """Resume a conservação de energia, momento e centro de massa.

    A coluna `deriva_E_rel` é o desvio relativo da energia total em relação ao
    instante inicial — deve ficar próximo de zero num integrador simpléctico.
    """
    d = diagnostico
    E0 = d["E_tot"].iloc[0]
    return pd.DataFrame({
        "t": d["t"],
        "E_tot": d["E_tot"],
        "deriva_E_rel": (d["E_tot"] - E0) / abs(E0),
        "|p|": np.hypot(d["px"], d["py"]),
        "com_x": d["com_x"],
        "com_y": d["com_y"],
    })


def resumo_conservacao(diagnostico):
    """Uma linha-resumo com os piores desvios ao longo de todo o run."""
    d = diagnostico
    E0 = d["E_tot"].iloc[0]
    deriva_E = np.abs((d["E_tot"] - E0) / abs(E0)).max()
    p = np.hypot(d["px"], d["py"])
    deriva_p = (p.max() - p.min())
    var_com = np.hypot(d["com_x"].max() - d["com_x"].min(),
                       d["com_y"].max() - d["com_y"].min())
    return pd.DataFrame([{
        "deriva_E_rel_max": deriva_E,
        "variacao_|p|": deriva_p,
        "variacao_COM": var_com,
    }])


def checar_kepler(config, observavel):
    """Compara o período orbital MEDIDO (das posições) com o teórico de Kepler.

    O período é medido pela velocidade angular do vetor separação (r2 - r1):
    ajusta-se uma reta a theta(t) (ângulo desenrolado) e T = 2*pi / |omega|.
    Para uma órbita ~circular, a separação média aproxima o semieixo maior `a`.
    """
    G = config.get("G", G_PADRAO)
    M = config["m1"] + config["m2"]

    dx = (observavel["x2"] - observavel["x1"]).to_numpy()
    dy = (observavel["y2"] - observavel["y1"]).to_numpy()
    a = np.hypot(dx, dy).mean()
    T_teorico = 2 * np.pi * np.sqrt(a ** 3 / (G * M))

    theta = np.unwrap(np.arctan2(dy, dx))
    t = observavel["t"].to_numpy()
    omega = np.polyfit(t, theta, 1)[0]
    T_medido = 2 * np.pi / abs(omega)

    return {
        "T_teorico": T_teorico,
        "T_medido": T_medido,
        "erro_rel": abs(T_medido - T_teorico) / T_teorico,
        "separacao_media": a,
    }


# ---------------------------------------------------------------------------
# 6) GRÁFICOS
# ---------------------------------------------------------------------------
def plot_orbita(observavel, ax=None):
    """Desenha as trajetórias observáveis dos 2 corpos (com marcadores de início)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.plot(observavel["x1"], observavel["y1"], "-", lw=1, label="corpo 1")
    ax.plot(observavel["x2"], observavel["y2"], "-", lw=1, label="corpo 2")
    ax.plot(observavel["x1"].iloc[0], observavel["y1"].iloc[0], "o", color="C0")
    ax.plot(observavel["x2"].iloc[0], observavel["y2"].iloc[0], "o", color="C1")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Órbitas (posições observáveis)")
    ax.legend()
    return ax


def plot_energia(diagnostico, ax=None):
    """Energia cinética, potencial e total ao longo do tempo."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(diagnostico["t"], diagnostico["E_cin"], label="cinética")
    ax.plot(diagnostico["t"], diagnostico["E_pot"], label="potencial")
    ax.plot(diagnostico["t"], diagnostico["E_tot"], label="total", lw=2, color="k")
    ax.set_xlabel("t")
    ax.set_ylabel("energia")
    ax.set_title("Conservação de energia")
    ax.legend()
    return ax


def plot_momento(diagnostico, ax=None):
    """Módulo do momento linear total ao longo do tempo."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    p = np.hypot(diagnostico["px"], diagnostico["py"])
    ax.plot(diagnostico["t"], p)
    ax.set_xlabel("t")
    ax.set_ylabel("|p|")
    ax.set_title("Conservação do momento linear")
    return ax
