# Figuras dos resultados preliminares (PFCEF I, Seção 4)

`gera_figuras.py` lê só o que já está salvo em `experimentos_ia/` e escreve em `pfcef1/imagens/`:

| arquivo | conteúdo | dados de origem |
|---|---|---|
| `prelim-fig1-trajetorias.png` | 4 painéis: órbita exata (círculo contínuo) e pontos do modelo | `mundo_01/*.csv` (API direta, jul/2026), `app_tests/*.csv` (aplicativos, jul/2026) |
| `prelim-fig2-erro-tempo.png` | erro de posição do corpo 2 no tempo, escala log; 4 casos acima + 12 scripts de agosto | idem + `scripts_gerados/*/gen01/run*/saida_stdout.txt` |
| `prelim-fig3-tempo-memoria.png` | tempo contra pico de memória dos 12 scripts, cor por método, marcador por modelo | `scripts_gerados/resultados.csv` |
| `prelim-tab-desfechos.tex` | tabela LaTeX: desfecho por modelo e por formato | `mundo_01/status_parcial.csv`, `mundo_01/resumo_chamadas.csv`, `app_tests/`, `resultados.csv` |

Gabarito: solução analítica da órbita circular ideal (G=1, m1=1, m2=0,1, r0=1, ω=√1,1), mesma convenção de sinais de `app_tests/calc_erro.py`.

## Dois achados ao gerar as figuras (23/10/2026)

1. **Patamar de 4,1e-6 nos scripts de agosto.** Contra a órbita ideal, 11 dos 12 scripts dão erro quadrático médio de 4,05e-6 a 4,10e-6, enquanto a avaliação de agosto (contra o simulador `ground_truth.csv`) dava 2,5e-8 a 4e-7. Causa confirmada: as condições iniciais do enunciado têm seis casas decimais (v2 = 0,953463 contra o ideal 0,95346259, diferença relativa 4,3e-7). Integrando essas condições arredondadas com RK4 de passo 1e-4, o erro contra a órbita ideal é 4,10e-6. Logo o patamar vem do enunciado, não dos integradores. Consequência para a Metodologia: o gabarito deve ser a solução exata das condições iniciais efetivamente enviadas (órbita de Kepler com e ≈ 1e-6), ou o enunciado deve dar as condições com precisão de máquina.
2. **Um script reproduz a fórmula fechada.** Nemotron 3 Super, geração 1: erro 4e-13 contra a órbita ideal. O `script.py` calcula x = r·cos(ωt), y = r·sin(ωt), sem integrar. É o risco de contaminação descrito na Fundamentação. O enunciado de agosto (`scripts_gerados/prompt.txt`) não exigia integração numérica: dizia "o método de resolução é a sua escolha". A Metodologia v2 já diz que o enunciado final exige solução numérica; além disso, o executor precisa detectar a forma fechada (por exemplo, exigir que o script aceite condições iniciais arbitrárias e testá-lo com uma órbita elíptica).

Rodar: `python3 gera_figuras.py` (matplotlib 3.9, sem SciPy).
