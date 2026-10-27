# Benchmark de modelos de linguagem no problema de dois corpos

Material do Projeto Final de Curso em Engenharia Física (UFRGS) de Rafael Luvison, orientado pela Profa. Dra. Carine Priscila Beatrici. O trabalho propõe um *benchmark* para medir a confiabilidade de modelos de linguagem de grande porte na escrita de programas que resolvem, por integração numérica, o problema gravitacional de dois corpos.

Este repositório contém apenas o que o texto do projeto cita: o simulador, o executor, os dados brutos dos ensaios iniciais e o código que gerou as figuras da Seção 4 (Resultados preliminares).

## Estrutura

```
sim2corpos/
  gravsim.py              simulador de dois corpos (velocity Verlet); gera o gabarito ground_truth.csv
  fase1_pedinte.py        pede os scripts aos modelos pela OpenRouter, em solicitações independentes
  fase2_avaliador.py      executor: roda cada script em subprocesso isolado e mede tempo, memória e erro
  fase3_analytics.py      leitura do resultados.csv e painéis exploratórios
  requirements.txt
  experimentos_ia/
    mundo_01/             ensaio de julho de 2026 pela API, sem exigir programa (Seção 4.1, Apêndice C)
      prompt.txt            enunciado enviado
      *_raw.txt             respostas integrais de cada modelo
      *.csv                 tabelas devolvidas; simulador.csv é o gabarito
      resumo_chamadas.csv   tokens e tempo de cada chamada
      status_parcial.csv    desfecho por modelo
    app_tests/            mesmo enunciado pelos aplicativos de dois modelos, julho de 2026 (Seção 4.1)
    scripts_gerados/      ensaio de agosto de 2026 com exigência de programa (Seção 4.2)
      prompt.txt            enunciado enviado
      <modelo>/gen01/runN/  script.py, resposta_raw.txt, meta.json, saída e avaliação de cada geração
      resultados.csv        tempo, memória, método e erro de cada execução
      ground_truth.csv      gabarito do simulador
      _wrapper.py           mede pico de memória e tempo de CPU de cada execução
    figuras/
      gera_figuras.py       gera as três figuras e a tabela da Seção 4 a partir dos dados acima
      README.md             o que cada figura mostra e dois achados feitos ao gerá-las
pfcef1/imagens/           figuras e tabela geradas (prelim-*)
```

## Como reproduzir as figuras

```bash
cd sim2corpos
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 experimentos_ia/figuras/gera_figuras.py
```

As saídas vão para `pfcef1/imagens/`.

## Como repetir a coleta

`fase2_avaliador.py` roda os scripts já salvos. Para pedir scripts novos, `fase1_pedinte.py` lê a chave da OpenRouter da variável de ambiente `OPENROUTER_API_KEY` ou de um arquivo `.env` (ignorado pelo Git).

## Citação

LUVISON, Rafael. Benchmark de modelos de linguagem no problema de dois corpos: scripts, dados e figuras. 2026. Disponível em: https://github.com/rafael-luvison/TCC.
