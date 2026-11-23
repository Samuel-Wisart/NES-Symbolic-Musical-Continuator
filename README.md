# TP2 - AI Music Generation
Projeto de geração musical autoregressiva usando o dataset NES-MDB. A tarefa escolhida é composição livre, comparando uma cadeia de Markov (baseline) com um Transformer.

## Estrutura
```text
data/
  raw/                         # NES-MDB original
  processed/                   # dados tokenizados
results/                       # resultados dos experimentos
scripts/
  step01_prepare_dataset.py    # preparação e tokenização
  step02_analyze_dataset.py    # estatísticas do dataset e da representação
tests/
  test01_dataset_tests.py      # testes da representação
```

## Ambiente
Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Dataset
Baixe o NES-MDB em formato MIDI e coloque o arquivo em:
```text
data/raw/nesmdb_midi.tar.gz
```

O projeto utiliza as partições oficiais de treino, validação e teste do dataset.

## Preparação
```bash
python scripts/step01_prepare_dataset.py
```

O script extrai os arquivos MIDI e converte cada música para uma sequência MIDI-Like.
A representação usa `NOTE_ON`, `NOTE_OFF` e `TIME_SHIFT`, preservando as quatro vozes do NES-MDB (`P1`, `P2`, `TR` e `NO`) e usando quantização temporal de 64 passos por segundo.

Os dados processados são salvos em:

```text
data/processed/nesmdb/
  tokens/
    train/
    valid/
    test/
  token_index.csv
  vocab.json
```

## Análise
Após preparar o dataset:

```bash
python scripts/step02_analyze_dataset.py
```

O script reporta as estatísticas básicas do dataset e da representação, incluindo quantidade de músicas e notas, tamanho do vocabulário e comprimento das sequências.

## Testes
Rode qualquer teste com:
```bash
python tests/[00_nome_do_teste].py
```
Os testes disponíveis são:

### test01_dataset_tests.py
Valida os pontos essenciais da representação MIDI-Like antes do processamento completo do dataset, incluindo vocabulário, `encode -> decode`, quantização e `TIME_SHIFT`.