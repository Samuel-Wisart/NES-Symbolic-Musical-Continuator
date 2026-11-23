# TP2 - AI Music Generation
Projeto de geração musical autoregressiva usando o dataset NES-MDB. A tarefa escolhida é composição livre, comparando uma cadeia de Markov (baseline) com um Transformer.

## Estrutura
```text
data/
  raw/                         # NES-MDB original
  processed/                   # dados tokenizados
nes_continuation/              # Transformer: modelo, treino, geração e áudio
  representation.py            # ponte para a representação do step01
results/                       # resultados dos experimentos
  transformer/                 # métricas, figura e README do Transformer
  generated/                   # continuações (transformer/, reference/, ...)
runs/                          # checkpoints e logs de treino (não versionado)
scripts/
  protocol.py                  # protocolo de avaliação e geração (Transformer e Markov)
  step01_prepare_dataset.py    # preparação e tokenização
  step02_analyze_dataset.py    # estatísticas do dataset e da representação
  step03_train_transformer.py  # treino do Transformer
  step04_evaluate_transformer.py  # perplexidade e acurácia no teste
  step05_generate_transformer.py  # continuações do protocolo + referência real
  step06_report_transformer.py    # figura de loss e results/transformer/README.md
tests/
  test01_dataset_tests.py      # testes da representação
  test02_transformer_tests.py  # testes do Transformer sobre os tokens do step01
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

### test02_transformer_tests.py
Valida a adaptação do Transformer aos tokens do step01 (requer o dataset preparado): `.npz` igual a `encode(read_notes(mid))`, corte do prompt em exatamente N segundos, `decode -> MIDI -> read_notes -> encode`, cobertura das janelas, `forward` com os 1155 tokens e respeito à máscara de eventos válidos na geração.

## Transformer
O pacote `nes_continuation/` (Transformer decoder-only em PyTorch) usa a representação do step01: importa `VOCAB`, `TO_ID`, `encode`, `decode` e `read_notes` de lá e usa os 1155 tokens. Lê os `.npz` de `data/processed/nesmdb/tokens/`, então o dataset precisa estar preparado (step01) também na máquina de treino.

### GPU
O `torch` do PyPI no Windows é só CPU. Instale a versão com CUDA indicada em https://pytorch.org/get-started/locally/ e confira:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

O treino usa BF16 nas GPUs que suportam e FP16 (com `GradScaler`) nas demais.

### 1. Medir o tempo por época
Roda alguns lotes, estima o tempo de uma época e sai sem salvar nada. Experimente lotes maiores se couber na memória:

```bash
python scripts/step03_train_transformer.py --benchmark 30 --batch-size 8
python scripts/step03_train_transformer.py --benchmark 30 --batch-size 32
```

### 2. Treinar
Configuração padrão: contexto 1024, 6 camadas, largura 256, 8 cabeças, dropout 0.1, AdamW 3e-4, lote 8. A cada época são salvos `runs/transformer/last.pt`, `best.pt` (menor loss de validação), `metrics.json` e `run_info.json`.

Em segundo plano (Linux), com log em `runs/transformer/train.log`:

```bash
mkdir -p runs/transformer
nohup python -u scripts/step03_train_transformer.py --epochs 20 --batch-size 8 >> runs/transformer/train.log 2>&1 &
tail -f runs/transformer/train.log
```

Para retomar depois de uma queda, repita o comando com `--resume` (mesmo lote, taxa e semente). Para treinar mais épocas, retome com um `--epochs` maior:

```bash
nohup python -u scripts/step03_train_transformer.py --epochs 20 --batch-size 8 --resume >> runs/transformer/train.log 2>&1 &
```

Com `tmux`: `tmux new -s treino`, rode `python -u scripts/step03_train_transformer.py --epochs 20 --batch-size 8 2>&1 | tee -a runs/transformer/train.log`, saia com `Ctrl+B D` e volte com `tmux attach -t treino`.

### 3. Avaliar, gerar e montar o relatório
Sempre com o `best.pt`:

```bash
python scripts/step04_evaluate_transformer.py   # results/transformer/test_metrics.json
python scripts/step05_generate_transformer.py   # results/generated/{transformer,reference}/
python scripts/step06_report_transformer.py     # results/transformer/loss_curves.png e README.md
```

A perplexidade é `exp` da média da entropia cruzada (log natural) de todos os tokens depois do BOS de cada música de teste: 435683 tokens. A geração segue `scripts/protocol.py`: os 50 prompts fixos são os primeiros 4 s de músicas de teste, a continuação tem exatamente 12 s (temperatura 1.0, top-k 32) e a máscara de eventos válidos (`nes_continuation.sampling.allowed_tokens`) garante vozes monofônicas e notas bem formadas. O Markov usa o mesmo protocolo.

### CLI do pacote
Os scripts acima usam a CLI, que também pode ser chamada diretamente:

```bash
python -m nes_continuation train --help
python -m nes_continuation evaluate --checkpoint runs/transformer/best.pt --split valid
python -m nes_continuation generate --checkpoint runs/transformer/best.pt     --prompt data/processed/nesmdb/tokens/test/<musica>.npz --prompt-seconds 4     --output exemplo.mid --wav
```
