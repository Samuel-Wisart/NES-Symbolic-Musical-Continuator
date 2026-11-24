# TP2 - AI Music Generation
Projeto de geração musical autoregressiva usando o dataset NES-MDB. A tarefa escolhida é a continuação musical: dado o início de uma música, o modelo compõe os segundos seguintes. Comparamos uma cadeia de Markov (baseline) com dois modelos neurais, um Transformer e um LSTM, todos sobre a mesma representação e o mesmo protocolo.

**Resultados prontos para a análise e o relatório: [`resultados/`](resultados/README.md)** (modelos treinados, métricas, comparação e as amostras geradas).

## Estrutura
```text
data/
  raw/                         # NES-MDB original
  processed/                   # dados tokenizados
nes_continuation/              # modelos neurais: Transformer e LSTM, treino, geração e áudio
  representation.py            # ponte para a representação do step01
  model.py                     # NESTransformer, NESLSTM e build_model
resultados/                    # resultados entregues (ver resultados/README.md)
  transformer/  lstm/          # modelo treinado (modelo.pt), métricas, curvas e README de cada modelo
  amostras/                    # continuações: transformer/, lstm/ e referencia/ (a música real)
  comparacao.md                # tabela comparando os modelos e estatísticas das amostras
runs/                          # checkpoints e logs de treino (não versionado)
scripts/
  protocol.py                  # protocolo de avaliação e geração (comum a todos os modelos)
  step01_prepare_dataset.py    # preparação e tokenização
  step02_analyze_dataset.py    # estatísticas do dataset e da representação
  step03_train_transformer.py  # treino do Transformer
  step03_train_lstm.py         # treino do LSTM
  step04_evaluate.py           # perplexidade e acurácia no teste (--model transformer|lstm)
  step05_generate.py           # continuações do protocolo + referência real (--model ...)
  step06_report.py             # curvas de loss e README de um modelo (--model ...)
  step07_compare.py            # comparação entre os modelos e estatísticas das amostras
tests/
  test01_dataset_tests.py      # testes da representação
  test02_transformer_tests.py  # testes do Transformer sobre os tokens do step01
  test03_lstm_tests.py         # testes do LSTM
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
Valida os pontos essenciais da representação MIDI-Like antes do processamento completo do dataset, incluindo vocabulário, `encode -> decode`, quantização e `TIME_SHIFT`. Rode com `PYTHONPATH=scripts`.

### test02_transformer_tests.py
Valida a adaptação do Transformer aos tokens do step01 (requer o dataset preparado): `.npz` igual a `encode(read_notes(mid))`, corte do prompt em exatamente N segundos, `decode -> MIDI -> read_notes -> encode`, cobertura das janelas, lista de prompts do protocolo, `forward` com os 1155 tokens e respeito à máscara de eventos válidos na geração.

### test03_lstm_tests.py
Valida o LSTM (não precisa do dataset): forma da saída, causalidade, saída compartilhada com o embedding, tamanho parecido com o do Transformer, checkpoint e geração com a máscara.

## Modelos neurais: Transformer e LSTM
O pacote `nes_continuation/` (PyTorch) usa a representação do step01: importa `VOCAB`, `TO_ID`, `encode`, `decode` e `read_notes` de lá e usa os 1155 tokens. Lê os `.npz` de `data/processed/nesmdb/tokens/`, então o dataset precisa estar preparado (step01) na máquina de treino.

Os dois modelos têm o mesmo treino, para que a comparação dependa só da arquitetura: mesmos tokens, mesmas janelas (1024 tokens, passo 512), AdamW 3e-4 (weight decay 0.01, clipping 1.0), lote 8, semente 42 e o `best.pt` escolhido pela menor loss de validação.

| | Transformer | LSTM |
|---|---|---|
| Arquitetura | decoder-only, 6 blocos pre-LN, largura 256, 8 cabeças | 2 camadas LSTM de largura 512 |
| Parâmetros | 5,3 M | 4,8 M |
| Treino | `python scripts/step03_train_transformer.py` | `python scripts/step03_train_lstm.py` |
| Épocas (padrão do script) | 100 | 79 |
| Checkpoints | `runs/transformer/` | `runs/lstm/` |

### GPU
O `torch` do PyPI no Windows é só CPU. Instale a versão com CUDA indicada em https://pytorch.org/get-started/locally/ e confira:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

O treino usa BF16 nas GPUs que suportam e FP16 (com `GradScaler`) nas demais. Numa RTX 4090, uma época leva ~31 s no Transformer e ~1,7 min no LSTM; na CPU, horas por época.

### 1. Medir o tempo por época (opcional)
Roda alguns lotes, estima o tempo de uma época e sai sem salvar nada:

```bash
python scripts/step03_train_transformer.py --benchmark 30
python scripts/step03_train_lstm.py --benchmark 30
```

### 2. Treinar
```bash
python -u scripts/step03_train_transformer.py
python -u scripts/step03_train_lstm.py
```

São salvos em `runs/<modelo>/`: `last.pt` (a cada 15 min, no fim de cada época e ao parar), `best.pt` (menor loss de validação), `metrics.json`, `run_info.json`, `progresso.json` e, no fim, `CONCLUIDO`.

O treino é retomável: **o mesmo comando começa do zero ou continua do `last.pt`**, inclusive no meio de uma época (o checkpoint guarda a posição do lote, o otimizador e os estados aleatórios). Ctrl+C salva e sai. Ele se recusa a retomar se lote, taxa, semente, dados ou arquitetura mudarem; só `--epochs` pode aumentar. Para recomeçar do zero, use `--do-zero` ou outra `--run-dir`.

### 3. Avaliar, gerar, relatar e comparar
```bash
python scripts/step04_evaluate.py --model transformer   # resultados/transformer/test_metrics.json e modelo.pt
python scripts/step05_generate.py --model transformer   # resultados/amostras/{transformer,referencia}/
python scripts/step06_report.py   --model transformer   # resultados/transformer/README.md e curvas_loss.png
# o mesmo com --model lstm
python scripts/step07_compare.py                        # resultados/comparacao.md, comparacao.json e curvas_validacao.png
```

O step04 e o step05 usam `runs/<modelo>/best.pt` quando o modelo foi treinado na máquina e, se não houver, o modelo entregue em `resultados/<modelo>/modelo.pt`. Ou seja, **dá para gerar e avaliar sem treinar**.

A perplexidade é `exp` da média da entropia cruzada (log natural) de todos os tokens depois do BOS de cada música de teste: 435683 tokens. A geração segue `scripts/protocol.py`: os 100 prompts fixos são os primeiros 4 s de músicas de teste, a continuação tem exatamente 12 s (temperatura 1.0, top-k 32) e a máscara de eventos válidos (`nes_continuation.sampling.allowed_tokens`) garante vozes monofônicas e notas bem formadas. O Markov usa o mesmo protocolo; se gravar em `resultados/markov/test_metrics.json` e `resultados/amostras/markov/`, entra sozinho no step07.

### CLI do pacote
Os scripts acima usam a CLI, que também pode ser chamada diretamente:

```bash
python -m nes_continuation train --help
python -m nes_continuation evaluate --checkpoint resultados/transformer/modelo.pt --split valid
python -m nes_continuation generate --checkpoint resultados/lstm/modelo.pt --prompt data/processed/nesmdb/tokens/test/<musica>.npz --prompt-seconds 4 --output exemplo.mid --wav
```
