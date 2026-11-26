# Baseline Markov

O baseline oficial do trabalho é uma cadeia de Markov de ordem 1 treinada sobre os mesmos tokens MIDI-Like do `step01` usados pelo Transformer e pelo LSTM.

O treinamento estima `P(x_t | x_{t-1})` a partir do split oficial de treino. Na geração, o mesmo prompt de 4 segundos do protocolo é preservado no início do arquivo, mas a cadeia usa apenas o último token para decidir o próximo. A máscara compartilhada em `nes_continuation.sampling.allowed_tokens` mantém as quatro vozes monofônicas e impede eventos inválidos.

## Execução

A partir da raiz do repositório, depois de preparar o dataset:

```bash
python run/markov/train_markov.py
python tests/test04_markov_tests.py
python run/markov/generate_markov.py
python scripts/step07_compare.py
```

O treino salva `data/processed/nesmdb/markov_order1.npz`. A geração usa os 100 prompts fixos definidos em `scripts/protocol.py`, com 4 s de prompt, 12 s de continuação, temperatura 1.0, top-k 32 e sementes `42 + i`. As saídas são gravadas em `resultados/amostras/markov/` no mesmo formato das amostras neurais (`.npz`, `.mid`, cinco `.wav` e `summary.json`).

`prepareData_markov.py` e `markov_ngram.py` permanecem no repositório como experimentos anteriores, mas não fazem parte do pipeline final do baseline. Para a comparação oficial, a preparação vem de `scripts/step01_prepare_dataset.py` e o baseline é o Markov de ordem 1 acima.
