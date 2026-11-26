# Geração musical simbólica autoregressiva

Projeto de continuação musical com o dataset [NES-MDB](https://github.com/chrisdonahue/nesmdb). Dado um trecho inicial de uma música, os modelos geram uma continuação. O projeto inclui um Transformer, um LSTM e uma cadeia de Markov como baseline.

## Estrutura de pastas

- `data/`: reúne o dataset MIDI original e os dados processados para os modelos.
- `data/raw/`: guarda o arquivo NES-MDB baixado e as músicas MIDI extraídas.
- `data/processed/`: contém as sequências tokenizadas, o vocabulário, os índices e a matriz de transições do Markov.
- `nes_continuation/`: implementa a representação musical, os modelos neurais, o treino, a avaliação, a geração e a síntese de áudio.
- `runs/`: recebe checkpoints e registros criados durante o treinamento.
- `resultados/`: reúne métricas, gráficos, modelos exportados e arquivos gerados.
- `resultados/transformer/` e `resultados/lstm/`: guardam o modelo exportado e as métricas de cada modelo neural.
- `resultados/amostras/`: organiza continuações e referências por sistema, com MIDI, tokens e os WAVs gerados.
- `resultados/musicasGeradas/`: contém um conjunto adicional de arquivos MIDI do Transformer, LSTM e Markov.
- `run/markov/`: contém o treinamento e a geração do baseline de Markov, além das ferramentas de preparação e síntese MIDI.
- `scripts/`: reúne os passos de preparação, análise, treino, avaliação, geração e comparação dos resultados.
- `tests/`: contém os testes da representação, dos modelos e da integração do Markov.

## Requisitos

- Python 3.10 ou superior
- Dependências listadas em [`requirements.txt`](requirements.txt)
- Dataset NES-MDB em MIDI, salvo como `data/raw/nesmdb_midi.tar.gz`

## Instalação

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Em Linux ou macOS, ative o ambiente com `source .venv/bin/activate`. Para usar uma GPU NVIDIA, instale a versão de PyTorch com CUDA indicada no [seletor oficial do PyTorch](https://pytorch.org/get-started/locally/).

## Reprodução

Prepare e analise o dataset:

```bash
python scripts/step01_prepare_dataset.py
python scripts/step02_analyze_dataset.py
```

Treine os modelos neurais, avalie-os e gere as continuações:

```bash
python scripts/step03_train_transformer.py
python scripts/step03_train_lstm.py
python scripts/step04_evaluate.py --model transformer
python scripts/step04_evaluate.py --model lstm
python scripts/step05_generate.py --model transformer
python scripts/step05_generate.py --model lstm
```

Treine e gere com o baseline de Markov:

```bash
python run/markov/train_markov.py
python run/markov/generate_markov.py
```

Gere os relatórios e gráficos a partir dos resultados:

```bash
python scripts/step06_report.py --model transformer
python scripts/step06_report.py --model lstm
python scripts/step07_compare.py
```

Os checkpoints de treino ficam em `runs/`. Métricas, arquivos MIDI, sequências tokenizadas, referências e gráficos ficam em `resultados/`. Os scripts de geração salvam WAVs para as primeiras cinco amostras.

Para executar os testes:

```bash
python tests/test01_dataset_tests.py
python tests/test02_transformer_tests.py
python tests/test03_lstm_tests.py
python tests/test04_markov_tests.py
```

Os testes do Transformer e de Markov precisam do dataset preparado.

## Integrantes e contribuições

| Integrante | Contribuição |
|---|---|
| Lucas Rezende | Organização do Dataset, Implementação da representação e seus testes, Parte do Markov, Relatório |
| Luiza Sodré | Auxilio no desenvolvimento do transforme, Algumas métricas, Relatório |
| Samuel | Definição de quais modelos usar (Transformer e LSTM), Treinamento desses modelos |
| Wilian | Parte do Markov, Relatório |

## Uso de ferramentas de IA
- Ajuda com os comentários do código, formatação e organização dos dados.
-
