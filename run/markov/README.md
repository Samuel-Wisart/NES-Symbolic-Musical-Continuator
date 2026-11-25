# Gerador de música NES com Cadeia de Markov

Pipeline que lê o dataset **NES-MDB**, converte as músicas para tokens MIDI-Like,
treina uma cadeia de Markov, gera músicas novas em MIDI e as converte em áudio
com um sintetizador chiptune. 

```
NES-MDB (.tar.gz) → tokens → modelo Markov → MIDIs gerados → áudio (.wav)
   prepareData       train / ngram     generate / ngram       midi_to_audio
```

## Scripts

| Script | Função |
|---|---|
| `prepareData_markov.py` | Extrai o NES-MDB e converte cada MIDI em tokens |
| `train_markov.py` | Treina a cadeia de Markov de **ordem 1** |
| `generate_markov.py` | Gera MIDIs com o modelo de ordem 1 |
| `markov_ngram.py` | Treina **e** gera com Markov de **ordem N** (2 a 5), com backoff |
| `midi_to_audio.py` | Converte os MIDIs gerados em `.wav` (e opcionalmente `.mp3`) |

Você escolhe **um** dos dois caminhos de modelo: ordem 1 (`train_markov` + `generate_markov`)
ou ordem N (`markov_ngram`). A ordem N costuma soar melhor.

## Requisitos

```bash
pip install numpy pretty_midi
```

Opcional: `ffmpeg` instalado no sistema, só se quiser gerar `.mp3`.

## Estrutura de pastas

Todos os scripts calculam a raiz do projeto como `Path(__file__).parents[2]`, ou seja,
devem ficar **duas pastas abaixo da raiz**. Coloque todos juntos na mesma pasta (eles
importam uns aos outros):

```
projeto/
├── data/
│   ├── raw/
│   │   └── nesmdb_midi.tar.gz        ← você coloca aqui o dataset
│   └── processed/nesmdb/             ← criado automaticamente
│       ├── tokens/{train,valid,test}/
│       ├── vocab.json
│       ├── token_index.csv
│       ├── markov_order1.npz         ← modelo de ordem 1
│       └── ngram_order4.npz          ← modelo de ordem N
├── output/
│   ├── markov_samples/               ← MIDIs gerados
│   └── markov_audio/                 ← áudios gerados
└── src/
    └── markov/
        ├── prepareData_markov.py
        ├── train_markov.py
        ├── generate_markov.py
        ├── markov_ngram.py
        └── midi_to_audio.py
```

Rode os comandos de dentro de `src/markov/`.

## Passo a passo

### 1. Preparar os dados (uma vez)

Coloque o `nesmdb_midi.tar.gz` (versão MIDI do NES-MDB) em `data/raw/` e execute:

```bash
python prepareData_markov.py
```

Isso extrai os MIDIs, tokeniza cada música e salva tudo em `data/processed/nesmdb/`.

> **Atenção:** rodar este passo de novo apaga `data/processed/nesmdb/` inteira,
> **incluindo os modelos treinados**. Depois de refazer a tokenização, treine de novo.

### 2A. Caminho ordem 1

```bash
python train_markov.py
python generate_markov.py
```

O treino imprime a perplexidade no split de validação (Markov vs. baseline unigrama).
Opções do gerador:

```bash
python generate_markov.py --n-samples 10 --temperature 1.2 --n-max 2000 --min-tokens 200 --seed 42
```

### 2B. Caminho ordem N (recomendado)

```bash
python markov_ngram.py train --order 4
python markov_ngram.py generate --order 4 --temperature 0.9 --top-p 0.9
```

O treino mostra a perplexidade de validação para as ordens 1 até N, o que ajuda
a escolher a ordem. Opções de geração:

| Opção | Padrão | O que faz |
|---|---|---|
| `--order` | 4 | Ordem do modelo (precisa ter sido treinado antes; máx. 5) |
| `--n-samples` | 10 | Quantos MIDIs gerar |
| `--temperature` | 0.9 | Menor = mais previsível; maior = mais caótico |
| `--top-p` | 0.9 | Descarta a cauda de tokens improváveis (1.0 desliga) |
| `--min-count` | 2 | Suporte mínimo de um contexto; menor copia mais o treino |
| `--n-max` | 2000 | Limite de tokens por música |
| `--min-tokens` | 200 | Impede que a música termine cedo demais |
| `--seed` | 42 | Mesma seed gera as mesmas músicas |

### 3. Converter em áudio

```bash
python midi_to_audio.py
```

Lê todos os `.mid` de `output/markov_samples/` e grava um `.wav` de cada em
`output/markov_audio/`. Arquivos que já têm `.wav` são pulados.

```bash
python midi_to_audio.py --overwrite              # reconverte tudo
python midi_to_audio.py --mp3                    # também gera .mp3 (precisa de ffmpeg)
python midi_to_audio.py --input-dir X --output-dir Y --sample-rate 48000
```

O áudio é gerado por um sintetizador chiptune próprio (pulso, triângulo e ruído),
sem precisar de FluidSynth nem de soundfont.

## Dicas de ajuste

- **Soa aleatório demais:** diminua `--temperature` (0.7 a 0.9) ou aumente a ordem.
- **Soa repetitivo ou copia o treino:** aumente `--temperature`, diminua a ordem ou
  aumente `--min-count`.
- **Ordem alta consome mais memória e disco.** Comece em 3 ou 4.
- **Timbre e mix:** ajuste `GANHO`, `DUTY`, `NOISE_BASE_HZ` e `NOISE_DECAY_S` no
  topo do `midi_to_audio.py`; programas GM e remapeamento do ruído ficam no topo do
  `generate_markov.py`.
- **Resolução de tempo:** `TIME_RESOLUTION_HZ` no `prepareData_markov.py` (padrão 64).
  Valores menores concentram as estatísticas, mas exigem refazer os passos 1 e 2.

## Problemas comuns

| Erro | Causa e solução |
|---|---|
| `ModuleNotFoundError: prepareData_markov` | Os scripts não estão na mesma pasta, ou você não está rodando de dentro dela |
| `Dataset não encontrado` | Falta o `nesmdb_midi.tar.gz` em `data/raw/` |
| `Matriz treinada com vocab de X tokens...` | O vocabulário mudou depois do treino; rode o treino de novo |
| `FileNotFoundError` ao gerar | O modelo não foi treinado (ou foi apagado pelo passo 1) |
| `Nenhum MIDI encontrado` no áudio | Gere os MIDIs antes (passo 2) |
| MIDI do dataset falha ao carregar | O script já define `MAX_TICK` como recomendam os autores; se persistir, trate o erro por arquivo em `tokenize_dataset` e pule o arquivo |
| `ffmpeg não encontrado` | Instale o ffmpeg ou rode sem `--mp3` |

## Limitações

Uma cadeia de Markov só enxerga os últimos N tokens, então não mantém ritmo, melodia
ou harmonia em escala longa. Para resultados melhores, o próximo passo é um modelo
neural (LSTM ou Transformer) treinado sobre os mesmos tokens.
