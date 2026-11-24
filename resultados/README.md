# Resultados: Transformer e LSTM

Tudo o que é preciso para a análise e o relatório, sem treinar nada. Os dois modelos foram
treinados com **o mesmo treino** (mesmos dados, janelas de 1024 tokens, AdamW 3e-4, lote 8,
semente 42) e avaliados e amostrados com **o mesmo protocolo** (`scripts/protocol.py`), que é
também o do Markov. A diferença entre eles é só a arquitetura.

## Resumo

| | Transformer | LSTM |
|---|---|---|
| Arquitetura | decoder-only, 6 blocos, largura 256, 8 cabeças | 2 camadas LSTM, largura 512 |
| Parâmetros | 5.296.896 | 4.794.880 |
| Épocas treinadas / melhor época | 100 / **40** | 79 / **12** |
| Loss de validação (melhor época) | 1,1063 | 2,0913 |
| **Perplexidade de teste** (janelas deslizantes) | **2,47** | **6,56** |
| Acurácia top-1 no teste | 74,3% | 50,1% |

Perplexidade = exp(média da entropia cruzada, em log natural, de todos os tokens depois do BOS
das 372 músicas de teste; 435.683 tokens). Menor é melhor.

Pontos para a análise:
- **Os dois modelos têm overfitting**, e o `modelo.pt` entregue é sempre o da melhor época de
  validação. O Transformer estabiliza por volta da época 30–40 e depois a validação sobe
  devagar. O LSTM chega ao melhor muito cedo (época 12) e piora rápido: na época 79, a loss de
  treino é 0,67 e a de validação, 2,46. Por isso o treino do LSTM foi encerrado na época 79.
  Ver `curvas_validacao.png`.
- **Com tamanho parecido, o Transformer prevê muito melhor.** A atenção vê diretamente todo o
  contexto de 1024 tokens (~17 s de música); o LSTM precisa resumir o passado num estado de
  tamanho fixo.
- **Estatísticas das amostras** (notas por segundo, vozes ativas, duração das notas, registro
  em relação ao prompt), comparadas com a continuação real: `comparacao.md`.

## Organização

```text
resultados/
  README.md               este arquivo
  comparacao.md           tabela dos modelos e estatísticas das amostras (step07_compare.py)
  comparacao.json         os mesmos números, para outras análises
  curvas_validacao.png    loss de treino e de validação por época dos dois modelos
  transformer/
    README.md             relatório do modelo: arquitetura, treino, teste, geração, limitações
    curvas_loss.png       loss por época
    modelo.pt             melhor modelo (época 40), só os pesos e a configuração (~21 MB)
    test_metrics.json     métricas de teste (janelas deslizantes e disjuntas)
    train_history.json    loss e métricas de validação de cada época
    run_info.json         configuração e hardware da execução
  lstm/                   o mesmo para o LSTM (melhor modelo: época 12, ~19 MB)
  amostras/
    transformer/          100 continuações do Transformer
    lstm/                 100 continuações do LSTM
    referencia/           a continuação real das mesmas 100 músicas
```

Cada pasta de `amostras/` tem, para cada uma das 100 músicas de teste do protocolo:
- `<música>.mid`: os 4 s de prompt (a música real) seguidos de 12 s de continuação, com as
  trilhas p1, p2, tr e no (canais MIDI 0, 1, 2 e 9; 120 bpm);
- `<música>.npz`: os tokens (`ids`) e onde termina o prompt (`prompt_length`), para análises
  sobre a representação (`nes_continuation.representation.decode` converte em notas);
- `summary.json`: protocolo usado e, por música, semente, tokens gerados e duração.

O mesmo nome de arquivo nas três pastas é a mesma música e o mesmo prompt, então a comparação é
pareada: dá para ouvir o prompt seguido de cada continuação, lado a lado. Continuações: 12 s
exatos, temperatura 1,0, top-k 32, semente `42 + i` para a i-ésima música, máscara de eventos
válidos (vozes monofônicas). A referência pode ser mais curta quando a música acaba antes.

## Como ouvir
Os `.mid` abrem em qualquer player ou DAW (sem os timbres do NES). Para uma prévia chiptune em
WAV, que não fica versionada, gere de novo com `--wav` ou rode o step05: as 5 primeiras músicas
saem também em WAV.

```bash
python -m nes_continuation generate --checkpoint resultados/transformer/modelo.pt \
    --prompt data/processed/nesmdb/tokens/test/<música>.npz --prompt-seconds 4 --output exemplo.mid --wav
```

## Como usar os modelos
`modelo.pt` carrega com o pacote (requer o dataset preparado pelo step01 para avaliar ou gerar
a partir de músicas de teste):

```bash
python scripts/step04_evaluate.py --model lstm    # usa resultados/lstm/modelo.pt se não houver runs/lstm/
python scripts/step05_generate.py --model lstm    # gera de novo as 100 continuações
python scripts/step07_compare.py                  # refaz comparacao.md, .json e a figura
```

## Observações
- **Hardware.** Os treinos rodaram em GPUs RTX 4090 e RTX 3090 Ti, trocando de máquina entre
  um trecho e outro. O treino é retomável e continua exatamente do mesmo lote. O `run_info.json`
  registra só a GPU do último trecho. Tempo por época numa RTX 4090: ~31 s para o Transformer e
  ~1,7 min para o LSTM.
- **Markov.** Para entrar na comparação, basta gravar `resultados/markov/test_metrics.json`
  (mesmo formato) e as continuações em `resultados/amostras/markov/` (mesmos nomes e formato) e
  rodar `python scripts/step07_compare.py`.
- **Observações de escuta.** Estão a preencher nos `README.md` de `transformer/` e `lstm/`.
