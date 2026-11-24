# CLAUDE.md

Contexto para continuar o trabalho no repositório.

## Projeto
TP2 de IA e Música (UFMG): continuação musical simbólica no NES-MDB (4 vozes:
p1, p2, tr, no). Comparação entre um Transformer (esta parte) e um baseline de
cadeia de Markov (feito por um colega, em paralelo, neste repositório). Entrega:
quarta-feira, 25/11/2026, 23:59.

## Regras do grupo
- `scripts/step01_prepare_dataset.py` (Lucas) é a representação OFICIAL: `VOCAB`
  (1155 tokens), `TO_ID`, `encode`, `decode`, `read_notes`. **Não alterar** (o
  Markov depende dele). Problemas encontrados nele são reportados, não corrigidos.
- Uma única fonte da verdade: o pacote importa a representação do step01
  (`nes_continuation/representation.py` ajusta o `sys.path`). Não duplicar o
  vocabulário; usar os 1155 tokens (mesmo os que nunca aparecem).
- Partições oficiais. O teste só é usado na avaliação final (step04) e para
  tirar os prompts de geração.
- Não versionar dados, checkpoints de treino nem WAV (`data/`, `runs/`,
  `resultados/amostras/**/*.wav` no .gitignore). JSONs, MIDI, `.npz` das amostras e
  figuras podem. Exceção pedida pelo usuário: os modelos exportados
  `resultados/<modelo>/modelo.pt` (só pesos, ~20 MB) são versionados.
- Branches: `main` (execução numa máquina comum e resultados), `execucao-slurm`
  (a `main` + scripts do cluster SONIC), `transformer` e `lstm` (histórico do
  desenvolvimento). Um commit por fase. Código e comentários em português,
  no estilo do Lucas (docstring com o nome do módulo, seções `### ...`).
- Convenções: `scripts/stepXX_*.py`; `tests/testXX_*.py` executáveis com
  `python tests/arquivo.py`.

## Estrutura dos modelos neurais
- `nes_continuation/representation.py`: ponte para o step01 + tabelas derivadas
  do VOCAB, `load_ids`, `split_at_seconds`/`cut_prompt`, `reference_continuation`,
  `NoteState` (notas abertas, FIFO como o `decode`), `close_open_notes`,
  `write_midi` (canais 0, 1, 2 e 9; 120 bpm, 480 ppq = 15 ticks por passo, exato).
- `data.py`: `NESWindows` (treino, passo = contexto/2; validação, passo = contexto)
  e `ScoringWindows` (avaliação, cada alvo pontuado uma vez).
- `model.py`: Transformer (Luiza) decoder-only pre-LN, embeddings de posição
  aprendidos, saída compartilhada com o embedding; LSTM (`NESLSTM`, 2×512, mesma saída
  compartilhada, FP32 dentro do autocast); `ModelConfig.arch` e `build_model`.
- `training.py`: treino retomável (o mesmo comando começa ou continua, inclusive no
  meio da época; `--do-zero` recomeça), checkpoint versão 2 (valida `vocab`),
  `--benchmark`, BF16/FP16, `run_info.json`, `progresso.json`, `CONCLUIDO`,
  `export_model` (modelo sem otimizador para `resultados/`). `stop.py`: Ctrl+C,
  SIGTERM/SIGUSR1 e `PRAZO_EPOCH` salvam e saem.
- `sampling.py`: `allowed_tokens` (máscara), `sample`, `save_generation`.
- `audio.py` (Luiza): prévia WAV chiptune; vozes pelo canal MIDI.
- `scripts/protocol.py`: protocolo compartilhado com o Markov; pastas de `resultados/`
  e `checkpoint_path(modelo)` (runs/<modelo>/best.pt ou resultados/<modelo>/modelo.pt).
- `scripts/step03_train_{transformer,lstm}.py`: treino de cada modelo;
  `step04_evaluate`, `step05_generate`, `step06_report` com `--model`;
  `step07_compare` (comparação e estatísticas das amostras).

## Decisões tomadas
- **Máscara de geração** (`sampling.allowed_tokens`): nunca PAD/BOS; NOTE_ON só
  em voz sem nota soando; NOTE_OFF só para (voz, pitch) aberto e **não no mesmo
  passo do NOTE_ON** (regra extra: o step01 garante duração mínima de 1 passo e o
  `decode` descartaria nota de duração zero); EOS só depois de `min_seconds`;
  TIME_SHIFT sem passar de `max_seconds`. Com essas regras a geração sempre
  termina, por isso não há limite de tokens por padrão.
- A regra de ordem das vozes dentro de um passo do código antigo da Luiza não foi mantida.
- **Prompt**: corte exato em N s (TIME_SHIFT dividido); eventos no próprio passo
  do corte ficam para a continuação; notas abertas continuam abertas.
- **Fechamento**: ao fim, NOTE_OFF das notas abertas (com um TIME_SHIFT_1 antes,
  se alguma começou no último passo) e EOS. Em `sample`, o EOS amostrado não entra
  na sequência; `ids[prompt_length:prompt_length + generated_tokens]` são os tokens
  amostrados e `stop_reason` diz se houve EOS.
- **Protocolo**: PROMPT_SECONDS 4, GEN_SECONDS 12, **MIN_GEN_SECONDS = 12** (EOS
  proibido: toda continuação tem exatamente 12 s, para comparar segmentos de mesma
  duração), N_PROMPTS 100 (eram 50; o professor pediu 100 em 2026-11-24), SEED 42,
  temperatura 1.0, top-k 32, semente `SEED + i` para a i-ésima música. A lista
  `PROMPT_SONGS` foi sorteada com `random.Random(42).sample(elegíveis em ordem
  alfabética, N)` entre as 259 músicas de teste com pelo menos 6 s e fica fixa no
  arquivo (o test02 confere); os 50 primeiros são a lista original.
- **Referência**: os mesmos 4 s de prompt + os 12 s reais seguintes (ou até o fim
  da música), notas fechadas, no mesmo formato da geração.
- **Perplexidade**: exp(média da entropia cruzada, em log natural, de todos os
  tokens depois do BOS de cada música de teste) = 435683 tokens. Medida principal
  com janelas deslizantes (passo = contexto/2, todo token com pelo menos meio
  contexto de histórico); a medida com janelas disjuntas também é salva.
- **Precisão**: BF16 onde a GPU suporta, FP16 com GradScaler nas demais, FP32 na
  CPU. A validação/teste roda em FP32.
- **Checkpoint**: `run_config["data"]` relativo ao repositório, para retomar em
  outra máquina. A retomada exige mesmo lote, taxa, semente e dados.
- Commit inicial na branch com o código recebido (o `main` não tinha commits).

## Fatos verificados nos dados
- Teste: 372 músicas, soma de len(ids) - 1 = 435683. Treino: 4500 músicas,
  8.743.119 tokens-alvo; ~60 tokens por segundo de música.
- 642 dos 1155 tokens aparecem no treino. Alturas: p1/p2 33–108, tr 21–108, no 1–16.
- **Problema no step01 (reportado, não corrigido)**: 13,8% dos NOTE_ON do treino
  (420.736 de 3.045.303, em 3.808 músicas) ocorrem com a voz já soando. Causa:
  notas mais curtas que meio passo são esticadas para 1 passo pelo
  `quantize_interval` e se sobrepõem à próxima nota da mesma voz, que começa no
  mesmo passo. 1272 casos têm o mesmo pitch (o round trip via MIDI não é exato
  neles, porque o pretty_midi fecha as duas notas no mesmo note-off).
- `python tests/test01_dataset_tests.py` (Lucas) falha sem `PYTHONPATH=scripts`;
  não foi alterado.

## Ambiente local (Windows)
- O `python` do PATH é o atalho da Microsoft Store. Use
  `C:\Users\Irídio\AppData\Local\Python\bin\python.exe` (3.14, torch 2.12.1 CPU).
- A GPU local (GTX 750, 2 GB) não serve para o PyTorch atual. Configuração
  completa na CPU: ~9,5 s por lote de 8×1024, ~2080 lotes por época (~4–5 h/época).
  O treino de verdade vai rodar em outra máquina com GPU.

## Estado (2026-11-24)
- Transformer e LSTM treinados, avaliados e com as 100 continuações geradas; tudo em
  `resultados/` (ver `resultados/README.md` e `resultados/comparacao.md`).
- Transformer: 100 épocas, melhor na 40; teste: perplexidade 2,47, acurácia 74,3%.
- LSTM: treino encerrado na época 79 (melhor na 12; a validação só piorou depois).
- Falta: observações de escuta nos READMEs de `resultados/`, resultados do Markov
  (`resultados/markov/` e `resultados/amostras/markov/`, depois `step07_compare.py`)
  e avisar o colega do Markov que o protocolo passou para 100 prompts.
- A execução no cluster SONIC (Slurm) fica só na branch `execucao-slurm`; a `main`
  tem apenas a execução numa máquina comum.

## Verificação
```bash
python tests/test02_transformer_tests.py
PYTHONPATH=scripts python tests/test01_dataset_tests.py
```
