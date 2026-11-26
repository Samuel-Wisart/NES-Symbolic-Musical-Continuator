# Markov original — sem prompt

Snapshot da implementação original do Markov antes da integração com o protocolo
de continuação musical.

Esta versão:

- usa cadeia de Markov de ordem 1;
- usa a representação MIDI-Like do NES-MDB;
- inicia a geração em `BOS`;
- não recebe os 4 segundos de prompt do Transformer/LSTM;
- usa temperatura 1.0, `min_tokens=200`, `n_max=2000` e seed 42;
- preserva 100 amostras para comparação histórica.

`amostras/` contém MIDI e NPZ das 100 gerações e WAV das 5 primeiras.
`analise/` contém a comparação e os boxplots produzidos pelo `step07_compare.py`.
`metrics.json` contém perplexidade de validação/teste do Markov e do unigrama.

Os `.npz` desta pasta foram criados apenas como adaptador de análise. Como esta
versão não usa prompt musical, `prompt_length=1` representa somente o token BOS.

Estes resultados não devem ser confundidos com a versão final condicionada por
prompt.
