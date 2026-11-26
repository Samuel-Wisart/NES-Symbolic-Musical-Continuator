"""
test03_lstm_tests

Testa o LSTM sem exigir o dataset preparado.
"""

### Imports
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from nes_continuation.model import ModelConfig, NESLSTM, NESTransformer, build_model
from nes_continuation.representation import BOS, VOCAB_SIZE
from nes_continuation.sampling import sample
from nes_continuation.training import CHECKPOINT_VERSION, load_checkpoint, restore_model, save_checkpoint
from nes_continuation.representation import VOCAB

### Auxiliares
def tiny_lstm(context=32):
    torch.manual_seed(0)
    return NESLSTM(ModelConfig(context=context, width=16, heads=1, layers=2, dropout=0.0, arch="lstm"))

### Testes
def test_forward_shape():
    """Confere a forma dos logits e a recusa de entradas maiores que o contexto."""
    model = tiny_lstm()
    tokens = torch.randint(0, VOCAB_SIZE, (3, 20))
    assert model(tokens).shape == (3, 20, VOCAB_SIZE)
    try:
        model(torch.randint(0, VOCAB_SIZE, (1, 33)))
    except ValueError:
        pass
    else:
        raise AssertionError("aceitou entrada maior que o contexto")

def test_causal():
    """Mudar tokens futuros não muda os logits das posições anteriores."""
    model = tiny_lstm().eval()
    tokens = torch.randint(0, VOCAB_SIZE, (2, 24))
    changed = tokens.clone()
    changed[:, 12:] = torch.randint(0, VOCAB_SIZE, (2, 12))
    with torch.no_grad():
        assert torch.allclose(model(tokens)[:, :12], model(changed)[:, :12], atol=1e-6)

def test_tied_output():
    """A saída usa os pesos do embedding (mesmo compartilhamento do Transformer)."""
    model = tiny_lstm()
    names = [name for name, _ in model.named_parameters()]
    assert not any("output" in name and "weight" in name and "norm" not in name for name in names), names

def test_config_and_build():
    """arch escolhe a classe. Configurações antigas, sem arch, continuam sendo Transformer."""
    assert isinstance(build_model(ModelConfig(context=16, width=16, heads=4, layers=1, arch="lstm")), NESLSTM)
    assert isinstance(build_model(ModelConfig(context=16, width=16, heads=4, layers=1)), NESTransformer)
    old = {"context": 16, "width": 16, "heads": 4, "layers": 1, "dropout": 0.1}
    assert ModelConfig(**old).arch == "transformer"
    # heads não precisa dividir width no LSTM
    ModelConfig(width=10, heads=3, arch="lstm")

def test_parameter_count():
    """A configuração do step03_train_lstm tem tamanho parecido com o do Transformer."""
    lstm = build_model(ModelConfig(context=1024, width=512, heads=8, layers=2, arch="lstm"))
    transformer = build_model(ModelConfig())
    count = lambda model: sum(parameter.numel() for parameter in model.parameters())
    ratio = count(lstm) / count(transformer)
    print(f"  parâmetros: LSTM {count(lstm):,} | Transformer {count(transformer):,} | razão {ratio:.2f}")
    assert 0.8 < ratio < 1.2, ratio

def test_checkpoint_round_trip():
    """restore_model reconstrói o LSTM a partir do checkpoint."""
    model = tiny_lstm()
    checkpoint = {
        "version": CHECKPOINT_VERSION, "vocab": list(VOCAB),
        "model_config": asdict(model.config), "model": model.state_dict(), "epoch": 1,
    }
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "lstm.pt"
        save_checkpoint(checkpoint, path)
        restored = restore_model(load_checkpoint(path), torch.device("cpu"))
    assert isinstance(restored, NESLSTM)
    tokens = torch.randint(0, VOCAB_SIZE, (1, 10))
    with torch.no_grad():
        assert torch.equal(model.eval()(tokens), restored.eval()(tokens))

def test_sampling():
    """A geração com máscara funciona com o LSTM (mesma interface do Transformer)."""
    model = tiny_lstm(context=64)
    result = sample(model, [BOS], torch.device("cpu"), seed=0, min_seconds=0.5, max_seconds=0.5)
    assert result["generated_tokens"] > 0
    assert result["generated_steps"] <= 32

def main():
    tests = [
        test_forward_shape,
        test_causal,
        test_tied_output,
        test_config_and_build,
        test_parameter_count,
        test_checkpoint_round_trip,
        test_sampling,
    ]
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")
    print(f"\n{len(tests)} testes passaram. LSTM ok.")

if __name__ == "__main__":
    main()
