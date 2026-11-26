"""
stop

Registra paradas por Ctrl+C, SIGTERM, SIGUSR1 ou pelo prazo `PRAZO_EPOCH`.
O treino consulta `should_stop()` entre lotes e salva o checkpoint antes de sair.
Os tratadores são instalados antes de importar PyTorch para capturar os sinais.
"""

### Imports
import os
import signal
import time

### Estado
STOP_REQUESTED = False
DEADLINE = float(os.environ.get("PRAZO_EPOCH", "inf"))


### Sinais de parada
def request_stop(signum, frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"[sinal] {signal.Signals(signum).name}: salvando e saindo no próximo ponto seguro", flush=True)

def install_handlers() -> None:
    for number in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(number, request_stop)

def should_stop() -> bool:
    return STOP_REQUESTED or time.time() >= DEADLINE
