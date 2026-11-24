"""
stop

Parada pedida pelo usuário (Ctrl+C) ou por sinal (SIGTERM, SIGUSR1, como os que
um gerenciador de filas envia antes de encerrar um processo). Não importa o
torch: os tratadores precisam ser instalados antes dos imports pesados, porque a
ação padrão de USR1/TERM mata o processo.

O treino consulta `should_stop()` depois de cada lote e, se for o caso, salva o
checkpoint e sai. Opcionalmente, PRAZO_EPOCH (instante Unix, em segundos) define
um prazo para parar sozinho.
"""

### Imports
import os
import signal
import time

### Estado
STOP_REQUESTED = False
DEADLINE = float(os.environ.get("PRAZO_EPOCH", "inf"))


def request_stop(signum, frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"[sinal] {signal.Signals(signum).name}: salvando e saindo no próximo ponto seguro", flush=True)


def install_handlers() -> None:
    for number in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(number, request_stop)


def should_stop() -> bool:
    return STOP_REQUESTED or time.time() >= DEADLINE
