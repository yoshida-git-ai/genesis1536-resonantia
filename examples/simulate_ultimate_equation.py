# -*- coding: utf-8 -*-
"""
simulate_ultimate_equation.py

FluctuationalNeuralCore を使って、簡単な入力 v(t) に対して
Phi, 観測者状態 o, 共鳴スコア s(t) がどう変化するかを見るためのサンプル。
"""

import math
import numpy as np

from src.ai_field_core.fluctuation_neural_core import (
    FluctuationalNeuralCore,
    CoreState,
    WeightState,
    ObserverState,
    DIM,
)


def gen_input_vector(t: float, dim: int = DIM) -> np.ndarray:
    """
    とりあえずのダミー入力。
    時間 t に応じて、1 個の成分だけが 1.0 になる「回転するワンホット」。
    """
    base = np.zeros(dim, dtype=np.float32)
    k = int((t * 3.0) % dim)
    base[k] = 1.0
    return base


def main() -> None:
    core = FluctuationalNeuralCore(dim=DIM)
    Phi0 = np.zeros(DIM, dtype=np.float32)
    W0 = WeightState(tensors={"beta": 0.0})
    o0 = ObserverState()

    state = CoreState(Phi=Phi0, W=W0, o=o0)

    dt = 0.02
    T = 3.0
    steps = int(T / dt)

    for i in range(steps):
        t = i * dt
        v_t = gen_input_vector(t)
        state, metrics = core.step(state, v_t, dt=dt)

        if i % int(0.5 / dt) == 0:
            print(
                f"t={t:5.2f}s  s={metrics['s']:.3f}  "
                f"beta={metrics['beta']:.4f}  "
                f"gain={state.o.gain:.2f}  "
                f"latency={state.o.latency_budget_ms}ms"
            )

    print("DONE")


if __name__ == "__main__":
    main()
