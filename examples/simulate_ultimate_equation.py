# -*- coding: utf-8 -*-
"""
simulate_ultimate_equation.py
- FluctuationalNeuralCore を最小条件で回し、共鳴スコア s(t) を観測
"""
import math, random
try:
    import numpy as np
except ImportError:
    import sys
    print("numpy 推奨ですが未インストールでも動作を試みます。")

from src.ai_field_core.fluctuation_neural_core import (
    FluctuationalNeuralCore, CoreState, WeightState, ObserverState, DIM
)

def gen_input_vector(t, dim=DIM):
    # 現実の入力 v(t) をダミー生成：ゆっくり回転するベクトル
    import numpy as np  # type: ignore
    base = np.zeros(dim, dtype=np.float32)
    k = int((t*3) % dim)
    base[k] = 1.0
    return base

def main():
    import numpy as np  # type: ignore
    core = FluctuationalNeuralCore(dim=DIM)
    Phi0 = np.zeros(DIM, dtype=np.float32)
    W0 = WeightState(tensors={"beta": 0.0})
    o0 = ObserverState()
    state = CoreState(Phi=Phi0, W=W0, o=o0)

    dt = 0.02
    T = 5.0
    steps = int(T/dt)
    hist = []
    for i in range(steps):
        t = i*dt
        v_t = gen_input_vector(t)
        state, metrics = core.step(state, v_t, dt=dt)
        hist.append((t, metrics["s"], metrics["beta"]))
        if i % int(0.5/dt) == 0:
            print(f"t={t:5.2f}s  s={metrics['s']:.3f}  beta={metrics['beta']:.4f}  gain={state.o.gain:.2f}  latency={state.o.latency_budget_ms}ms")

    # 最終スコア
    s_avg = sum(s for _,s,_ in hist[-int(1.0/dt):]) / max(1, int(1.0/dt))
    print(f"\n[RESULT] avg s(last 1s) = {s_avg:.3f}")

if __name__ == "__main__":
    main()
