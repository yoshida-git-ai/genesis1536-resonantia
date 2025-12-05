# -*- coding: utf-8 -*-
"""
WAVE Fluctuation Field Parameters
- 通信層からコアへ渡す、揺らぎ生成の境界条件（Seed, PhaseDrift など）
- 実運用ではQUIC/Controlメッセージから注入する想定
"""
from dataclasses import dataclass
import time
import random

@dataclass
class FluctuationField:
    seed: int
    phase_drift: float  # small drift per second
    entropy_rate: float

    @staticmethod
    def default():
        ns = time.monotonic_ns()
        seed = (ns ^ 0x9E3779B97F4A7C15) & 0xFFFFFFFF
        random.seed(seed)
        return FluctuationField(seed=seed, phase_drift=1e-3, entropy_rate=1e-2)
