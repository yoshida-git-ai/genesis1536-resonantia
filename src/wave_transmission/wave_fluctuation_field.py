# -*- coding: utf-8 -*-
"""
WAVE Fluctuation Field Parameters

通信層（WAVEプロトコル）側からコアへ渡す「揺らぎに関する境界条件」を表すクラス。
- seed         : ノイズ生成の種
- phase_drift  : 位相ドリフトの大きさ（1秒あたり）
- entropy_rate : 情報エネルギー（揺らぎ）の変化率イメージ
"""

from dataclasses import dataclass
import time
import random


@dataclass
class FluctuationField:
    seed: int
    phase_drift: float
    entropy_rate: float

    @staticmethod
    def default() -> "FluctuationField":
        """
        単純なデフォルト実装。
        - 現在時刻から seed を作り
        - 小さな位相ドリフトとエントロピー変化率を与える
        """
        ns = time.monotonic_ns()
        seed = (ns ^ 0x9E3779B97F4A7C15) & 0xFFFFFFFF
        random.seed(seed)
        return FluctuationField(
            seed=seed,
            phase_drift=1e-3,
            entropy_rate=1e-2,
        )
