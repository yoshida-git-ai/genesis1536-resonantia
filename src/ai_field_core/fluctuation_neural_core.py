# -*- coding: utf-8 -*-
"""
Fluctuational Neural Core (Discrete Approximation)
- 現実を 1536 次元ベクトルとして受け取り、
- 内部場 Phi, 重み W, 観測者状態 o を更新していくコアロジックのたたき台。

依存:
    numpy があると高速に動作します。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any
import math
import random

try:
    import numpy as np
except ImportError as e:
    raise ImportError(
        "numpy が必要です。pip install numpy でインストールしてください。"
    ) from e

DIM = 1536


@dataclass
class ObserverState:
    """観測者（Observer）の状態。どれくらい世界を受け取るか、どの帯域を重視するかなど。"""
    gain: float = 1.0
    emphasis_band: tuple[float, float] = (0.5, 4.0)  # 抽象的な「帯域」
    latency_budget_ms: int = 40                     # 許容レイテンシのイメージ


@dataclass
class WeightState:
    """重みパラメータのコンテナ。ここでは簡単のためスカラー1つだけ持つ。"""
    tensors: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoreState:
    """コア全体の状態。Phi（場）、W（重み）、o（観測者）のセット。"""
    Phi: np.ndarray
    W: WeightState
    o: ObserverState


def cos_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """コサイン類似度。Phi と v の共鳴度合いを測る。"""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class FluctuationalNeuralCore:
    """
    「究極の式」をざっくり離散化した 1 ステップ更新クラス。
    - A: 内在ダイナミクス（弱い緩和）
    - G: ゲイン（観測者状態に応じた入力の受け取り具合）
    - R: 共鳴項（差分 + 帯域強調）
    - SigmaPhi: 揺らぎの強さ
    - C: 秩序化（Phi を v に寄せる弱い力）
    - GammaW: 重みの揺らぎ
    - F: 観測者状態の更新
    """

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    # --- 内在ダイナミクス A ---
    def A(self, Phi: np.ndarray) -> np.ndarray:
        # 少しだけゼロへ戻る（緩和）
        return -0.05 * Phi

    # --- 観測者ゲイン G ---
    def G(self, o: ObserverState) -> float:
        return o.gain

    # --- 疑似バンド処理 H_band（ここでは定数係数） ---
    def H_band(self, Phi: np.ndarray, band: tuple[float, float]) -> np.ndarray:
        lo, hi = band
        k = 0.5 * (lo + hi)  # 本当は周波数帯で変調したいところを単純化
        return k * Phi

    # --- 共鳴項 R ---
    def R(self, Phi: np.ndarray, v: np.ndarray, o: ObserverState) -> np.ndarray:
        kappa = 0.6
        beta = 0.2
        return kappa * (v - Phi) + beta * self.H_band(Phi, o.emphasis_band)

    # --- 場の揺らぎ強度 SigmaPhi ---
    def SigmaPhi(self, Phi: np.ndarray, sigma: float = 0.02) -> float:
        return sigma

    # --- 共鳴整形 C（秩序化） ---
    def C(self, Phi: np.ndarray, v: np.ndarray, o: ObserverState) -> np.ndarray:
        return 0.1 * (v - Phi)

    # --- 重みの揺らぎ GammaW ---
    def GammaW(self, W: WeightState, sigma: float = 0.01) -> float:
        return sigma

    # --- 観測者状態の更新 F ---
    def F(self, s: float, o: ObserverState) -> ObserverState:
        # 目標共鳴スコア ~0.8 を目指すように gain と latency を微調整
        g = o.gain + 0.05 * (s - 0.8)
        g = max(0.5, min(2.0, g))

        lat = o.latency_budget_ms + int(-5 * (s - 0.8))
        lat = max(10, min(120, lat))

        return ObserverState(
            gain=g,
            emphasis_band=o.emphasis_band,
            latency_budget_ms=lat,
        )

    # --- 1 ステップ更新 ---
    def step(
        self,
        state: CoreState,
        v_t: np.ndarray,
        dt: float = 0.02,
        eta: float = 0.1,
        alpha: float = 0.05,
    ) -> tuple[CoreState, Dict[str, float]]:
        """
        1 ステップ分、Phi, W, o を更新する。
        v_t: 現実からの入力ベクトル（1536次元）
        dt : 時間刻み
        eta: 学習率（誤差項）
        alpha: 秩序化項の重み
        """
        Phi = state.Phi
        W = state.W
        o = state.o

        # drift = A + G * v + R
        drift = self.A(Phi) + self.G(o) * v_t + self.R(Phi, v_t, o)

        # 場の揺らぎ
        sigma_phi = self.SigmaPhi(Phi)
        xi_phi = np.random.normal(0.0, 1.0, size=self.dim)
        Phi_next = Phi + drift * dt + sigma_phi * math.sqrt(dt) * xi_phi

        # 重みの更新（ここでは簡略化して平均誤差のみ見る）
        grad_like = Phi - v_t
        order_term = self.C(Phi, v_t, o)
        grad_mean = float(np.mean(grad_like))
        order_mean = float(np.mean(order_term))

        beta = float(W.tensors.get("beta", 0.0))
        sigma_w = self.GammaW(W)
        noise_w = sigma_w * math.sqrt(dt) * random.gauss(0.0, 1.0)

        beta_next = beta + (-eta * grad_mean + alpha * order_mean) * dt + noise_w
        W_next = WeightState(tensors={**W.tensors, "beta": beta_next})

        # 観測者状態の更新
        s = cos_similarity(Phi_next, v_t)
        o_next = self.F(s, o)

        next_state = CoreState(Phi=Phi_next, W=W_next, o=o_next)
        metrics = {"s": s, "beta": beta_next}
        return next_state, metrics
