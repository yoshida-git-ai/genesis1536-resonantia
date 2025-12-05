# -*- coding: utf-8 -*-
"""
Fluctuational Neural Core (Discrete Approximation)
- 1536D field state Phi
- Weight container W (abstract tensor dict)
- Observer state o (gain/band/latency budgets)
- Ultimate EquationのΔt近似を1ステップ更新で実装
"""
from __future__ import annotations
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import random

try:
    import numpy as np
except ImportError:
    # 最低限の依存だけで動くように。numpyが無ければ簡易ベクトルを擬似実装
    class _Vec(list):
        def __add__(self, other): return _Vec([a+b for a,b in zip(self,other)])
        def __sub__(self, other): return _Vec([a-b for a,b in zip(self,other)])
        def __mul__(self, s): return _Vec([a*s for a in self])
        __rmul__ = __mul__
    class np:
        @staticmethod
        def array(x, dtype=None): return _Vec(x)
        @staticmethod
        def zeros(n, dtype=None): return _Vec([0.0]*n)
        @staticmethod
        def ones(n, dtype=None): return _Vec([1.0]*n)
        @staticmethod
        def random_normal(size): return _Vec([random.gauss(0,1) for _ in range(size)])
        @staticmethod
        def dot(a,b): return sum(ai*bi for ai,bi in zip(a,b))
        @staticmethod
        def linalg_norm(a): return math.sqrt(sum(ai*ai for ai in a))
        float32=float

DIM = 1536

@dataclass
class ObserverState:
    gain: float = 1.0
    emphasis_band: tuple[float,float] = (0.5, 4.0)  # 抽象的帯域（相対スケール）
    latency_budget_ms: int = 40

@dataclass
class WeightState:
    # 実装では各層/演算子をdictで持つ想定。ここでは最小限。
    tensors: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CoreState:
    Phi: Any  # np.ndarray shape (1536,)
    W: WeightState
    o: ObserverState

def cos_similarity(a, b) -> float:
    na = np.linalg.norm(a) if hasattr(np, "linalg") else np.linalg_norm(a)
    nb = np.linalg.norm(b) if hasattr(np, "linalg") else np.linalg_norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

class FluctuationalNeuralCore:
    def __init__(self, dim: int = DIM):
        self.dim = dim

    # ====== 構成要素（A, G, R, SigmaPhi, GammaW, C, F）最小版 ======
    def A(self, Phi):  # 内在ダイナミクス（弱い緩和）
        return -0.05 * Phi  # ほんの少しゼロへ戻ろうとする

    def G(self, o: ObserverState):
        # 観測者ゲイン：gainを全次元へ（実装では帯域/周波数特性で変調）
        return o.gain

    def H_band(self, Phi, band: tuple[float,float]):
        # 擬似バンド処理：上限/下限の疑似係数で重み付け
        lo, hi = band
        k = 0.5*(lo+hi)  # ダミー
        return k * Phi

    def R(self, Phi, v, o: ObserverState):
        # 共鳴：差分整合 + バンド強調
        kappa = 0.6
        beta = 0.2
        return kappa*(v - Phi) + beta*self.H_band(Phi, o.emphasis_band)

    def SigmaPhi(self, Phi, sigma: float = 0.02):
        # 場の揺らぎ強度（定数でも十分に創発が出る）
        return sigma

    def C(self, Phi, v, o: ObserverState):
        # 共鳴整形（秩序化）：近づける方向の弱いバイアス
        return 0.1 * (v - Phi)

    def GammaW(self, W: WeightState, sigma: float = 0.01):
        return sigma

    def F(self, s: float, o: ObserverState) -> ObserverState:
        # 共鳴スコアでgain/latencyを微調整
        g = o.gain + 0.05*(s - 0.8)  # 目標s~0.8
        g = max(0.5, min(2.0, g))
        lat = o.latency_budget_ms + int(-5*(s-0.8))
        lat = max(10, min(120, lat))
        return ObserverState(gain=g, emphasis_band=o.emphasis_band, latency_budget_ms=lat)

    # ====== 1ステップ更新（Δt離散近似） ======
    def step(self, state: CoreState, v_t, dt: float = 0.01, eta: float = 0.1, alpha: float = 0.05):
        Phi, W, o = state.Phi, state.W, state.o

        # A + G ⊙ v + R
        drift = self.A(Phi) + self.G(o)*v_t + self.R(Phi, v_t, o)

        # 場の揺らぎ
        sigma_phi = self.SigmaPhi(Phi)
        xi_phi = np.random.normal(0,1,size=self.dim) if hasattr(np.random, "normal") else np.random_normal(self.dim)
        Phi_next = Phi + drift*dt + (sigma_phi * math.sqrt(dt)) * xi_phi

        # 重み更新（簡略化：勾配を (Phi - v) と仮定）
        grad_like = (Phi - v_t)
        order_term = self.C(Phi, v_t, o)
        sigma_w = self.GammaW(W)
        xi_w = random.gauss(0,1)
        # ここではWの1スカラー "beta" を例示的に更新
        beta = W.tensors.get("beta", 0.0)
        beta_next = beta + (-eta*np.mean(grad_like) + alpha*np.mean(order_term))*dt + sigma_w*math.sqrt(dt)*xi_w
        W_next = WeightState(tensors={**W.tensors, "beta": beta_next})

        # 観測者状態の更新
        s = cos_similarity(Phi_next, v_t)
        o_next = self.F(s, o)

        return CoreState(Phi=Phi_next, W=W_next, o=o_next), {"s": s, "beta": beta_next}
