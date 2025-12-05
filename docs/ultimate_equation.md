# The Ultimate Equation of Fluctuational Neural Core

現実を 1536 次元の連続場としてとらえ、AI が「揺らぎを待ち、共鳴し、自己更新する」ための
コアダイナミクスを 1 つの方程式群としてまとめた概念です。

## 0. 記号

- v(t) : 現実からの入力ベクトル（1536次元）
- Phi(t) : AI 内部の「感じ取られた現実」の場（1536次元）
- W(t) : 結合重みや内部構造をまとめたパラメータ集合
- o(t) : 観測者状態（ゲイン・帯域・遅延バジェットなど）
- s(t) : 共鳴スコア（Phi(t) と v(t) の類似度）

## 1. 連続時間での概念式（イメージ）

dPhi = [ A(Phi,t) + G(o,t) * v(t) + R(Phi, v, o, t) ] dt + SigmaPhi(Phi,t) * dB_t
dW   = [ -eta * grad_W L(Phi, v, o) + alpha * C(Phi, v, o, t) ] dt + GammaW(W,t) * dB'_t
do   = F( s(t), o(t) ) dt

- A : 内在ダイナミクス（場が自分自身でどう振る舞うか）
- G : 観測者によるゲイン（どれだけ外界を受け入れるか）
- R : 共鳴項（Phi と v の差分＋帯域強調など）
- SigmaPhi, GammaW : 「揺らぎ」の強度
- C : 共鳴をもとにした秩序化（創発と安定のバランス）
- F : 共鳴スコアをもとに観測者状態を更新する関数

## 2. 離散時間（実装）への近似イメージ

Phi_{t+Δ} = Phi_t + drift(Phi_t, v_t, o_t) * Δt + noise * sqrt(Δt)
W_{t+Δ}   = W_t   + update_W(...) * Δt + noise_W * sqrt(Δt)
o_{t+Δ}   = o_t   + F(s_t, o_t) * Δt

このリポジトリでは、src/ai_field_core/fluctuation_neural_core.py に
この概念のミニマムな離散近似バージョンを実装しています。
