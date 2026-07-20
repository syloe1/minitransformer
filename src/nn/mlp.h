#pragma once

#include <cstddef>

#include "linear.h"
#include "tensor/tensor.h"

namespace mt {

// ---------------------------------------------------------------------------
// SiLU — Sigmoid Linear Unit  (also called "swish")
// ---------------------------------------------------------------------------
//   SiLU(x) = x · σ(x) = x / (1 + exp(-x))
//
// This is the activation used inside SwiGLU.  It is smooth (unlike ReLU),
// non-monotonic for small negative x, and tends to work better than GELU
// in recent large language models (Llama, PaLM).
// ---------------------------------------------------------------------------

/// Element-wise SiLU activation.
[[nodiscard]] auto silu(const Tensor &x) -> Tensor;

// ---------------------------------------------------------------------------
// MLP — SwiGLU Feed-Forward Network  (Llama-style)
// ---------------------------------------------------------------------------
//   gate = SiLU(X @ W_gate)
//   up   = X @ W_up
//   out  = (gate ⊙ up) @ W_down
//
// This is the "channel-wise" complement to Attention.  While Attention
// mixes information *across* tokens, the MLP transforms each token
// independently through a wider hidden dimension.
//
// d_hidden defaults to 8/3 · d_model, matching the Llama convention.
// ---------------------------------------------------------------------------

class MLP {
public:
  // -------------------------------------------------------------------
  // Construction
  // -------------------------------------------------------------------

  /// @param d_model   Input/output dimension.
  /// @param d_hidden  Hidden dimension (0 = auto: d_model * 8 / 3).
  explicit MLP(std::size_t d_model, std::size_t d_hidden = 0);

  // -------------------------------------------------------------------
  // Forward pass
  // -------------------------------------------------------------------

  /// @param X  Input  shape [seq, d_model_]
  /// @return   Output shape [seq, d_model_]
  /// @throws std::invalid_argument if X.ndim() != 2 or dim mismatch
  [[nodiscard]] auto forward(const Tensor &X) const -> Tensor;

  // -------------------------------------------------------------------
  // Weight access (mutable — for checkpoint loading)
  // -------------------------------------------------------------------
  // 非const, 加载权重时写入
  [[nodiscard]] auto w_gate() -> Linear &;
  [[nodiscard]] auto w_up() -> Linear &;
  [[nodiscard]] auto w_down() -> Linear &;

  [[nodiscard]] auto w_gate() const -> const Linear &;
  [[nodiscard]] auto w_up() const -> const Linear &;
  [[nodiscard]] auto w_down() const -> const Linear &;

  [[nodiscard]] auto d_model() const noexcept -> std::size_t;
  [[nodiscard]] auto d_hidden() const noexcept -> std::size_t;

private:
  // 两个超参缓存
  std::size_t d_model_; // 输入输出特征维度
  std::size_t d_hidden_;

  Linear w_gate_; // [d_model_, d_hidden_]
  Linear w_up_;   // [d_model_, d_hidden_]
  Linear w_down_; // [d_hidden_, d_model_]
};

} // namespace mt
