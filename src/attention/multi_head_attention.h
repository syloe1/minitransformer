#pragma once

#include <cstddef>

#include "attention.h"
#include "nn/linear.h"
#include "tensor/tensor.h"

namespace mt {

// ---------------------------------------------------------------------------
// MultiHeadAttention — n_heads parallel scaled dot-product attentions
// ---------------------------------------------------------------------------
// MultiHeadAttention wraps the Q/K/V/O linear projections and runs `n_heads`
// independent single-head Attention computations in parallel subspaces.
//
//   d_head = d_model / n_heads        (must divide evenly)
//
// For each head h ∈ [0, n_heads):
//   1. Extract the h-th d_head-wide slice of Q, K, V
//   2. Compute scaled dot-product attention (single head)
//   3. Write the result into the corresponding slice of the output buffer
//
// After all heads finish, one final linear projection (W_o) produces the
// layer output.
//
// This module *owns* its Q/K/V/O projection weights — callers only provide
// the input tensor X.
// ---------------------------------------------------------------------------

class MultiHeadAttention {
public:
  // -------------------------------------------------------------------
  // Construction
  // -------------------------------------------------------------------

  /// @param d_model   Total model dimension.  Must be divisible by n_heads.
  /// @param n_heads   Number of attention heads.
  /// @throws std::invalid_argument if d_model % n_heads != 0
  explicit MultiHeadAttention(std::size_t d_model, std::size_t n_heads);

  // -------------------------------------------------------------------
  // Forward pass
  // -------------------------------------------------------------------

  /// Self-attention forward.
  ///
  /// @param X  Input  shape [seq, d_model]
  /// @return   Output shape [seq, d_model]
  ///
  /// @throws std::invalid_argument if X.ndim() != 2
  /// @throws std::invalid_argument if X.shape()[1] != d_model_
  // const 推理阶段只读所有权重， 不修改内部Linear
  [[nodiscard]] auto forward(const Tensor &X) const -> Tensor;

  // -------------------------------------------------------------------
  // Projection layers (mutable — for weight loading)
  // -------------------------------------------------------------------
  // 可写， 加载权重时使用
  [[nodiscard]] auto w_q() -> Linear &;
  [[nodiscard]] auto w_k() -> Linear &;
  [[nodiscard]] auto w_v() -> Linear &;
  [[nodiscard]] auto w_o() -> Linear &;

  // Const accessors
  // 只读， 推理计算使用
  [[nodiscard]] auto w_q() const -> const Linear &;
  [[nodiscard]] auto w_k() const -> const Linear &;
  [[nodiscard]] auto w_v() const -> const Linear &;
  [[nodiscard]] auto w_o() const -> const Linear &;

  // Metadata
  [[nodiscard]] auto d_model() const noexcept -> std::size_t;
  [[nodiscard]] auto n_heads() const noexcept -> std::size_t;
  [[nodiscard]] auto d_head() const noexcept -> std::size_t;

private:
  // 超参缓存， 避免重复除法
  // d_model = n_heads * d_head 每个头单独计算Q K V切片 [seq, d_head]
  std::size_t d_model_;
  std::size_t n_heads_;
  std::size_t d_head_;
  // 带权重投影层
  Linear w_q_; // [d_model_, d_model_]
  Linear w_k_; // [d_model_, d_model_]
  Linear w_v_; // [d_model_, d_model_]
  Linear w_o_; // [d_model_, d_model_]
  // Attetion 工具实例， 供所有头循环复用
  Attention attention_; // single-head, d_k = d_head_
};

} // namespace mt
