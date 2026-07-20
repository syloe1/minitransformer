#pragma once

#include <cstddef>

#include "tensor/tensor.h"

namespace mt {

// ---------------------------------------------------------------------------
// Attention — scaled dot-product attention (single head)
// ---------------------------------------------------------------------------
// This is the computational heart of the Transformer.  Given Query, Key,
// and Value tensors it computes:
//
//   scores  = Q * K^T / sqrt(d_k)
//   weights = softmax(scores)            (row-wise)
//   output  = weights * V
//
// The module is deliberately *stateless beyond the scale factor* — it
// receives pre-projected Q / K / V from the caller (typically a
// DecoderBlock that owns the Linear projection layers).  This keeps
// Attention focused on the core attention algorithm and lets the caller
// decide whether Q ≡ K ≡ V (self-attention) or Q ≠ K ≠ V (cross-attention).
// Q K V由外部Linear层提前投射
// Analogy:  Attention is a parameterised SQL query.
//   Q — the "WHERE" clause template
//   K — the index each row is looked up by
//   V — the payload returned
//   softmax(Q·K^T / √d_k) — a fuzzy JOIN condition
// ---------------------------------------------------------------------------

class Attention {
public:
  // -------------------------------------------------------------------
  // Construction
  // -------------------------------------------------------------------

  /// @param d_k  Dimension of the Query / Key vectors.
  ///             The scale factor 1/√d_k is precomputed once here.
  explicit Attention(std::size_t d_k);

  // -------------------------------------------------------------------
  // Forward pass
  // -------------------------------------------------------------------

  /// Scaled dot-product attention.
  ///
  /// @param Q  Query tensor   shape [seq, d_k]
  /// @param K  Key tensor     shape [seq, d_k]
  /// @param V  Value tensor   shape [seq, d_v]
  /// @return   Output tensor  shape [seq, d_v]
  ///
  /// @throws std::invalid_argument if Q, K, V are not 2-D
  /// @throws std::invalid_argument if Q.shape()[1] != K.shape()[1]
  /// @throws std::invalid_argument if Q.shape()[0] != V.shape()[0]
  [[nodiscard]] auto forward(const Tensor &Q, const Tensor &K,
                             const Tensor &V) const -> Tensor;

  /// Precomputed scale factor.
  [[nodiscard]] auto scale() const noexcept -> float;

private:
  float scale_; // 1.0f / std::sqrt(static_cast<float>(d_k)) //无任何可学习权重
};

} // namespace mt
