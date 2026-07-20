#pragma once

#include "tensor/tensor.h"

namespace mt {

// ---------------------------------------------------------------------------
// softmax — numerically-stable softmax along the last dimension
// ---------------------------------------------------------------------------
// Softmax converts an unconstrained vector into a probability distribution:
//   softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
//
// The naive implementation overflows for x_i > ~88.7 in float32 because
// exp(89) = +Inf.  We use the standard "subtract the max" trick:
//   softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
// which is mathematically identical (divide numerator and denominator by
// exp(max)) but keeps all exponent inputs ≤ 0, so exp never overflows.
//
// Analogy: normalising a database column before division so intermediate
// aggregates stay within representable range.
//
// This function always operates on the *last* dimension, which is the only
// axis where softmax is needed in a decoder-only Transformer (attention
// scores and, optionally, the output projection).
// ---------------------------------------------------------------------------

/// Numerically-stable softmax along the last axis.
///
/// @param X  Input tensor, ndim >= 1.
/// @return   Tensor of the same shape as X.  Each slice along the last
///           dimension is a valid probability distribution (elements in
///           [0, 1], sum ≈ 1.0).
///
/// @throws std::invalid_argument if X.ndim() == 0
// softmax算子沿张量最后一维计算稳定版softmax把实数向量转换成概率分布
[[nodiscard]] auto softmax(const Tensor &X) -> Tensor;

} // namespace mt
