#pragma once

#include <cstddef>

#include "tensor/tensor.h"

namespace mt {

// ---------------------------------------------------------------------------
// RMSNorm — Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019)
// ---------------------------------------------------------------------------
// Used in Llama instead of standard LayerNorm.  Faster: no mean subtraction,
// only scale by 1 / RMS(x).
//
//   RMSNorm(x) = x / sqrt(mean(x²) + ε)  *  γ
//
// where γ (gamma) is a learnable scale vector of shape [d].
// ---------------------------------------------------------------------------
// 只使用均方根做缩放， CPU推理更快
class RMSNorm {
public:
  /// @param d     Normalisation dimension (typically d_model).
  /// @param eps   Small constant to avoid division by zero.
  explicit RMSNorm(std::size_t d, float eps = 1e-6f);

  /// Normalise along the last dimension.
  /// @param X  [..., d]  →  output same shape.
  [[nodiscard]] auto forward(const Tensor &X) const -> Tensor;

  /// Learnable scale  shape [d].
  [[nodiscard]] auto gamma() -> Tensor &;
  [[nodiscard]] auto gamma() const -> const Tensor &;

  [[nodiscard]] auto d() const noexcept -> std::size_t;
  [[nodiscard]] auto eps() const noexcept -> float;

private:
  std::size_t d_; // 缓存归一维度， 推理不用反复传递
  float eps_;     // 防除0常数
  Tensor gamma_;  // [d_], init to 1.0 //科学性缩放权重
};

} // namespace mt
