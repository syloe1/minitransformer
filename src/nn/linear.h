#pragma once

#include <cstddef>

#include "tensor/tensor.h"

namespace mt {

// ---------------------------------------------------------------------------
// Linear — fully-connected layer  Y = X @ W + b
// ---------------------------------------------------------------------------
// Linear is the first *stateful* module in the pipeline.  It owns the
// weight matrix W and an optional bias vector b — these are the "learned
// parameters" that would be loaded from a checkpoint in a real inference
// engine.
//
// Convention (differs from PyTorch):
//   PyTorch  nn.Linear(in, out)   W: [out, in]   Y = X @ W^T + b
//   Ours     Linear(in, out)      W: [in,  out]  Y = X @ W   + b
//
// We store W as [in, out] so that forward() maps directly to our
// existing matmul(A, B, transpose_b=false).  When importing weights
// from HuggingFace / PyTorch checkpoints the caller is responsible
// for transposing W.
//
// Typical usage:
//   Linear proj(4096, 1024);
//   proj.weight() = load_from_checkpoint(...);
//   Tensor output = proj.forward(input);   // [batch, 1024]
// ---------------------------------------------------------------------------
// 神经网络最基础的可训练算子
// Y = x * w + b
// x 输入特征  w 权重矩阵 b 偏置向量
class Linear {
public:
  // -------------------------------------------------------------------
  // Construction
  // -------------------------------------------------------------------

  /// Create a Linear layer with zero-initialised parameters.
  ///
  /// @param in_features   Input dimension  K
  /// @param out_features  Output dimension N
  /// @param has_bias      When true, an additional bias vector of shape
  ///                      [out_features] is allocated and added after
  ///                      the matrix multiplication.
  // 输入特征维度k,输出特征维度N
  explicit Linear(std::size_t in_features, std::size_t out_features,
                  bool has_bias = true);

  // -------------------------------------------------------------------
  // Forward pass
  // -------------------------------------------------------------------

  /// Y = X @ W + b
  ///
  /// @param X  Input tensor  shape [..., in_features]
  ///            Must be 2-D in the current implementation.
  /// @return   Output tensor shape [..., out_features]
  ///
  /// @throws std::invalid_argument if X.ndim() != 2
  /// @throws std::invalid_argument if X.shape()[1] != in_features_
  [[nodiscard]] auto forward(const Tensor &X) const -> Tensor;

  // -------------------------------------------------------------------
  // Parameter access
  // -------------------------------------------------------------------

  /// Weight matrix  shape [in_features_, out_features_].
  [[nodiscard]] auto weight() const -> const Tensor &;
  [[nodiscard]] auto weight() -> Tensor &;

  /// Bias vector  shape [out_features_] (empty when has_bias_ == false).
  [[nodiscard]] auto bias() const -> const Tensor &;
  [[nodiscard]] auto bias() -> Tensor &;

  /// Read-only metadata.
  [[nodiscard]] auto in_features() const noexcept -> std::size_t;
  [[nodiscard]] auto out_features() const noexcept -> std::size_t;
  [[nodiscard]] auto has_bias() const noexcept -> bool;

private:
  std::size_t in_features_;  // 输入特征维度 K
  std::size_t out_features_; // 输出特征维度 N
  bool has_bias_;            // 是否启用偏置b
  /*
    has_bias_ =  true, 计算y = xw + b
    has_bias_ = false, 计算y = xw
*/
  // X [B, k]  B是batch批量大小
  Tensor weight_; // [in_features_, out_features_] //可学习权重矩阵 [K, N]
  Tensor bias_;   // [out_features_]  or empty
  // has-bias_ = true 才有数据， 有效时形状 [N] = [out_features_]
};

} // namespace mt
