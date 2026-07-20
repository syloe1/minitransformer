#pragma once

#include <cstddef>

#include "attention/multi_head_attention.h"
#include "nn/mlp.h"
#include "nn/rms_norm.h"
#include "rope/rope.h"
#include "tensor/tensor.h"

namespace mt {

// ---------------------------------------------------------------------------
// DecoderBlock — one Transformer layer  (Llama-style, pre-norm)
// ---------------------------------------------------------------------------
// Data flow:
//   X ──→ RMSNorm ──→ MHA (+RoPE) ──→ + ──→ RMSNorm ──→ MLP ──→ + ──→ out
//        (pre-norm)     (attention)    ↑    (pre-norm)   (ffn)    ↑
//                                       └── X (residual)          └── X
//                                       (residual)
//
// This is the fundamental building block that is stacked N times
// to form the complete Transformer.
// ---------------------------------------------------------------------------

class DecoderBlock {
public:
  // -------------------------------------------------------------------
  // Construction
  // -------------------------------------------------------------------

  /// @param d_model      Model dimension.
  /// @param n_heads      Number of attention heads.
  /// @param d_hidden     MLP hidden dim (0 = auto: d_model * 8 / 3).
  /// @param max_seq_len  RoPE precompute length.
  explicit DecoderBlock(std::size_t d_model, std::size_t n_heads,
                        std::size_t d_hidden = 0,
                        std::size_t max_seq_len = 2048);

  // -------------------------------------------------------------------
  // Forward pass
  // -------------------------------------------------------------------

  /// @param X  Input  shape [seq, d_model_]
  /// @param start_pos  RoPE offset (0 = prefill).
  /// @return   Output shape [seq, d_model_]
  [[nodiscard]] auto forward(const Tensor &X, std::size_t start_pos = 0) const
      -> Tensor;

  // -------------------------------------------------------------------
  // Sub-module access (for weight loading)
  // -------------------------------------------------------------------

  [[nodiscard]] auto mha() -> MultiHeadAttention &;
  [[nodiscard]] auto mlp() -> MLP &;
  [[nodiscard]] auto norm1() -> RMSNorm &;
  [[nodiscard]] auto norm2() -> RMSNorm &;
  [[nodiscard]] auto rope() -> RoPE &;
  // 维度查询接口
  [[nodiscard]] auto d_model() const noexcept -> std::size_t;

private:
  std::size_t d_model_;    // 传入模型维度
  RMSNorm norm1_;          // 注意力前归一化
  MultiHeadAttention mha_; // 多头注意力
  RMSNorm norm2_;          // MLP前归一化
  MLP mlp_;                // 前馈
  RoPE rope_;              // 位置编码算子
};

} // namespace mt
