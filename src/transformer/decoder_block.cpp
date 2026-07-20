#include "decoder_block.h"

#include <cstddef>
#include <sstream>
#include <stdexcept>

namespace mt {

// ===========================================================================
// Construction
// ===========================================================================

DecoderBlock::DecoderBlock(const std::size_t d_model, const std::size_t n_heads,
                           const std::size_t d_hidden,
                           const std::size_t max_seq_len)
    : d_model_(d_model), norm1_(d_model), mha_(d_model, n_heads),
      norm2_(d_model), mlp_(d_model, d_hidden),
      rope_(d_model / n_heads, max_seq_len) {

  // Wire RoPE into MHA
  mha_.set_rope(&rope_);
}

// ===========================================================================
// Forward
// ===========================================================================

auto DecoderBlock::forward(const Tensor &X, const std::size_t start_pos) const
    -> Tensor {

  if (X.ndim() != 2 || X.shape()[1] != d_model_) {
    std::ostringstream oss;
    oss << "DecoderBlock::forward: expected [*, " << d_model_ << "], got [";
    const auto &sh = X.shape();
    for (std::size_t d = 0; d < sh.size(); ++d) {
      if (d > 0)
        oss << ", ";
      oss << sh[d];
    }
    oss << ']';
    throw std::invalid_argument(oss.str());
  }

  // -- Attention sub-block (pre-norm + residual) -----------------------
  Tensor attn_in = norm1_.forward(X);
  Tensor attn_out = mha_.forward(attn_in, start_pos);

  // Residual: Y = X + attn_out
  // 残差相加， 纯CPU循环
  // ————restrict__ C++编译器提示， 编译器可做向量化优化
  Tensor hidden(X.shape(), 0.0f);
  {
    const float *__restrict__ x = X.data();
    const float *__restrict__ ao = attn_out.data();
    float *__restrict__ h = hidden.data();
    const std::size_t n = hidden.size();
    for (std::size_t i = 0; i < n; ++i) {
      h[i] = x[i] + ao[i];
    }
  }

  // -- FFN sub-block (pre-norm + residual) -----------------------------
  Tensor ffn_in = norm2_.forward(hidden);
  Tensor ffn_out = mlp_.forward(ffn_in);

  // Residual: Y = hidden + ffn_out
  Tensor output(X.shape(), 0.0f);
  {
    const float *__restrict__ h = hidden.data();
    const float *__restrict__ fo = ffn_out.data();
    float *__restrict__ o = output.data();
    const std::size_t n = output.size();
    for (std::size_t i = 0; i < n; ++i) {
      o[i] = h[i] + fo[i];
    }
  }

  return output;
}

// ===========================================================================
// Accessors
// ===========================================================================

auto DecoderBlock::mha() -> MultiHeadAttention & { return mha_; }
auto DecoderBlock::mlp() -> MLP & { return mlp_; }
auto DecoderBlock::norm1() -> RMSNorm & { return norm1_; }
auto DecoderBlock::norm2() -> RMSNorm & { return norm2_; }
auto DecoderBlock::rope() -> RoPE & { return rope_; }

auto DecoderBlock::d_model() const noexcept -> std::size_t { return d_model_; }

} // namespace mt
