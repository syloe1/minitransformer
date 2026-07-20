#include "multi_head_attention.h"

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "rope/rope.h"

namespace mt {

// ===========================================================================
// Construction
// ===========================================================================

MultiHeadAttention::MultiHeadAttention(const std::size_t d_model,
                                       const std::size_t n_heads)
    : d_model_(d_model), n_heads_(n_heads), d_head_(d_model / n_heads),
      w_q_(d_model, d_model, /*has_bias=*/false),
      w_k_(d_model, d_model, /*has_bias=*/false),
      w_v_(d_model, d_model, /*has_bias=*/false),
      w_o_(d_model, d_model, /*has_bias=*/false), attention_(d_head_) {

  if (d_model % n_heads != 0) {
    std::ostringstream oss;
    oss << "MultiHeadAttention: d_model (" << d_model
        << ") must be divisible by n_heads (" << n_heads << ")";
    throw std::invalid_argument(oss.str());
  }
}

// ===========================================================================
// Forward pass
// ===========================================================================

auto MultiHeadAttention::forward(const Tensor& X,
                                 const std::size_t start_pos) const -> Tensor {

  if (X.ndim() != 2) {
    std::ostringstream oss;
    oss << "MultiHeadAttention::forward: input must be 2-D, got " << X.ndim()
        << "-D";
    throw std::invalid_argument(oss.str());
  }

  const auto &x_shape = X.shape();
  const std::size_t seq_len = x_shape[0];
  const std::size_t d_in = x_shape[1];

  if (d_in != d_model_) {
    std::ostringstream oss;
    oss << "MultiHeadAttention::forward: input last dim " << d_in
        << " != d_model " << d_model_;
    throw std::invalid_argument(oss.str());
  }

  // -- 1. Linear projections  [seq, d_model_] each ----------------------
  // QKV全局投影
  const Tensor Q = w_q_.forward(X);
  const Tensor K = w_k_.forward(X);
  const Tensor V = w_v_.forward(X);

  // -- 2. Per-head attention + concatenation ---------------------------
  // Output buffer (before W_o): [seq, d_model_]
  // 预分配拼接缓冲区
  Tensor concat(std::vector<std::size_t>{seq_len, d_model_}, 0.0f);

  for (std::size_t h = 0; h < n_heads_; ++h) {
    // --- extract head slices of Q, K, V [seq, d_head_] ----------
    const std::size_t col_start = h * d_head_;
    // 读取全局QKV, 生成单头张量
    Tensor Q_h(std::vector<std::size_t>{seq_len, d_head_});
    Tensor K_h(std::vector<std::size_t>{seq_len, d_head_});
    Tensor V_h(std::vector<std::size_t>{seq_len, d_head_});

    const float *__restrict__ q_src = Q.data();
    const float *__restrict__ k_src = K.data();
    const float *__restrict__ v_src = V.data();
    float *__restrict__ q_dst = Q_h.data();
    float *__restrict__ k_dst = K_h.data();
    float *__restrict__ v_dst = V_h.data();

    for (std::size_t i = 0; i < seq_len; ++i) {
      const std::size_t src_row = i * d_model_ + col_start;
      const std::size_t dst_row = i * d_head_;
      for (std::size_t j = 0; j < d_head_; ++j) {
        q_dst[dst_row + j] = q_src[src_row + j];
        k_dst[dst_row + j] = k_src[src_row + j];
        v_dst[dst_row + j] = v_src[src_row + j];
      }
    }

    // --- RoPE (optional) ---------------------------------------------
    if (rope_) {
      rope_->forward(Q_h, K_h, start_pos);
    }

    // --- single-head attention  → [seq, d_head_] ------------------
    const Tensor head_out = attention_.forward(Q_h, K_h, V_h);

    // --- write head output into concat buffer --------------------
    const float *__restrict__ h_src = head_out.data();
    float *__restrict__ c_dst = concat.data();

    for (std::size_t i = 0; i < seq_len; ++i) {
      const std::size_t src_row = i * d_head_;
      const std::size_t dst_row = i * d_model_ + col_start;
      for (std::size_t j = 0; j < d_head_; ++j) {
        c_dst[dst_row + j] = h_src[src_row + j];
      }
    }
  }

  // -- 3. Output projection  [seq, d_model_] ---------------------------
  return w_o_.forward(concat);
}

// ===========================================================================
// Projection layer access
// ===========================================================================

auto MultiHeadAttention::w_q() -> Linear & { return w_q_; }
auto MultiHeadAttention::w_k() -> Linear & { return w_k_; }
auto MultiHeadAttention::w_v() -> Linear & { return w_v_; }
auto MultiHeadAttention::w_o() -> Linear & { return w_o_; }

auto MultiHeadAttention::w_q() const -> const Linear & { return w_q_; }
auto MultiHeadAttention::w_k() const -> const Linear & { return w_k_; }
auto MultiHeadAttention::w_v() const -> const Linear & { return w_v_; }
auto MultiHeadAttention::w_o() const -> const Linear & { return w_o_; }

auto MultiHeadAttention::d_model() const noexcept -> std::size_t {
  return d_model_;
}
auto MultiHeadAttention::n_heads() const noexcept -> std::size_t {
  return n_heads_;
}
auto MultiHeadAttention::d_head() const noexcept -> std::size_t {
  return d_head_;
}

} // namespace mt
