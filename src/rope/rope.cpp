#include "rope.h"

#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>

namespace mt {

// ===========================================================================
// Construction
// ===========================================================================
// 计算全部位置， 维度对的cos/sin三角函数表
RoPE::RoPE(const std::size_t d_head, const std::size_t max_seq_len,
           const float theta_base)
    : d_head_(d_head), max_seq_len_(max_seq_len), n_pairs_(d_head / 2),
      cos_(std::vector<std::size_t>{max_seq_len, n_pairs_}, 0.0f),
      sin_(std::vector<std::size_t>{max_seq_len, n_pairs_}, 0.0f) {

  if (d_head % 2 != 0) {
    throw std::invalid_argument("RoPE: d_head must be even");
  }

  // Precompute: freq_i = 1 / theta_base^(2*i / d_head)
  //             angle(p, i) = p * freq_i
  for (std::size_t p = 0; p < max_seq_len; ++p) {
    for (std::size_t i = 0; i < n_pairs_; ++i) {
      const float freq =
          1.0f / std::pow(theta_base, static_cast<float>(2 * i) /
                                          static_cast<float>(d_head));
      const float angle = static_cast<float>(p) * freq;
      cos_.at({p, i}) = std::cos(angle);
      sin_.at({p, i}) = std::sin(angle);
    }
  }
}

// ===========================================================================
// Forward
// ===========================================================================
// 查表原地旋转QK每一组二维维度对
void RoPE::forward(Tensor &Q, Tensor &K, const std::size_t start_pos) const {

  if (Q.ndim() != 2 || K.ndim() != 2) {
    throw std::invalid_argument("RoPE::forward: Q and K must be 2-D");
  }

  const auto &q_shape = Q.shape();
  const auto &k_shape = K.shape();
  const std::size_t seq_q = q_shape[0];
  const std::size_t d_q = q_shape[1];
  const std::size_t seq_k = k_shape[0];
  const std::size_t d_k = k_shape[1];
  // Q/K 最后一维必须等于 d\_head\_
  if (d_q != d_head_ || d_k != d_head_) {
    std::ostringstream oss;
    oss << "RoPE::forward: last dim must be d_head=" << d_head_
        << ", got Q=" << d_q << " K=" << d_k;
    throw std::invalid_argument(oss.str());
  }
  // 真实位置不能超过预计算表最大长度，防止查表越界
  if (start_pos + seq_q > max_seq_len_ || start_pos + seq_k > max_seq_len_) {
    std::ostringstream oss;
    oss << "RoPE::forward: position " << (start_pos + std::max(seq_q, seq_k))
        << " exceeds max_seq_len " << max_seq_len_;
    throw std::invalid_argument(oss.str());
  }
  // 获取原始浮点指针 + restrict优化
  float *__restrict__ q_data = Q.data();
  float *__restrict__ k_data = K.data();
  const float *__restrict__ cos_data = cos_.data();
  const float *__restrict__ sin_data = sin_.data();

  const std::size_t seq = std::max(seq_q, seq_k);

  for (std::size_t pos_idx = 0; pos_idx < seq; ++pos_idx) {
    const std::size_t p = start_pos + pos_idx;
    const std::size_t cos_off = p * n_pairs_;

    // Rotate Q row (if within seq_q)
    if (pos_idx < seq_q) {
      const std::size_t q_off = pos_idx * d_head_;
      for (std::size_t i = 0; i < n_pairs_; ++i) {
        const float c = cos_data[cos_off + i];
        const float s = sin_data[cos_off + i];
        const std::size_t idx0 = q_off + 2 * i;
        const std::size_t idx1 = q_off + 2 * i + 1;
        const float x = q_data[idx0];
        const float y = q_data[idx1];
        q_data[idx0] = x * c - y * s;
        q_data[idx1] = x * s + y * c;
      }
    }

    // Rotate K row (if within seq_k)
    if (pos_idx < seq_k) {
      const std::size_t k_off = pos_idx * d_head_;
      for (std::size_t i = 0; i < n_pairs_; ++i) {
        const float c = cos_data[cos_off + i];
        const float s = sin_data[cos_off + i];
        const std::size_t idx0 = k_off + 2 * i;
        const std::size_t idx1 = k_off + 2 * i + 1;
        const float x = k_data[idx0];
        const float y = k_data[idx1];
        k_data[idx0] = x * c - y * s;
        k_data[idx1] = x * s + y * c;
      }
    }
  }
}

// ===========================================================================
// Accessors
// ===========================================================================

auto RoPE::d_head() const noexcept -> std::size_t { return d_head_; }
auto RoPE::max_seq_len() const noexcept -> std::size_t { return max_seq_len_; }

} // namespace mt
