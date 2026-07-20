#include "attention.h"

#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>

#include "nn/matmul.h"
#include "nn/softmax.h"

namespace mt {

// ===========================================================================
// Construction
// ===========================================================================

Attention::Attention(const std::size_t d_k)
    : scale_(1.0f / std::sqrt(static_cast<float>(d_k))) {}

// ===========================================================================
// Forward pass
// ===========================================================================

auto Attention::forward(const Tensor &Q, const Tensor &K, const Tensor &V) const
    -> Tensor {

  // -- validation -------------------------------------------------------
  if (Q.ndim() != 2 || K.ndim() != 2 || V.ndim() != 2) {
    std::ostringstream oss;
    oss << "Attention::forward: Q, K, V must be 2-D, got Q(" << Q.ndim()
        << "D), K(" << K.ndim() << "D), V(" << V.ndim() << "D)";
    throw std::invalid_argument(oss.str());
  }

  const auto &q_shape = Q.shape(); // [seq, d_k]
  const auto &k_shape = K.shape(); // [seq, d_k]
  const auto &v_shape = V.shape(); // [seq, d_v]

  const std::size_t seq_q = q_shape[0];
  const std::size_t d_k = q_shape[1];
  const std::size_t seq_k = k_shape[0];
  const std::size_t d_k2 = k_shape[1]; // must == d_k
  const std::size_t seq_v = v_shape[0];
  // mataul(Q, k ,false) Q(K转置）
  if (d_k != d_k2) {
    std::ostringstream oss;
    oss << "Attention::forward: Q and K last dim must match — Q is [" << seq_q
        << ", " << d_k << "], K is [" << seq_k << ", " << d_k2 << "]";
    throw std::invalid_argument(oss.str());
  }
  // W V内维度
  if (seq_k != seq_v) {
    std::ostringstream oss;
    oss << "Attention::forward: K and V seq_len must match — K has " << seq_k
        << ", V has " << seq_v;
    throw std::invalid_argument(oss.str());
  }

  // -- step 1: scores = Q @ K^T   [seq_q, seq_k] -----------------------
  // 注意力原始分数
  Tensor scores = matmul(Q, K, /*transpose_b=*/true);

  // -- step 2: scale (in-place multiply; faster than division) ----------
  // 原地缩放 scores = scores / sqrt(dk)
  {
    // W SEQq , SEQk   V SEQk dv
    float *__restrict__ s = scores.data();
    const std::size_t n = scores.size();
    for (std::size_t i = 0; i < n; ++i) {
      s[i] *= scale_;
    }
  }

  // -- step 3: softmax (row-wise)  [seq_q, seq_k] ----------------------
  Tensor weights = softmax(scores); // 注意力权重

  // -- step 4: output = weights @ V   [seq_q, d_v] --------------------
  Tensor output = matmul(weights, V);

  return output;
}

// ===========================================================================
// Accessors
// ===========================================================================

auto Attention::scale() const noexcept -> float { return scale_; }

} // namespace mt
