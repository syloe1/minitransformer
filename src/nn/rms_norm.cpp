#include "rms_norm.h"

#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>

namespace mt {

// ===========================================================================
// Construction
// ===========================================================================

RMSNorm::RMSNorm(const std::size_t d, const float eps)
    : d_(d), eps_(eps), gamma_(Tensor::ones({d})) {}

// ===========================================================================
// Forward
// ===========================================================================

auto RMSNorm::forward(const Tensor &X) const -> Tensor {

  if (X.ndim() < 1) {
    throw std::invalid_argument(
        "RMSNorm::forward: input must have at least 1 dimension");
  }

  const auto &shape = X.shape();
  const std::size_t D = shape.back();

  if (D != d_) {
    std::ostringstream oss;
    oss << "RMSNorm::forward: last dim " << D << " != d " << d_;
    throw std::invalid_argument(oss.str());
  }

  const std::size_t num_rows = X.size() / D;

  // Output (copy input, normalise in-place)
  Tensor Y = X;
  float *__restrict__ y_data = Y.data();
  const float *__restrict__ g_data = gamma_.data();

  const float epsilon = eps_;
  const float inv_d = 1.0f / static_cast<float>(D);

  for (std::size_t r = 0; r < num_rows; ++r) {
    const std::size_t off = r * D; // 当前行起始偏移

    // Pass 1：求该行所有元素平方和 sum_sq
    float sum_sq = 0.0f;
    for (std::size_t j = 0; j < D; ++j) {
      const float v = y_data[off + j];
      sum_sq += v * v;
    }

    // 计算 RMS
    const float rms = std::sqrt(sum_sq * inv_d + epsilon);
    const float inv_rms = 1.0f / rms;

    // Pass 2：归一 + gamma缩放，原地写入
    for (std::size_t j = 0; j < D; ++j) {
      y_data[off + j] = y_data[off + j] * inv_rms * g_data[j];
    }
  }

  return Y;
}

// ===========================================================================
// Accessors
// ===========================================================================

auto RMSNorm::gamma() -> Tensor & { return gamma_; }
auto RMSNorm::gamma() const -> const Tensor & { return gamma_; }

auto RMSNorm::d() const noexcept -> std::size_t { return d_; }
auto RMSNorm::eps() const noexcept -> float { return eps_; }

} // namespace mt
