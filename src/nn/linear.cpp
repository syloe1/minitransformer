#include "linear.h"

#include <cstddef>
#include <sstream>
#include <stdexcept>

#include "matmul.h"

namespace mt {

// ===========================================================================
// Construction
// ===========================================================================

Linear::Linear(const std::size_t in_features, const std::size_t out_features,
               const bool has_bias)
    : in_features_(in_features), out_features_(out_features),
      has_bias_(has_bias),
      weight_(std::vector<std::size_t>{in_features, out_features}, 0.0f),
      bias_(has_bias ? Tensor(std::vector<std::size_t>{out_features}, 0.0f)
                     : Tensor{}) {}

// ===========================================================================
// Forward pass
// ===========================================================================

auto Linear::forward(const Tensor &X) const -> Tensor {
  if (X.ndim() != 2) {
    std::ostringstream oss;
    oss << "Linear::forward: input must be 2-D, got " << X.ndim()
        << "-D shape [";
    const auto &sh = X.shape();
    for (std::size_t d = 0; d < sh.size(); ++d) {
      if (d > 0)
        oss << ", ";
      oss << sh[d];
    }
    oss << ']';
    throw std::invalid_argument(oss.str());
  }
  // Xmk Wkn
  const auto &x_shape = X.shape();
  const std::size_t M = x_shape[0];
  const std::size_t K = x_shape[1];
  // 内维数要相等
  if (K != in_features_) {
    std::ostringstream oss;
    oss << "Linear::forward: input last dim " << K << " != in_features "
        << in_features_;
    throw std::invalid_argument(oss.str());
  }

  // Y0 = X @ W   →  [M, out_features_]
  Tensor Y = matmul(X, weight_);

  // Add bias:  Y[i][j] += bias[j]
  if (has_bias_) {
    const float *__restrict__ b = bias_.data();
    float *__restrict__ y = Y.data();

    for (std::size_t i = 0; i < M; ++i) {
      // 张量是行优先连续内存
      const std::size_t row_off = i * out_features_;
      for (std::size_t j = 0; j < out_features_; ++j) {
        y[row_off + j] += b[j];
      }
    }
  }

  return Y;
}

// ===========================================================================
// Parameter access
// ===========================================================================

auto Linear::weight() const -> const Tensor & { return weight_; }
auto Linear::weight() -> Tensor & { return weight_; }

auto Linear::bias() const -> const Tensor & { return bias_; }
auto Linear::bias() -> Tensor & { return bias_; }

auto Linear::in_features() const noexcept -> std::size_t {
  return in_features_;
}
auto Linear::out_features() const noexcept -> std::size_t {
  return out_features_;
}
auto Linear::has_bias() const noexcept -> bool { return has_bias_; }

} // namespace mt
