#include "embedding.h"

#include <cstddef>
#include <cstring>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace mt {

// ===========================================================================
// Internal helpers
// ===========================================================================

namespace {

[[nodiscard]] std::size_t product(const std::vector<std::size_t> &shape) {
  if (shape.empty())
    return 0;
  return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                         std::multiplies<>{});
}

} // namespace

// ===========================================================================
// Construction
// ===========================================================================

Embedding::Embedding(const std::size_t vocab_size, const std::size_t d_model)
    : vocab_size_(vocab_size), d_model_(d_model),
      weight_(std::vector<std::size_t>{vocab_size, d_model}, 0.0f) {}

// ===========================================================================
// Forward pass
// ===========================================================================

auto Embedding::forward(std::span<const std::size_t> token_ids,
                        std::vector<std::size_t> shape) const -> Tensor {

  const std::size_t num_tokens = token_ids.size();
  const std::size_t expected = product(shape);
  if (num_tokens != expected) {
    std::ostringstream oss;
    oss << "Embedding::forward: token_ids size " << num_tokens
        << " != product(shape) " << expected;
    throw std::invalid_argument(oss.str());
  }

  // Validate token IDs are in range.
  for (std::size_t i = 0; i < num_tokens; ++i) {
    // 校验所有token下标不越界
    if (token_ids[i] >= vocab_size_) {
      std::ostringstream oss;
      oss << "Embedding::forward: token_ids[" << i << "] = " << token_ids[i]
          << " >= vocab_size " << vocab_size_;
      throw std::invalid_argument(oss.str());
    }
  }

  // Output shape = input_shape + [d_model_]
  auto out_shape = shape;
  out_shape.push_back(d_model_);

  // TODO(wk): pre-allocated workspace buffer to avoid zero-init overhead
  Tensor output(out_shape, 0.0f);
  //
  float *__restrict__ out_data = output.data();
  const float *__restrict__ w = weight_.data();

  // Gather: for each token, copy its embedding row.

  for (std::size_t i = 0; i < num_tokens; ++i) {
    const std::size_t token = token_ids[i];
    const std::size_t src_off = token * d_model_;
    const std::size_t dst_off = i * d_model_;
    // Each embedding row is d_model_ contiguous floats — memcpy is the
    // right tool here (compiler will inline for small d_model).
    // 批量拷贝一整行向量， 比逐元素加法循环更快， 编译器会做批量内存拷贝优化
    std::memcpy(out_data + dst_off, w + src_off, d_model_ * sizeof(float));
  }

  return output;
}

// ===========================================================================
// Parameter access
// ===========================================================================

auto Embedding::weight() const -> const Tensor & { return weight_; }
auto Embedding::weight() -> Tensor & { return weight_; }

auto Embedding::vocab_size() const noexcept -> std::size_t {
  return vocab_size_;
}
auto Embedding::d_model() const noexcept -> std::size_t { return d_model_; }

} // namespace mt
