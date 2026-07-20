#include "softmax.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace mt {
/*
raw vector[x0,x1,..,xd-1]

遍历求max_val

算exp(xj - max) 存入输出， 累加sum


每个值/sum 归一化


改行概率分布完成

*/
auto softmax(const Tensor &X) -> Tensor {
  // 0维度标量不存在最后一维， 无法做softmax
  if (X.ndim() == 0) {
    throw std::invalid_argument(
        "softmax: input must have at least 1 dimension");
  }

  const auto &shape = X.shape();
  const std::size_t D = shape.back();        // 最后一维长度（要做softmax的轴）
  const std::size_t num_rows = X.size() / D; // 有多少个独立向量切片

  // Allocate output (same shape, initially a copy of input — we'll
  // overwrite every element anyway, so we could skip the init, but
  // the copy constructor gives us correct shape/strides for free).
  Tensor Y(shape);
  // 拿到原始浮点指针， 开启编译器优化
  const float *__restrict__ src = X.data();
  float *__restrict__ dst = Y.data();

  // Process each row independently.
  for (std::size_t r = 0; r < num_rows; ++r) {
    const std::size_t off = r * D;

    // ---- Pass 1: find the maximum in this row -------------------
    float max_val = -std::numeric_limits<float>::infinity();
    for (std::size_t j = 0; j < D; ++j) {
      const float v = src[off + j];
      if (v > max_val)
        max_val = v;
    }

    // ---- Pass 2: exp(x - max) and accumulate sum ----------------
    float sum = 0.0f;
    for (std::size_t j = 0; j < D; ++j) {
      const float v = std::exp(src[off + j] - max_val);
      dst[off + j] = v;
      sum += v;
    }

    // ---- Pass 3: normalise (divide by sum) ----------------------
    for (std::size_t j = 0; j < D; ++j) {
      dst[off + j] /= sum;
    }
  }

  return Y;
}

} // namespace mt
