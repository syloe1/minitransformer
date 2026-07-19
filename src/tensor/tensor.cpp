#include "tensor.h"

#include <cassert>
#include <numeric>
#include <sstream>
#include <utility>

namespace mt {

// ===========================================================================
// Internal helpers  (anonymous namespace → TU-local linkage)
// ===========================================================================

namespace {

/// Compute total element count from shape.
[[nodiscard]] std::size_t product(const std::vector<std::size_t> &shape) {
  if (shape.empty())
    return 0;
  return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                         std::multiplies<>{});
}

/// Compute row-major strides from shape.
// strides把多维坐标换算成底层一维data_数组的偏移下标

[[nodiscard]] std::vector<std::size_t>
compute_strides(const std::vector<std::size_t> &shape) {
  if (shape.empty())
    return {};
  std::vector<std::size_t> s(shape.size());
  s.back() = 1;
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
    // 当前步长等与 右边维度尺寸  * 右边步长
    s[static_cast<std::size_t>(i)] = s[static_cast<std::size_t>(i) + 1] *
                                     shape[static_cast<std::size_t>(i) + 1];
  }
  return s;
}
// 把维度数组shape格式化打印成易读字符串
/// Format shape as "[a, b, c]" for error messages.
[[nodiscard]] std::string format_shape(const std::vector<std::size_t> &shape) {
  std::ostringstream oss;
  oss << '[';
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i > 0)
      oss << ", ";
    oss << shape[i];
  }
  oss << ']';
  return oss.str();
}

} // namespace

// ===========================================================================
// Construction
// ===========================================================================

Tensor::Tensor(std::vector<std::size_t> shape)
    // 参数列表初始化
    : shape_(std::move(shape)), strides_(compute_strides(shape_)) {
  const auto n = product(shape_);
  if (n > 0) {
    // 给底层一维数组开辟内存
    data_.resize(n);
  }
}

Tensor::Tensor(std::vector<std::size_t> shape, const float init_value)
    : shape_(std::move(shape)), strides_(compute_strides(shape_)) {
  const auto n = product(shape_);
  if (n > 0) {
    data_.assign(n, init_value);
  }
}

// ---------------------------------------------------------------------------
// Copy
// ---------------------------------------------------------------------------

Tensor::Tensor(const Tensor &other)
    // 参数列表初始化
    : data_(other.data_), shape_(other.shape_), strides_(other.strides_) {}

Tensor &Tensor::operator=(const Tensor &other) {
  if (this != &other) {
    data_ = other.data_;
    shape_ = other.shape_;
    strides_ = other.strides_;
  }
  return *this;
}

// ---------------------------------------------------------------------------
// Move
// ---------------------------------------------------------------------------

Tensor::Tensor(Tensor &&other)
    : data_(std::exchange(other.data_, {})),
      shape_(std::exchange(other.shape_, {})),
      strides_(std::exchange(other.strides_, {})) {}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
  if (this != &other) {
    data_ = std::exchange(other.data_, {});
    shape_ = std::exchange(other.shape_, {});
    strides_ = std::exchange(other.strides_, {});
  }
  return *this;
}

// ===========================================================================
// Static factories
// ===========================================================================

auto Tensor::zeros(std::vector<std::size_t> shape) -> Tensor {
  return Tensor{std::move(shape), 0.0f};
}

auto Tensor::ones(std::vector<std::size_t> shape) -> Tensor {
  return Tensor{std::move(shape), 1.0f};
}

auto Tensor::from_data(std::vector<std::size_t> shape, std::vector<float> data)
    -> Tensor {
  const auto expected = product(shape);
  if (data.size() != expected) {
    std::ostringstream oss;
    oss << "Tensor::from_data: data size " << data.size()
        << " != shape product " << expected << " for shape "
        << format_shape(shape);
    throw std::invalid_argument(oss.str());
  }
  Tensor t;
  t.data_ = std::move(data);
  t.shape_ = std::move(shape);
  t.strides_ = compute_strides(t.shape_);
  return t;
}

// ===========================================================================
// Metadata
// ===========================================================================

auto Tensor::shape() const noexcept -> const std::vector<std::size_t> & {
  return shape_;
}

auto Tensor::strides() const noexcept -> const std::vector<std::size_t> & {
  return strides_;
}

auto Tensor::ndim() const noexcept -> std::size_t { return shape_.size(); }

auto Tensor::size() const noexcept -> std::size_t { return data_.size(); }

auto Tensor::byte_size() const noexcept -> std::size_t {
  return data_.size() * sizeof(float);
}

auto Tensor::empty() const noexcept -> bool { return data_.empty(); }

// ---------------------------------------------------------------------------
// Offset calculation
// ---------------------------------------------------------------------------
// 算出该位置在底层一维data_数组的偏移量
auto Tensor::offset(const std::vector<std::size_t> &idx) const -> std::size_t {
  validate_index_dims(idx);
  std::size_t off = 0;
  for (std::size_t d = 0; d < idx.size(); ++d) {
    if (idx[d] >= shape_[d]) {
      std::ostringstream oss;
      oss << "Tensor::offset: index " << format_shape(idx)
          << " out of range for shape " << format_shape(shape_);
      throw std::out_of_range(oss.str());
    }
    // 下标 * 当前维度步长
    // total_offset = sigma(idx[d] * strides[d] for d from 0 to ndim - 1)
    off += idx[d] * strides_[d];
  }
  return off;
}

auto Tensor::offset_unchecked(
    const std::vector<std::size_t> &idx) const noexcept -> std::size_t {
  std::size_t off = 0;
  for (std::size_t d = 0; d < idx.size(); ++d) {
    off += idx[d] * strides_[d];
  }
  return off;
}

// ===========================================================================
// Element access
// ===========================================================================

auto Tensor::operator[](const std::size_t i) -> float & {
  assert(i < data_.size());
  return data_[i];
}

auto Tensor::operator[](const std::size_t i) const -> const float & {
  assert(i < data_.size());
  return data_[i];
}

auto Tensor::at(const std::vector<std::size_t> &idx) -> float & {
  return data_[offset(idx)];
}

auto Tensor::at(const std::vector<std::size_t> &idx) const -> const float & {
  return data_[offset(idx)];
}

// ===========================================================================
// Raw pointer / span
// ===========================================================================

auto Tensor::data() noexcept -> float * { return data_.data(); }

auto Tensor::data() const noexcept -> const float * { return data_.data(); }

auto Tensor::span() noexcept -> std::span<float> { return std::span{data_}; }

auto Tensor::span() const noexcept -> std::span<const float> {
  return std::span{data_};
}

// ===========================================================================
// Shape mutation
// ===========================================================================
// 原地修改shape
void Tensor::reshape(std::vector<std::size_t> new_shape) {
  const auto n = product(new_shape);
  if (n != data_.size()) {
    std::ostringstream oss;
    oss << "Tensor::reshape: cannot reshape " << format_shape(shape_)
        << " (size=" << data_.size() << ") to " << format_shape(new_shape)
        << " (size=" << n << ")";
    throw std::invalid_argument(oss.str());
  }
  shape_ = std::move(new_shape);
  strides_ = compute_strides(shape_);
}

// ===========================================================================
// Comparison
// ===========================================================================

auto Tensor::operator==(const Tensor &other) const -> bool {
  if (shape_ != other.shape_)
    return false;
  return data_ == other.data_;
}

auto Tensor::operator!=(const Tensor &other) const -> bool {
  return !(*this == other);
}

// ===========================================================================
// Private helpers
// ===========================================================================

void Tensor::recompute_strides() { strides_ = compute_strides(shape_); }

void Tensor::validate_index_dims(const std::vector<std::size_t> &idx) const {
  if (idx.size() != shape_.size()) {
    std::ostringstream oss;
    oss << "Tensor: index has " << idx.size() << " dimensions, "
        << "but tensor has " << shape_.size() << " dimensions";
    throw std::out_of_range(oss.str());
  }
}

} // namespace mt
