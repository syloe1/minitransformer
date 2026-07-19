#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
// RAII自动内存管理
// 行优先C-Order布局 CPU缓存友好
// 预计算strides步长, 多维转1维O（1）计算
namespace mt {

// ---------------------------------------------------------------------------
// Tensor — multi-dimensional array, row-major, owns its data
// ---------------------------------------------------------------------------
// This is the fundamental data container for the entire mini-transformer.
// Every module (Embedding, Attention, MLP, LayerNorm, ...) reads from and
// writes to Tensors.
//
// Design principles:
//   1. RAII ownership via std::vector<float>  — no manual new/delete
//   2. Row-major (C-order) layout             — matches CPU cache lines
//   3. Precomputed strides                    — O(1) per-element index cost
//   4. std::span views                        — zero-copy hand-off to
//      downstream compute kernels
//   5. Move-enabled                           — cheap transfer of ownership
//   6. Static factories (zeros/ones/from_data)— explicit intent over
//      constructor overloading
// ---------------------------------------------------------------------------
// tiny Transformer框架唯一多维浮点容器, Embedding, SelfAttention.MLP,
// LayerNorm, Softmax全部模块输入用它承载
class Tensor {
public:
  // 无参默认构造
  Tensor() = default;
  // 指定形状，未初始化内存
  explicit Tensor(std::vector<std::size_t> shape);
  // 指定形状 + 填充固定值
  Tensor(std::vector<std::size_t> shape, float init_value);
  ~Tensor() = default;
  Tensor(const Tensor &other);
  Tensor &operator=(const Tensor &other);

  Tensor(Tensor &&other);
  Tensor &operator=(Tensor &&other) noexcept;
  // 推荐创建张量
  static auto zeros(std::vector<std::size_t> shape) -> Tensor;
  static auto ones(std::vector<std::size_t> shape) -> Tensor;

  // 接管外围已有一维浮点数组， 构造多维张量
  static auto from_data(std::vector<std::size_t> shape, std::vector<float> data)
      -> Tensor;
  // 返回张量维度数组
  [[nodiscard]] auto shape() const noexcept -> const std::vector<std::size_t> &;
  // 返回预计算的步长数组
  [[nodiscard]] auto strides() const noexcept
      -> const std::vector<std::size_t> &;
  // 返回维度数量
  [[nodiscard]] auto ndim() const noexcept -> std::size_t;
  // 返回张量总元素个数
  [[nodiscard]] auto size() const noexcept -> std::size_t;
  // 整个张量占用多少字节内存
  [[nodiscard]] auto byte_size() const noexcept -> std::size_t;
  // 判断张量是否为空
  [[nodiscard]] auto empty() const noexcept -> bool;
  // 输入多维坐标{i, j, k}算出在一维data数组里的线性下标
  [[nodiscard]] auto offset(const std::vector<std::size_t> &idx) const
      -> std::size_t;
  // 同样计算多维坐标偏移， 完全无边界校验
  [[nodiscard]] auto
  offset_unchecked(const std::vector<std::size_t> &idx) const noexcept
      -> std::size_t;

  // 扁平化一维直接读写张量底层float数组
  auto operator[](std::size_t i) -> float &;             // 可修改元素
  auto operator[](std::size_t i) const -> const float &; // 只读， 不能修改

  // 多维安全访问
  auto at(const std::vector<std::size_t> &idx) -> float &;
  auto at(const std::vector<std::size_t> &idx) const -> const float &;

  // data() 裸指针接口
  [[nodiscard]] auto data() noexcept -> float *;
  [[nodiscard]] auto data() const noexcept -> const float *; // 只读
  // span零拷贝视图， span不持有内存， 只包裹一段连续数组的轻量视图

  [[nodiscard]] auto span() noexcept -> std::span<float>;
  [[nodiscard]] auto span() const noexcept -> std::span<const float>;

  // 原地修改shape, strides,底层float数据完全不移动O(1)
  void reshape(std::vector<std::size_t> new_shape);

  // 张量相等比较
  auto operator==(const Tensor &other) const -> bool;
  auto operator!=(const Tensor &other) const -> bool;

private:
  // 每次shape修改自动调用， 重新生成strides数组
  void recompute_strides();
  // 校验多维索引数组长度和ndim匹配， 维度数量不对直接抛异常
  void validate_index_dims(const std::vector<std::size_t> &idx) const;
  std::vector<float> data_;          // 一维数组存数据
  std::vector<std::size_t> shape_;   // dimension, [N, seq_len, hidden_dim]这种
  std::vector<std::size_t> strides_; // 每一维跨多少元素才能到下一维度
};
} // namespace mt
