#include "kv_cache.h"

#include <cstring>
#include <sstream>
#include <stdexcept>

namespace mt {
// initialize
KVCache::KVCache(const std::size_t n_layers, const std::size_t n_heads,
                 const std::size_t d_head, const std::size_t max_seq_len)
    : n_layers_(n_layers), n_heads_(n_heads), d_head_(d_head),
      max_seq_len_(max_seq_len) {

  k_cache_.resize(n_layers_);
  v_cache_.resize(n_layers_);
  for (std::size_t l = 0; l < n_layers_; ++l) {
    k_cache_[l].reserve(n_heads_);
    v_cache_[l].reserve(n_heads_);
    for (std::size_t h = 0; h < n_heads_; ++h) {
      k_cache_[l].emplace_back(std::vector<std::size_t>{max_seq_len_, d_head_},
                               0.0f);
      v_cache_[l].emplace_back(std::vector<std::size_t>{max_seq_len_, d_head_},
                               0.0f);
    }
  }
}

void KVCache::store_k(const std::size_t layer, const std::size_t head,
                      const Tensor &K, const std::size_t start_pos) {
  // 获取输入k的行数
  const auto &k_shape = K.shape();
  const std::size_t rows = k_shape[0];

  if (start_pos + rows > max_seq_len_) {
    std::ostringstream oss;
    oss << "KVCache::store_k: overflow at layer " << layer << " head " << head;
    throw std::out_of_range(oss.str());
  }

  float *__restrict__ dst = k_cache_[layer][head].data();
  const float *__restrict__ src = K.data();
  // 计算目标缓冲区偏移
  const std::size_t dst_off = start_pos * d_head_;
  std::memcpy(dst + dst_off, src, rows * d_head_ * sizeof(float));
}

void KVCache::store_v(const std::size_t layer, const std::size_t head,
                      const Tensor &V, const std::size_t start_pos) {

  const auto &v_shape = V.shape();
  const std::size_t rows = v_shape[0];

  if (start_pos + rows > max_seq_len_) {
    std::ostringstream oss;
    oss << "KVCache::store_v: overflow at layer " << layer << " head " << head;
    throw std::out_of_range(oss.str());
  }

  float *__restrict__ dst = v_cache_[layer][head].data();
  const float *__restrict__ src = V.data();

  const std::size_t dst_off = start_pos * d_head_;
  std::memcpy(dst + dst_off, src, rows * d_head_ * sizeof(float));
}

auto KVCache::get_k(const std::size_t layer, const std::size_t head) const
    -> Tensor {

  // Return a *copy* of [size(), d_head_]
  Tensor out(std::vector<std::size_t>{size_, d_head_}, 0.0f);
  if (size_ > 0) {
    const float *src = k_cache_[layer][head].data();
    float *dst = out.data();
    std::memcpy(dst, src, size_ * d_head_ * sizeof(float));
  }
  return out;
}

auto KVCache::get_v(const std::size_t layer, const std::size_t head) const
    -> Tensor {

  Tensor out(std::vector<std::size_t>{size_, d_head_}, 0.0f);
  if (size_ > 0) {
    const float *src = v_cache_[layer][head].data();
    float *dst = out.data();
    std::memcpy(dst, src, size_ * d_head_ * sizeof(float));
  }
  return out;
}

auto KVCache::size() const noexcept -> std::size_t { return size_; }

void KVCache::advance(const std::size_t n) { size_ += n; }

auto KVCache::max_seq_len() const noexcept -> std::size_t {
  return max_seq_len_;
}

} // namespace mt
