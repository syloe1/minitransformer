#pragma once

#include <cstddef>
#include <vector>

#include "tensor/tensor.h"

namespace mt {

// ---------------------------------------------------------------------------
// KVCache — Key-Value cache for autoregressive generation
// ---------------------------------------------------------------------------
// During decode, each new token only needs to attend to its own Q against
// *all previous* K and V.  Without a cache we would recompute K/V for the
// entire sequence every step — O(N²) work.
//
// KVCache stores K and V per (layer, head) as a pre-allocated buffer of
// shape [max_seq_len, d_head].  Each decode step appends one row.
//
// Prefill:  store all N positions at once.
// Decode:   store one position at a time.
//
// Shapes:
//   K/V buffers:  [max_seq_len_, d_head_]  per (layer, head)
//   Retrieved:    [size(), d_head_]          (the filled portion)
// ---------------------------------------------------------------------------

class KVCache {
public:
  /// @param n_layers     Number of DecoderBlocks.
  /// @param n_heads      Attention heads per layer.
  /// @param d_head       Dimension per head.
  /// @param max_seq_len  Maximum sequence length.
  KVCache(std::size_t n_layers, std::size_t n_heads, std::size_t d_head,
          std::size_t max_seq_len);
  // 写入缓存
  /// Store one head's K at a specific position.
  /// @param K  [1, d_head_] for decode, or [N, d_head_] for prefill.
  void store_k(std::size_t layer, std::size_t head, const Tensor &K,
               std::size_t start_pos);

  /// Store one head's V at a specific position.
  void store_v(std::size_t layer, std::size_t head, const Tensor &V,
               std::size_t start_pos);

  /// Retrieve the cached K for a (layer, head).
  /// Returns a *copy* of the filled region  shape [size(), d_head_].
  [[nodiscard]] auto get_k(std::size_t layer, std::size_t head) const -> Tensor;

  /// Retrieve the cached V for a (layer, head).
  [[nodiscard]] auto get_v(std::size_t layer, std::size_t head) const -> Tensor;

  /// Number of tokens currently cached.
  [[nodiscard]] auto size() const noexcept -> std::size_t;

  /// Increment the cached token counter (call after store_k/store_v).
  // 更新缓存计数
  void advance(std::size_t n = 1);

  [[nodiscard]] auto max_seq_len() const noexcept -> std::size_t;

private:
  std::size_t n_layers_;    // 解码器层数
  std::size_t n_heads_;     // 单层注意力头数量
  std::size_t d_head_;      // 单头维度
  std::size_t max_seq_len_; // 缓存最大容器
  std::size_t size_ = 0;

  // k_cache_[layer][head]  shape [max_seq_len_, d_head_]
  std::vector<std::vector<Tensor>> k_cache_; // 当前层
  std::vector<std::vector<Tensor>> v_cache_;
};

} // namespace mt
