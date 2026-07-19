#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "tensor/tensor.h"

namespace mt {

// ---------------------------------------------------------------------------
// Embedding — discrete token IDs → dense float vectors
// ---------------------------------------------------------------------------
// Embedding is the first layer of a Transformer.  It converts a sequence
// of integer token IDs (e.g. [154, 3872, 421]) into a dense float tensor
// of shape [batch, seq_len, d_model] by performing a *gather* over the
// embedding table weight_.
//
// Analogy: a database clustered-index lookup.
//   SELECT row FROM weight_ WHERE row_index IN (token_ids)
// The table weight_ has shape [vocab_size, d_model]; for each token ID
// we copy one contiguous row of d_model floats into the output.
//
// Token IDs are passed as std::span<const std::size_t> rather than as a
// Tensor because our Tensor currently only holds float32 data.  Adding
// an integer-Tensor variant would be over-engineering at this stage.
// --------------------
// -------------------------------------------------------
// Embedding 是 Decoder Transformer第一层， 做离散tokenID -> 连续稠密向量映射，
// 内部嵌入表weight_, 形状[vocab_size, d_model] 每行对应一个token的特征向量

class Embedding {
public:
  // -------------------------------------------------------------------
  // Construction
  // -------------------------------------------------------------------

  /// Create an embedding table with zero-initialised weights.
  ///
  /// @param vocab_size  Number of tokens in the vocabulary.
  /// @param d_model     Dimension of each embedding vector.
  explicit Embedding(std::size_t vocab_size, std::size_t d_model);

  // -------------------------------------------------------------------
  // Forward pass  (gather)
  // -------------------------------------------------------------------

  /// Look up embeddings for a batch of token IDs.
  ///
  /// @param token_ids  Flat array of token indices.  Every element must
  ///                   be in [0, vocab_size_).  Length must equal
  ///                   product(shape).
  /// @param shape      Logical shape of the token-ID batch,
  ///                   e.g. {batch, seq_len}.
  /// @return  Tensor of shape  shape + [d_model_].
  ///
  /// @throws std::invalid_argument if any token_id >= vocab_size_
  /// @throws std::invalid_argument if token_ids.size() != product(shape)
  // std::span<const size_t> 轻量只读视图， 不拷贝内存
  [[nodiscard]] auto forward(std::span<const std::size_t> token_ids,
                             std::vector<std::size_t> shape) const -> Tensor;

  // -------------------------------------------------------------------
  // Parameter access
  // -------------------------------------------------------------------

  /// Embedding table  shape [vocab_size_, d_model_].
  // 推理只读
  [[nodiscard]] auto weight() const -> const Tensor &;
  [[nodiscard]] auto weight() -> Tensor &;

  // Read-only metadata.
  [[nodiscard]] auto vocab_size() const noexcept -> std::size_t;
  [[nodiscard]] auto d_model() const noexcept -> std::size_t;

private:
  std::size_t vocab_size_; // 词表总token数量
  std::size_t d_model_;    // 单个词向量维度

  Tensor weight_; // [vocab_size_, d_model_]
};

} // namespace mt
