#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "decoder_block.h"
#include "nn/embedding.h"
#include "nn/linear.h"
#include "nn/rms_norm.h"
#include "tensor/tensor.h"

namespace mt {

// ---------------------------------------------------------------------------
// Transformer — complete Decoder-only model  (Llama-style)
// ---------------------------------------------------------------------------
//
// Architecture:
//   Embedding → N × DecoderBlock → Final RMSNorm → LM Head
//
//   Token IDs: [batch, seq]          (integers)
//   Output:     [batch, seq, vocab]   (logits)
//
//   Greedy generation:
//     while len < max_new_tokens:
//       logits = forward(ids, &kv_cache)
//       next = argmax(logits[-1])
//       ids.append(next)
// ---------------------------------------------------------------------------

class Transformer {
public:
  /// @param vocab_size   Vocabulary size. 词表总大小
  /// @param d_model      Model dimension. 向量维度
  /// @param n_heads      Attention heads. //注意力头数量
  /// @param n_layers     Number of DecoderBlocks. 层数
  /// @param max_seq_len  Max sequence length (for RoPE + KV Cache).
  Transformer(std::size_t vocab_size, std::size_t d_model, std::size_t n_heads,
              std::size_t n_layers, std::size_t max_seq_len = 2048);

  // -------------------------------------------------------------------
  // Forward pass
  // -------------------------------------------------------------------

  /// Convert token IDs to logits.
  ///
  /// @param token_ids  Flat token indices  length = product(shape).
  /// @param shape      Shape of token_ids, e.g. {batch, seq}.
  /// @return           Logits tensor  shape [batch, seq, vocab_size_].
  [[nodiscard]] auto forward(std::span<const std::size_t> token_ids,
                             std::vector<std::size_t> shape) const -> Tensor;

  // -------------------------------------------------------------------
  // Greedy generation
  // -------------------------------------------------------------------

  /// Generate tokens autoregressively.
  ///
  /// @param prompt       Initial token IDs (the prompt).
  /// @param max_new_tokens  Maximum tokens to generate.
  /// @return  Full sequence: prompt + generated tokens.
  // 自回归贪心生成
  [[nodiscard]] auto generate(std::span<const std::size_t> prompt,
                              std::size_t max_new_tokens) const
      -> std::vector<std::size_t>;

  // -------------------------------------------------------------------
  // Sub-module access (for weight loading)
  // -------------------------------------------------------------------

  [[nodiscard]] auto embed() -> Embedding &;
  [[nodiscard]] auto blocks() -> std::vector<DecoderBlock> &;
  [[nodiscard]] auto final_norm() -> RMSNorm &;
  [[nodiscard]] auto lm_head() -> Linear &;

  // Metadata
  [[nodiscard]] auto vocab_size() const noexcept -> std::size_t;
  [[nodiscard]] auto d_model() const noexcept -> std::size_t;
  [[nodiscard]] auto n_heads() const noexcept -> std::size_t;
  [[nodiscard]] auto n_layers() const noexcept -> std::size_t;

private:
  std::size_t vocab_size_;
  std::size_t d_model_;
  std::size_t n_heads_;
  std::size_t n_layers_;

  Embedding embed_;                  // [vocab, d_model]  词嵌入层
  std::vector<DecoderBlock> blocks_; // N layers  多层解码器数组
  RMSNorm final_norm_;               // [d_model]  输出归一化
  Linear lm_head_;                   // [d_model, vocab] LM输出头
};

} // namespace mt
