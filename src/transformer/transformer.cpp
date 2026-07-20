#include "transformer.h"

#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace mt {

namespace {

[[nodiscard]] std::size_t product(const std::vector<std::size_t>& shape) {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                           std::multiplies<>{});
}

} // namespace

// ===========================================================================
// Construction
// ===========================================================================

Transformer::Transformer(const std::size_t vocab_size,
                         const std::size_t d_model,
                         const std::size_t n_heads,
                         const std::size_t n_layers,
                         const std::size_t max_seq_len)
    : vocab_size_(vocab_size)
    , d_model_(d_model)
    , n_heads_(n_heads)
    , n_layers_(n_layers)
    , embed_(vocab_size, d_model)
    , final_norm_(d_model)
    , lm_head_(d_model, vocab_size, /*has_bias=*/false) {

    blocks_.reserve(n_layers_);
    for (std::size_t i = 0; i < n_layers_; ++i) {
        blocks_.emplace_back(d_model_, n_heads_, /*d_hidden=*/0, max_seq_len);
    }
}

// ===========================================================================
// Forward
// ===========================================================================

auto Transformer::forward(std::span<const std::size_t> token_ids,
                          std::vector<std::size_t> shape) const -> Tensor {

    const std::size_t num_tokens = token_ids.size();
    if (num_tokens != product(shape)) {
        throw std::invalid_argument(
            "Transformer::forward: token_ids size != product(shape)");
    }

    // 1. Embedding: [batch, seq] → [batch, seq, d_model] -----------------
    Tensor hidden = embed_.forward(token_ids, std::move(shape));
    // hidden shape: input_shape + [d_model_]  e.g. {batch, seq, d_model}

    // 2. Flatten to 2D: [batch*seq, d_model_] for DecoderBlock -----------
    const auto& h_shape = hidden.shape();
    const std::size_t D = h_shape.back();  // d_model_
    const std::size_t num_rows = hidden.size() / D;

    // Save batch dims for later restoration  e.g. {1, 4} from {1, 4, 8}
    std::vector<std::size_t> batch_shape(h_shape.begin(),
                                         h_shape.end() - 1);

    // Reinterpret as 2D (reshape in-place)
    hidden.reshape({num_rows, D});

    // 3. N × DecoderBlock ------------------------------------------------
    for (std::size_t l = 0; l < n_layers_; ++l) {
        hidden = blocks_[l].forward(hidden);
    }

    // 4. Final RMSNorm ---------------------------------------------------
    hidden = final_norm_.forward(hidden);

    // 5. LM Head: [num_rows, d_model_] → [num_rows, vocab_size_] --------
    hidden = lm_head_.forward(hidden);

    // 6. Restore batch dims: [batch, seq, vocab_size_] -------------------
    auto out_shape = batch_shape;
    out_shape.push_back(vocab_size_);
    hidden.reshape(out_shape);

    return hidden;
}

// ===========================================================================
// Greedy generation
// ===========================================================================

auto Transformer::generate(std::span<const std::size_t> prompt,
                           const std::size_t max_new_tokens) const
    -> std::vector<std::size_t> {

    std::vector<std::size_t> tokens(prompt.begin(), prompt.end());

    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        // Forward on current sequence → logits [1, seq, vocab]
        Tensor logits = forward(tokens, {1, tokens.size()});

        // Get logits of the last token: shape [1, seq, vocab]
        const auto& l_shape = logits.shape();
        const std::size_t last_pos = l_shape[1] - 1;

        // Find argmax over the vocab dimension
        const float* __restrict__ d = logits.data();
        const std::size_t offset = last_pos * vocab_size_;

        std::size_t next_id = 0;
        float max_val = d[offset];
        for (std::size_t v = 1; v < vocab_size_; ++v) {
            if (d[offset + v] > max_val) {
                max_val = d[offset + v];
                next_id = v;
            }
        }

        tokens.push_back(next_id);
    }

    return tokens;
}

// ===========================================================================
// Sub-module access
// ===========================================================================

auto Transformer::embed()      -> Embedding&                { return embed_; }
auto Transformer::blocks()     -> std::vector<DecoderBlock>& { return blocks_; }
auto Transformer::final_norm() -> RMSNorm&                   { return final_norm_; }
auto Transformer::lm_head()    -> Linear&                    { return lm_head_; }

auto Transformer::vocab_size() const noexcept -> std::size_t { return vocab_size_; }
auto Transformer::d_model()    const noexcept -> std::size_t { return d_model_; }
auto Transformer::n_heads()    const noexcept -> std::size_t { return n_heads_; }
auto Transformer::n_layers()   const noexcept -> std::size_t { return n_layers_; }

} // namespace mt
