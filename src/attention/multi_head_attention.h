#pragma once

#include <cstddef>

#include "attention.h"
#include "nn/linear.h"
#include "tensor/tensor.h"

namespace mt {

class RoPE;  // forward declaration — defined in rope/rope.h

// ---------------------------------------------------------------------------
// MultiHeadAttention — n_heads parallel scaled dot-product attentions
// ---------------------------------------------------------------------------
// MultiHeadAttention wraps the Q/K/V/O linear projections and runs `n_heads`
// independent single-head Attention computations in parallel subspaces.
//
//   d_head = d_model / n_heads        (must divide evenly)
//
// An optional RoPE module can be attached via set_rope().  When set, Q and K
// slices are rotated *after* head extraction and *before* the single-head
// attention call.
//
// After all heads finish, one final linear projection (W_o) produces the
// layer output.
// ---------------------------------------------------------------------------

class MultiHeadAttention {
public:
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// @param d_model   Must be divisible by n_heads.
    /// @param n_heads   Number of attention heads.
    /// @throws std::invalid_argument if d_model % n_heads != 0
    explicit MultiHeadAttention(std::size_t d_model, std::size_t n_heads);

    // -------------------------------------------------------------------
    // Forward pass
    // -------------------------------------------------------------------

    /// Self-attention forward.
    /// @param X  Input  shape [seq, d_model]
    /// @param start_pos  RoPE position offset (0 = prefill, k = decode step k).
    /// @return   Output shape [seq, d_model]
    [[nodiscard]] auto forward(const Tensor& X,
                               std::size_t start_pos = 0) const -> Tensor;

    // -------------------------------------------------------------------
    // RoPE (optional — call set_rope to enable)
    // -------------------------------------------------------------------

    void set_rope(const RoPE* rope) { rope_ = rope; }

    // -------------------------------------------------------------------
    // Projection layers (mutable — for weight loading)
    // -------------------------------------------------------------------

    [[nodiscard]] auto w_q() -> Linear&;
    [[nodiscard]] auto w_k() -> Linear&;
    [[nodiscard]] auto w_v() -> Linear&;
    [[nodiscard]] auto w_o() -> Linear&;

    [[nodiscard]] auto w_q() const -> const Linear&;
    [[nodiscard]] auto w_k() const -> const Linear&;
    [[nodiscard]] auto w_v() const -> const Linear&;
    [[nodiscard]] auto w_o() const -> const Linear&;

    [[nodiscard]] auto d_model() const noexcept -> std::size_t;
    [[nodiscard]] auto n_heads() const noexcept -> std::size_t;
    [[nodiscard]] auto d_head() const noexcept -> std::size_t;

private:
    std::size_t d_model_;
    std::size_t n_heads_;
    std::size_t d_head_;

    Linear    w_q_;
    Linear    w_k_;
    Linear    w_v_;
    Linear    w_o_;
    Attention attention_;
    const RoPE* rope_ = nullptr;   // optional
};

} // namespace mt
