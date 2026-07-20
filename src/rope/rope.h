#pragma once

#include <cstddef>

#include "tensor/tensor.h"

namespace mt {

// ---------------------------------------------------------------------------
// RoPE — Rotary Position Embedding  (Su et al., 2021)
// ---------------------------------------------------------------------------
// RoPE encodes position information by *rotating* pairs of dimensions in
// the query and key vectors.  The rotation angle for dimension pair
// (2i, 2i+1) at position `pos` is  pos / theta_base^(2i / d_head).
//
// Key property:  (R(m)·Q_m) · (R(n)·K_n)^T = Q_m · R(m-n) · K_n^T
// The attention score depends only on the *relative* distance (m-n),
// not on the absolute positions.
// ---------------------------------------------------------------------------

class RoPE {
public:
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// @param d_head       Dimension per head (must be even).
    /// @param max_seq_len  Maximum sequence length to support.
    /// @param theta_base   Frequency base (Llama: 10000.0).
    /// @throws std::invalid_argument if d_head is odd
    explicit RoPE(std::size_t d_head,
                  std::size_t max_seq_len = 2048,
                  float theta_base = 10000.0f);

    // -------------------------------------------------------------------
    // Forward  (in-place rotation of Q and K)
    // -------------------------------------------------------------------

    /// Apply rotary position encoding to one head's Q and K.
    /// @param Q  Query tensor  [seq, d_head_]   modified in-place
    /// @param K  Key tensor    [seq, d_head_]   modified in-place
    /// @param start_pos  Offset for autoregressive generation.
    /// @throws std::invalid_argument if shapes are not [*, d_head_]
    void forward(Tensor& Q, Tensor& K, std::size_t start_pos = 0) const;

    // -------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------

    [[nodiscard]] auto d_head() const noexcept -> std::size_t;
    [[nodiscard]] auto max_seq_len() const noexcept -> std::size_t;

private:
    std::size_t d_head_;
    std::size_t max_seq_len_;
    std::size_t n_pairs_;    // d_head_ / 2

    Tensor cos_;  // [max_seq_len_, n_pairs_]
    Tensor sin_;  // [max_seq_len_, n_pairs_]
};

} // namespace mt
