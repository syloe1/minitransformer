#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "tensor/tensor.h"
#include "transformer/decoder_block.h"

using mt::DecoderBlock;
using mt::Tensor;

// ===========================================================================
// Basic
// ===========================================================================

TEST(DecoderBlockTest, OutputShape) {
    DecoderBlock block(/*d_model=*/8, /*n_heads=*/2,
                       /*d_hidden=*/16, /*max_seq_len=*/64);
    auto X = Tensor::ones({3, 8});
    auto Y = block.forward(X);
    EXPECT_EQ(Y.shape(), (std::vector<std::size_t>{3, 8}));
}

TEST(DecoderBlockTest, ZeroWeightsZeroOutput) {
    // All weights zero-initialized → all Linear outputs are zero
    // RMSNorm of zero → zero, residual adds input → output equals input.
    // Actually: output = input + attn_out, and attn_out=0 (all zero weights).
    // Then residual again: output = hidden + ffn_out, ffn_out=0.
    // So output = input.
    DecoderBlock block(/*d_model=*/4, /*n_heads=*/2,
                       /*d_hidden=*/8, /*max_seq_len=*/8);
    auto X = Tensor::from_data({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto Y = block.forward(X);
    EXPECT_EQ(Y.shape(), (std::vector<std::size_t>{1, 4}));
    EXPECT_FLOAT_EQ(Y.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 2}), 3.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 3}), 4.0f);
}

// ===========================================================================
// Sub-module access
// ===========================================================================

TEST(DecoderBlockTest, SubModuleAccess) {
    DecoderBlock block(4, 2, 8, 16);
    EXPECT_EQ(block.d_model(), 4);
    // Verify we can access sub-modules without crash
    EXPECT_NO_THROW({ (void)block.mha(); (void)block.mlp(); (void)block.norm1(); (void)block.norm2(); (void)block.rope(); });
}

// ===========================================================================
// Validation
// ===========================================================================

TEST(DecoderBlockTest, WrongDimThrows) {
    DecoderBlock block(4, 2, 8, 16);
    auto X = Tensor::ones({2, 6});
    EXPECT_THROW({ static_cast<void>(block.forward(X)); }, std::invalid_argument);
}

TEST(DecoderBlockTest, Non2DThrows) {
    DecoderBlock block(4, 2, 8, 16);
    auto X = Tensor::ones({1, 2, 4});
    EXPECT_THROW({ static_cast<void>(block.forward(X)); }, std::invalid_argument);
}

// ===========================================================================
// start_pos
// ===========================================================================

TEST(DecoderBlockTest, StartPosWorks) {
    DecoderBlock block(4, 2, 8, 16);
    auto X = Tensor::ones({1, 4});
    auto Y = block.forward(X, /*start_pos=*/5);
    EXPECT_EQ(Y.shape(), (std::vector<std::size_t>{1, 4}));
}
