#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "attention/multi_head_attention.h"
#include "tensor/tensor.h"

using mt::MultiHeadAttention;
using mt::Tensor;

// ===========================================================================
// Helper: set a Linear layer's weight to identity-like fill for testing
// ===========================================================================

namespace {

void fill_identity(Tensor& w) {
    // w is [K, N] where K == N (square matrix)
    const auto& sh = w.shape();
    const std::size_t dim = sh[0];
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            w.at({i, j}) = (i == j) ? 1.0f : 0.0f;
        }
    }
}

} // namespace

// ===========================================================================
// Basic correctness
// ===========================================================================

TEST(MultiHeadAttentionTest, SingleHeadEquivalent) {
    // d_model=2, n_heads=1 → should behave like our single-head Attention
    MultiHeadAttention mha(/*d_model=*/2, /*n_heads=*/1);

    // Set all projection weights to identity
    fill_identity(mha.w_q().weight());
    fill_identity(mha.w_k().weight());
    fill_identity(mha.w_v().weight());
    fill_identity(mha.w_o().weight());

    // Input: 2×2 identity matrix
    auto X = Tensor::from_data({2, 2}, {
        1.0f, 0.0f,
        0.0f, 1.0f
    });

    auto out = mha.forward(X);

    EXPECT_EQ(out.shape(), (std::vector<std::size_t>{2, 2}));

    // With identity weights and d_model=2, this is single-head attention
    // with d_k=2, scale=1/sqrt(2). Output matches our Attention test.
    EXPECT_NEAR(out.at({0, 0}), 0.6698f, 1e-3f);
    EXPECT_NEAR(out.at({0, 1}), 0.3302f, 1e-3f);
    EXPECT_NEAR(out.at({1, 0}), 0.3302f, 1e-3f);
    EXPECT_NEAR(out.at({1, 1}), 0.6698f, 1e-3f);
}

TEST(MultiHeadAttentionTest, TwoHeadsIdentityWeights) {
    // d_model=4, n_heads=2, d_head=2
    MultiHeadAttention mha(/*d_model=*/4, /*n_heads=*/2);

    fill_identity(mha.w_q().weight());
    fill_identity(mha.w_k().weight());
    fill_identity(mha.w_v().weight());
    fill_identity(mha.w_o().weight());

    // Each 2×2 block is identity; heads will see [[1,0],[0,1]] each
    auto X = Tensor::from_data({2, 4}, {
        1.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 1.0f
    });

    auto out = mha.forward(X);

    EXPECT_EQ(out.shape(), (std::vector<std::size_t>{2, 4}));

    // Head 0 (cols 0,1) and Head 1 (cols 2,3) both process [[1,0],[0,1]]
    // So output cols [0,1] = output cols [2,3] = softmax result
    EXPECT_NEAR(out.at({0, 0}), 0.6698f, 1e-3f);
    EXPECT_NEAR(out.at({0, 1}), 0.3302f, 1e-3f);
    EXPECT_NEAR(out.at({0, 2}), 0.6698f, 1e-3f);
    EXPECT_NEAR(out.at({0, 3}), 0.3302f, 1e-3f);

    EXPECT_NEAR(out.at({1, 0}), 0.3302f, 1e-3f);
    EXPECT_NEAR(out.at({1, 1}), 0.6698f, 1e-3f);
    EXPECT_NEAR(out.at({1, 2}), 0.3302f, 1e-3f);
    EXPECT_NEAR(out.at({1, 3}), 0.6698f, 1e-3f);
}

// ===========================================================================
// Shape correctness
// ===========================================================================

TEST(MultiHeadAttentionTest, OutputShapeMatchesInputShape) {
    MultiHeadAttention mha(/*d_model=*/8, /*n_heads=*/4);
    auto X = Tensor::zeros({5, 8});
    auto out = mha.forward(X);
    EXPECT_EQ(out.shape(), (std::vector<std::size_t>{5, 8}));
}

TEST(MultiHeadAttentionTest, LongerSequence) {
    MultiHeadAttention mha(/*d_model=*/16, /*n_heads=*/4);
    auto X = Tensor::zeros({32, 16});
    auto out = mha.forward(X);
    EXPECT_EQ(out.shape(), (std::vector<std::size_t>{32, 16}));
}

// ===========================================================================
// Validation
// ===========================================================================

TEST(MultiHeadAttentionTest, DModelNotDivisibleByNHeadsThrows) {
    EXPECT_THROW({
        MultiHeadAttention mha(5, 2);  // 5 % 2 != 0
    }, std::invalid_argument);
}

TEST(MultiHeadAttentionTest, WrongInputDimThrows) {
    MultiHeadAttention mha(4, 2);
    auto X = Tensor::ones({3, 5}); // last dim 5 != 4
    EXPECT_THROW({
        static_cast<void>(mha.forward(X));
    }, std::invalid_argument);
}

TEST(MultiHeadAttentionTest, Non2DInputThrows) {
    MultiHeadAttention mha(4, 2);
    auto X = Tensor::ones({2, 2, 4}); // 3-D
    EXPECT_THROW({
        static_cast<void>(mha.forward(X));
    }, std::invalid_argument);
}

// ===========================================================================
// Metadata
// ===========================================================================

TEST(MultiHeadAttentionTest, Metadata) {
    MultiHeadAttention mha(64, 8);
    EXPECT_EQ(mha.d_model(), 64);
    EXPECT_EQ(mha.n_heads(), 8);
    EXPECT_EQ(mha.d_head(), 8);
}

TEST(MultiHeadAttentionTest, EightHeads) {
    MultiHeadAttention mha(64, 8);
    EXPECT_EQ(mha.d_head(), 8);
}

// ===========================================================================
// Projection layer access
// ===========================================================================

TEST(MultiHeadAttentionTest, ProjectionLayersAccessible) {
    MultiHeadAttention mha(4, 2);
    EXPECT_EQ(mha.w_q().in_features(), 4);
    EXPECT_EQ(mha.w_q().out_features(), 4);
    EXPECT_EQ(mha.w_k().in_features(), 4);
    EXPECT_EQ(mha.w_v().in_features(), 4);
    EXPECT_EQ(mha.w_o().in_features(), 4);
}

TEST(MultiHeadAttentionTest, ConstAccessorsWork) {
    const MultiHeadAttention mha(4, 2);
    EXPECT_EQ(mha.d_model(), 4);
    EXPECT_EQ(mha.n_heads(), 2);
    EXPECT_EQ(mha.w_q().in_features(), 4);
}

// ===========================================================================
// Zero weights produce zero output
// ===========================================================================

TEST(MultiHeadAttentionTest, ZeroWeightsAllZeroOutput) {
    MultiHeadAttention mha(4, 2);
    auto X = Tensor::ones({3, 4});
    auto out = mha.forward(X);

    EXPECT_EQ(out.shape(), (std::vector<std::size_t>{3, 4}));
    for (std::size_t i = 0; i < out.size(); ++i) {
        EXPECT_FLOAT_EQ(out[i], 0.0f);
    }
}
