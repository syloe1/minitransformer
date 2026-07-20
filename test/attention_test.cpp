#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "attention/attention.h"
#include "tensor/tensor.h"

using mt::Attention;
using mt::Tensor;

// ===========================================================================
// Basic correctness
// ===========================================================================

TEST(AttentionTest, Basic2x2IdentityQK) {
    // Q = K = I₂ (identity), so Q@K^T = I₂
    // d_k=2 → scale = 1/sqrt(2) ≈ 0.7071
    Attention attn(/*d_k=*/2);

    auto Q = Tensor::from_data({2, 2}, {
        1.0f, 0.0f,
        0.0f, 1.0f
    });
    auto K = Tensor::from_data({2, 2}, {
        1.0f, 0.0f,
        0.0f, 1.0f
    });
    auto V = Tensor::from_data({2, 2}, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    auto out = attn.forward(Q, K, V);

    EXPECT_EQ(out.shape(), (std::vector<std::size_t>{2, 2}));

    // Row 0 weights: softmax([0.7071, 0]) ≈ [0.6698, 0.3302]
    // output[0] = [0.6698*1+0.3302*3, 0.6698*2+0.3302*4]
    EXPECT_NEAR(out.at({0, 0}), 1.6604f, 1e-3f);
    EXPECT_NEAR(out.at({0, 1}), 2.6604f, 1e-3f);

    // Row 1 weights: softmax([0, 0.7071]) ≈ [0.3302, 0.6698]
    // output[1] = [0.3302*1+0.6698*3, 0.3302*2+0.6698*4]
    EXPECT_NEAR(out.at({1, 0}), 2.3396f, 1e-3f);
    EXPECT_NEAR(out.at({1, 1}), 3.3396f, 1e-3f);
}

TEST(AttentionTest, OutputShapeCorrect) {
    // Q: [3, 4]   K: [3, 4]   V: [3, 5]   → output [3, 5]
    Attention attn(/*d_k=*/4);

    auto Q = Tensor::ones({3, 4});
    auto K = Tensor::ones({3, 4});
    auto V = Tensor::ones({3, 5});

    auto out = attn.forward(Q, K, V);
    EXPECT_EQ(out.shape(), (std::vector<std::size_t>{3, 5}));
}

TEST(AttentionTest, DifferentSeqLengths) {
    // Q: [2, 3]  K: [4, 3]  V: [4, 2]  → output [2, 2]
    // (e.g., cross-attention: 2 queries attending over 4 key-value pairs)
    Attention attn(/*d_k=*/3);

    auto Q = Tensor::ones({2, 3});
    auto K = Tensor::ones({4, 3});
    auto V = Tensor::ones({4, 2});

    auto out = attn.forward(Q, K, V);
    EXPECT_EQ(out.shape(), (std::vector<std::size_t>{2, 2}));
}

// ===========================================================================
// Scale factor
// ===========================================================================

TEST(AttentionTest, ScaleFactor) {
    Attention attn(/*d_k=*/16);
    EXPECT_FLOAT_EQ(attn.scale(), 1.0f / std::sqrt(16.0f)); // 0.25

    Attention attn2(/*d_k=*/64);
    EXPECT_FLOAT_EQ(attn2.scale(), 1.0f / std::sqrt(64.0f)); // 0.125
}

// ===========================================================================
// Shape validation
// ===========================================================================

TEST(AttentionTest, QKMismatchedLastDimThrows) {
    Attention attn(4);
    auto Q = Tensor::ones({3, 4});
    auto K = Tensor::ones({3, 5}); // last dim 5 ≠ 4
    auto V = Tensor::ones({3, 6});

    EXPECT_THROW({ static_cast<void>(attn.forward(Q, K, V)); }, std::invalid_argument);
}

TEST(AttentionTest, KVSeqLenMismatchThrows) {
    Attention attn(4);
    auto Q = Tensor::ones({3, 4});
    auto K = Tensor::ones({3, 4});
    auto V = Tensor::ones({5, 6}); // seq_len 5 ≠ 3

    EXPECT_THROW({ static_cast<void>(attn.forward(Q, K, V)); }, std::invalid_argument);
}

TEST(AttentionTest, Non2DInputThrows) {
    Attention attn(4);
    auto Q = Tensor::ones({2, 3, 4}); // 3-D
    auto K = Tensor::ones({3, 4});
    auto V = Tensor::ones({3, 5});

    EXPECT_THROW({ static_cast<void>(attn.forward(Q, K, V)); },
                 std::invalid_argument);
}

// ===========================================================================
// d_k ≠ d_v
// ===========================================================================

TEST(AttentionTest, DifferentDimKV) {
    // d_k = 3, d_v = 5
    Attention attn(/*d_k=*/3);

    auto Q = Tensor::ones({2, 3});
    auto K = Tensor::ones({2, 3});
    auto V = Tensor::ones({2, 5});

    auto out = attn.forward(Q, K, V);
    EXPECT_EQ(out.shape(), (std::vector<std::size_t>{2, 5}));
}

// ===========================================================================
// const-correctness / input not modified
// ===========================================================================

TEST(AttentionTest, InputNotModified) {
    Attention attn(2);
    auto Q = Tensor::from_data({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    auto K = Tensor::from_data({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    auto V = Tensor::from_data({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});

    auto out = attn.forward(Q, K, V);

    // Inputs unchanged
    EXPECT_FLOAT_EQ(Q.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(K.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(V.at({0, 0}), 5.0f);
}

// ===========================================================================
// Single-token edge case
// ===========================================================================

TEST(AttentionTest, SingleToken) {
    Attention attn(/*d_k=*/8);
    auto Q = Tensor::ones({1, 8});
    auto K = Tensor::ones({1, 8});
    auto V = Tensor::from_data({1, 3}, {2.0f, 4.0f, 6.0f});

    auto out = attn.forward(Q, K, V);

    EXPECT_EQ(out.shape(), (std::vector<std::size_t>{1, 3}));
    // Single token → weight is [1.0], output = V unchanged
    EXPECT_FLOAT_EQ(out.at({0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(out.at({0, 1}), 4.0f);
    EXPECT_FLOAT_EQ(out.at({0, 2}), 6.0f);
}
