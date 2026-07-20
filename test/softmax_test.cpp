#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "nn/softmax.h"
#include "tensor/tensor.h"

using mt::Tensor;

namespace {

/// Helper: verify that every row (slice along the last dim) sums to ~1.0.
void expect_rows_sum_to_one(const Tensor& t, float eps = 1e-4f) {
    const auto& shape = t.shape();
    const std::size_t D = shape.back();
    const std::size_t num_rows = t.size() / D;

    for (std::size_t r = 0; r < num_rows; ++r) {
        float sum = 0.0f;
        for (std::size_t j = 0; j < D; ++j) {
            sum += t[r * D + j];
        }
        EXPECT_NEAR(sum, 1.0f, eps) << "row " << r << " sum != 1";
    }
}

} // namespace

// ===========================================================================
// Basic correctness
// ===========================================================================

TEST(SoftmaxTest, Basic1D) {
    auto X = Tensor::from_data({3}, {1.0f, 2.0f, 3.0f});
    auto Y = mt::softmax(X);

    EXPECT_EQ(Y.shape(), (std::vector<std::size_t>{3}));

    // softmax([1,2,3]) = exp([1,2,3]) / sum(exp([1,2,3]))
    // exp(1)=2.718, exp(2)=7.389, exp(3)=20.086, sum=30.193
    // ≈ [0.0900, 0.2447, 0.6652]
    EXPECT_NEAR(Y[0], 0.0900f, 1e-3f);
    EXPECT_NEAR(Y[1], 0.2447f, 1e-3f);
    EXPECT_NEAR(Y[2], 0.6652f, 1e-3f);
    expect_rows_sum_to_one(Y);
}

TEST(SoftmaxTest, AllSameValues) {
    // softmax([5,5,5]) → [1/3, 1/3, 1/3]
    auto X = Tensor::from_data({4}, {5.0f, 5.0f, 5.0f, 5.0f});
    auto Y = mt::softmax(X);

    EXPECT_EQ(Y.shape(), (std::vector<std::size_t>{4}));
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(Y[i], 0.25f, 1e-5f);
    }
    expect_rows_sum_to_one(Y);
}

TEST(SoftmaxTest, SingleElement) {
    auto X = Tensor::from_data({1}, {42.0f});
    auto Y = mt::softmax(X);

    EXPECT_EQ(Y.shape(), (std::vector<std::size_t>{1}));
    EXPECT_FLOAT_EQ(Y[0], 1.0f);
}

// ===========================================================================
// Multi-dimensional
// ===========================================================================

TEST(SoftmaxTest, TwoDBatch) {
    // X shape [2, 3]: two independent rows
    auto X = Tensor::from_data({2, 3}, {
        1.0f, 2.0f, 3.0f,
        1.0f, 1.0f, 1.0f
    });
    auto Y = mt::softmax(X);

    EXPECT_EQ(Y.shape(), (std::vector<std::size_t>{2, 3}));

    // Row 0: same as Basic1D
    EXPECT_NEAR(Y.at({0, 0}), 0.0900f, 1e-3f);
    EXPECT_NEAR(Y.at({0, 1}), 0.2447f, 1e-3f);
    EXPECT_NEAR(Y.at({0, 2}), 0.6652f, 1e-3f);

    // Row 1: all equal → [1/3, 1/3, 1/3]
    EXPECT_NEAR(Y.at({1, 0}), 0.3333f, 1e-3f);
    EXPECT_NEAR(Y.at({1, 1}), 0.3333f, 1e-3f);
    EXPECT_NEAR(Y.at({1, 2}), 0.3333f, 1e-3f);

    expect_rows_sum_to_one(Y);
}

TEST(SoftmaxTest, ThreeD) {
    // shape [2, 2, 3]
    auto X = Tensor::from_data({2, 2, 3}, {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        0.0f, 0.0f, 0.0f,
        100.0f, 0.0f, 0.0f
    });
    auto Y = mt::softmax(X);

    EXPECT_EQ(Y.shape(), (std::vector<std::size_t>{2, 2, 3}));
    expect_rows_sum_to_one(Y);

    // Row with [100, 0, 0]: after safe softmax, first element ≈ 1.0
    EXPECT_NEAR(Y.at({1, 1, 0}), 1.0f, 1e-4f);
    EXPECT_NEAR(Y.at({1, 1, 1}), 0.0f, 1e-4f);
    EXPECT_NEAR(Y.at({1, 1, 2}), 0.0f, 1e-4f);
}

// ===========================================================================
// Numerical stability
// ===========================================================================

TEST(SoftmaxTest, LargeValuesDoNotOverflow) {
    // 1000 is way beyond float32 exp limit (~88.7)
    auto X = Tensor::from_data({3}, {1000.0f, 1000.0f, 1000.0f});
    auto Y = mt::softmax(X);

    EXPECT_EQ(Y.shape(), (std::vector<std::size_t>{3}));
    // All equal → should be [1/3, 1/3, 1/3], not NaN or Inf
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(std::isfinite(Y[i]));
        EXPECT_NEAR(Y[i], 1.0f / 3.0f, 1e-4f);
    }
    expect_rows_sum_to_one(Y);
}

TEST(SoftmaxTest, LargeValueSpread) {
    // One dominant value should still work
    auto X = Tensor::from_data({2}, {500.0f, 0.0f});
    auto Y = mt::softmax(X);

    // After subtracting 500, only exp(0)=1 matters
    EXPECT_NEAR(Y[0], 1.0f, 1e-5f);
    EXPECT_NEAR(Y[1], 0.0f, 1e-5f);

    // Both must be finite
    EXPECT_TRUE(std::isfinite(Y[0]));
    EXPECT_TRUE(std::isfinite(Y[1]));
}

TEST(SoftmaxTest, AllNegativeValues) {
    auto X = Tensor::from_data({2}, {-100.0f, -200.0f});
    auto Y = mt::softmax(X);

    // exp(-100) / (exp(-100) + exp(-200)) ≈ exp(-100) / exp(-100) = 1.0
    EXPECT_NEAR(Y[0], 1.0f, 1e-5f);
    EXPECT_NEAR(Y[1], 0.0f, 1e-5f);
    expect_rows_sum_to_one(Y);
}

// ===========================================================================
// Error handling
// ===========================================================================

TEST(SoftmaxTest, ScalarTensorThrows) {
    auto X = Tensor::from_data({}, {}); // ndim = 0
    EXPECT_THROW({ static_cast<void>(mt::softmax(X)); }, std::invalid_argument);
}

// ===========================================================================
// Input not modified
// ===========================================================================

TEST(SoftmaxTest, InputNotModified) {
    auto X = Tensor::from_data({3}, {1.0f, 2.0f, 3.0f});
    auto Y = mt::softmax(X);

    // Input values unchanged
    EXPECT_FLOAT_EQ(X[0], 1.0f);
    EXPECT_FLOAT_EQ(X[1], 2.0f);
    EXPECT_FLOAT_EQ(X[2], 3.0f);
}

// ===========================================================================
// Edge: zero row
// ===========================================================================

TEST(SoftmaxTest, AllZeros) {
    auto X = Tensor::zeros({4});
    auto Y = mt::softmax(X);

    // softmax of all zeros → uniform
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(Y[i], 0.25f, 1e-5f);
    }
    expect_rows_sum_to_one(Y);
}
