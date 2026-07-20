#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "nn/mlp.h"
#include "tensor/tensor.h"

using mt::MLP;
using mt::Tensor;

// ===========================================================================
// SiLU
// ===========================================================================

TEST(SiLUTest, Scalar) {
    auto x = Tensor::from_data({1}, {0.0f});
    auto y = mt::silu(x);
    EXPECT_FLOAT_EQ(y[0], 0.0f);
}

TEST(SiLUTest, Positive) {
    // siLU(2) = 2 * sigmoid(2) = 2 / (1+e^-2) ≈ 2*0.8808 = 1.7616
    auto x = Tensor::from_data({1}, {2.0f});
    auto y = mt::silu(x);
    EXPECT_NEAR(y[0], 1.7616f, 1e-3f);
}

TEST(SiLUTest, Negative) {
    // siLU(-2) = -2 / (1+e^2) ≈ -2 / 8.389 = -0.2384
    auto x = Tensor::from_data({1}, {-2.0f});
    auto y = mt::silu(x);
    EXPECT_NEAR(y[0], -0.2384f, 1e-3f);
}

TEST(SiLUTest, Batch1D) {
    auto x = Tensor::from_data({3}, {-2.0f, 0.0f, 2.0f});
    auto y = mt::silu(x);
    EXPECT_NEAR(y[0], -0.2384f, 1e-3f);
    EXPECT_FLOAT_EQ(y[1], 0.0f);
    EXPECT_NEAR(y[2], 1.7616f, 1e-3f);
}

TEST(SiLUTest, LargeNegativeSafe) {
    auto x = Tensor::from_data({1}, {-100.0f}); // exp(100) overflows
    auto y = mt::silu(x);
    EXPECT_NEAR(y[0], 0.0f, 1e-5f); // SiLU(-100) ≈ 0
    EXPECT_TRUE(std::isfinite(y[0]));
}

// ===========================================================================
// MLP
// ===========================================================================

TEST(MLPTest, OutputShape) {
    MLP mlp(/*d_model=*/6, /*d_hidden=*/8);
    auto X = Tensor::ones({3, 6});
    auto out = mlp.forward(X);
    EXPECT_EQ(out.shape(), (std::vector<std::size_t>{3, 6}));
}

TEST(MLPTest, DefaultHiddenDim) {
    MLP mlp(/*d_model=*/12); // auto d_hidden = 12 * 8 / 3 = 32
    EXPECT_EQ(mlp.d_model(), 12);
    EXPECT_EQ(mlp.d_hidden(), 32);
}

TEST(MLPTest, ZeroWeightsZeroOutput) {
    MLP mlp(4, 8);
    auto X = Tensor::ones({2, 4});
    auto out = mlp.forward(X);
    EXPECT_EQ(out.shape(), (std::vector<std::size_t>{2, 4}));
    for (std::size_t i = 0; i < out.size(); ++i) {
        EXPECT_FLOAT_EQ(out[i], 0.0f);
    }
}

// ===========================================================================
// Validation
// ===========================================================================

TEST(MLPTest, WrongDimThrows) {
    MLP mlp(4, 8);
    auto X = Tensor::ones({2, 5}); // last dim 5 != 4
    EXPECT_THROW({ static_cast<void>(mlp.forward(X)); }, std::invalid_argument);
}

TEST(MLPTest, Non2DThrows) {
    MLP mlp(4, 8);
    auto X = Tensor::ones({1, 2, 4});
    EXPECT_THROW({ static_cast<void>(mlp.forward(X)); }, std::invalid_argument);
}

// ===========================================================================
// Weight access
// ===========================================================================

TEST(MLPTest, WeightShapes) {
    MLP mlp(6, 12);
    EXPECT_EQ(mlp.w_gate().weight().shape(), (std::vector<std::size_t>{6, 12}));
    EXPECT_EQ(mlp.w_up().weight().shape(), (std::vector<std::size_t>{6, 12}));
    EXPECT_EQ(mlp.w_down().weight().shape(), (std::vector<std::size_t>{12, 6}));
}

TEST(MLPTest, MutableWeights) {
    MLP mlp(2, 4);
    mlp.w_gate().weight().at({0, 0}) = 1.0f;
    EXPECT_FLOAT_EQ(mlp.w_gate().weight().at({0, 0}), 1.0f);
}

// ===========================================================================
// Basic forward with known weights
// ===========================================================================

TEST(MLPTest, SimpleForwardWithIdentityDown) {
    MLP mlp(2, 4);
    // Set w_down to identity-like: map [4] → first 2 dims go to output
    auto& wd = mlp.w_down().weight();
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            wd.at({i, j}) = (i == j) ? 1.0f : 0.0f;

    // Gate and up weights remain zero → gate=0, up=0 → hidden=0 → output=0
    auto X = Tensor::ones({1, 2});
    auto out = mlp.forward(X);
    EXPECT_EQ(out.shape(), (std::vector<std::size_t>{1, 2}));
    EXPECT_FLOAT_EQ(out.at({0, 0}), 0.0f);
}
