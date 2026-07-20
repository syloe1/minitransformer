#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "nn/rms_norm.h"
#include "tensor/tensor.h"

using mt::RMSNorm;
using mt::Tensor;

// ===========================================================================
// Basic
// ===========================================================================

TEST(RMSNormTest, Basic1D_DefaultGamma) {
    RMSNorm norm(3);
    auto X = Tensor::from_data({3}, {0.0f, 3.0f, 4.0f});
    // mean(x²) = (0+9+16)/3 = 8.333, rms = sqrt(8.333+1e-6) ≈ 2.887
    // output = [0, 3/2.887, 4/2.887] ≈ [0, 1.039, 1.386]
    auto Y = norm.forward(X);
    EXPECT_EQ(Y.shape(), (std::vector<std::size_t>{3}));
    EXPECT_NEAR(Y[1], 1.0392f, 1e-3f);
    EXPECT_NEAR(Y[2], 1.3856f, 1e-3f);
}

TEST(RMSNormTest, CustomGamma) {
    RMSNorm norm(2);
    norm.gamma() = Tensor::from_data({2}, {2.0f, 3.0f});
    auto X = Tensor::from_data({1, 2}, {1.0f, 0.0f});
    // mean(x²) = (1+0)/2 = 0.5, rms ≈ 0.7071
    // y[0]=(1/0.7071)*2≈2.828, y[1]=0*3=0
    auto Y = norm.forward(X);
    EXPECT_NEAR(Y.at({0, 0}), 2.8284f, 1e-3f);
    EXPECT_FLOAT_EQ(Y.at({0, 1}), 0.0f);
}

// ===========================================================================
// Multi-dimensional
// ===========================================================================

TEST(RMSNormTest, TwoD) {
    RMSNorm norm(3);
    norm.gamma() = Tensor::ones({3});
    auto X = Tensor::from_data({2, 3}, {
        1.0f, 2.0f, 2.0f,
        0.0f, 0.0f, 0.0f
    });
    auto Y = norm.forward(X);
    EXPECT_EQ(Y.shape(), (std::vector<std::size_t>{2, 3}));
    // Row 0: mean_sq=(1+4+4)/3=3, rms=sqrt(3)=1.732, row=[0.577,1.155,1.155]
    EXPECT_NEAR(Y.at({0, 0}), 0.5774f, 1e-3f);
    // Row 1: all zeros → rms ≈ sqrt(eps) → output ≈ 0/eps ≈ 0
    EXPECT_NEAR(Y.at({1, 0}), 0.0f, 1e-4f);
}

// ===========================================================================
// Validation
// ===========================================================================

TEST(RMSNormTest, DimMismatchThrows) {
    RMSNorm norm(3);
    auto X = Tensor::ones({2, 5});
    EXPECT_THROW({ static_cast<void>(norm.forward(X)); }, std::invalid_argument);
}

// ===========================================================================
// Metadata
// ===========================================================================

TEST(RMSNormTest, DefaultEpsilon) {
    RMSNorm norm(4);
    EXPECT_EQ(norm.d(), 4);
    EXPECT_FLOAT_EQ(norm.eps(), 1e-6f);
}

TEST(RMSNormTest, GammaShape) {
    RMSNorm norm(5);
    EXPECT_EQ(norm.gamma().shape(), (std::vector<std::size_t>{5}));
    // Default gamma = all ones
    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_FLOAT_EQ(norm.gamma()[i], 1.0f);
}
