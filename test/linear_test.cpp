#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "nn/linear.h"
#include "tensor/tensor.h"

using mt::Linear;
using mt::Tensor;

// ===========================================================================
// Basic forward pass
// ===========================================================================

TEST(LinearTest, ForwardWithBias) {
    // W [3, 2] = [[1, 4], [2, 5], [3, 6]]
    // b [2]    = [0.1, 0.2]
    Linear linear(/*in=*/3, /*out=*/2, /*has_bias=*/true);
    linear.weight() = Tensor::from_data({3, 2}, {
        1.0f, 4.0f,
        2.0f, 5.0f,
        3.0f, 6.0f
    });
    linear.bias() = Tensor::from_data({2}, {0.1f, 0.2f});

    // X [2, 3] = [[1, 2, 3], [4, 5, 6]]
    auto X = Tensor::from_data({2, 3}, {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    });

    auto Y = linear.forward(X);

    // Y = X @ W + b
    // Y[0] = [1*1+2*2+3*3+0.1, 1*4+2*5+3*6+0.2] = [14.1, 32.2]
    // Y[1] = [4*1+5*2+6*3+0.1, 4*4+5*5+6*6+0.2] = [32.1, 77.2]
    EXPECT_EQ(Y.shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ(Y.at({0, 0}), 14.1f);
    EXPECT_FLOAT_EQ(Y.at({0, 1}), 32.2f);
    EXPECT_FLOAT_EQ(Y.at({1, 0}), 32.1f);
    EXPECT_FLOAT_EQ(Y.at({1, 1}), 77.2f);
}

TEST(LinearTest, ForwardWithoutBias) {
    Linear linear(/*in=*/2, /*out=*/3, /*has_bias=*/false);
    linear.weight() = Tensor::from_data({2, 3}, {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    });

    auto X = Tensor::from_data({2, 2}, {
        1.0f, 0.0f,
        0.0f, 1.0f
    });

    auto Y = linear.forward(X);

    // Y = X @ W  (no bias)
    // Row 0 of X is [1,0], so Y[0] = W row 0 = [1,2,3]
    // Row 1 of X is [0,1], so Y[1] = W row 1 = [4,5,6]
    EXPECT_EQ(Y.shape(), (std::vector<std::size_t>{2, 3}));
    EXPECT_FLOAT_EQ(Y.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 2}), 3.0f);
    EXPECT_FLOAT_EQ(Y.at({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(Y.at({1, 1}), 5.0f);
    EXPECT_FLOAT_EQ(Y.at({1, 2}), 6.0f);
}

TEST(LinearTest, SingleRowInput) {
    Linear linear(/*in=*/4, /*out=*/2, /*has_bias=*/true);
    linear.weight() = Tensor::ones({4, 2});
    linear.bias()   = Tensor::ones({2});

    auto X = Tensor::ones({1, 4});

    auto Y = linear.forward(X);

    EXPECT_EQ(Y.shape(), (std::vector<std::size_t>{1, 2}));
    // Each output = sum(1*1 over 4) + 1 = 5.0
    EXPECT_FLOAT_EQ(Y.at({0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(Y.at({0, 1}), 5.0f);
}

// ===========================================================================
// Shape validation
// ===========================================================================

TEST(LinearTest, WrongInputDimThrows) {
    Linear linear(4, 3);
    auto X = Tensor::ones({2, 5}); // last dim 5 != 4
    EXPECT_THROW({ static_cast<void>(linear.forward(X)); }, std::invalid_argument);
}

TEST(LinearTest, Non2DInputThrows) {
    Linear linear(4, 3);
    auto X = Tensor::ones({2, 3, 4}); // 3-D
    EXPECT_THROW({ static_cast<void>(linear.forward(X)); }, std::invalid_argument);
}

TEST(LinearTest, CorrectShapeDoesNotThrow) {
    Linear linear(4, 3);
    auto X = Tensor::zeros({2, 4});
    EXPECT_NO_THROW({ static_cast<void>(linear.forward(X)); });
}

// ===========================================================================
// Metadata
// ===========================================================================

TEST(LinearTest, Metadata) {
    Linear linear(64, 128, /*has_bias=*/true);
    EXPECT_EQ(linear.in_features(), 64);
    EXPECT_EQ(linear.out_features(), 128);
    EXPECT_TRUE(linear.has_bias());

    Linear no_bias(32, 64, false);
    EXPECT_FALSE(no_bias.has_bias());
}

// ===========================================================================
// Weight / bias access
// ===========================================================================

TEST(LinearTest, WeightShape) {
    Linear linear(64, 128);
    EXPECT_EQ(linear.weight().shape(), (std::vector<std::size_t>{64, 128}));
}

TEST(LinearTest, BiasShape) {
    Linear linear(64, 128);
    EXPECT_EQ(linear.bias().shape(), (std::vector<std::size_t>{128}));
}

TEST(LinearTest, NoBiasBiasIsEmpty) {
    Linear linear(64, 128, false);
    EXPECT_TRUE(linear.bias().empty());
}

TEST(LinearTest, MutableWeightAccess) {
    Linear linear(2, 3);
    // Modify weight through mutable accessor
    linear.weight().at({0, 0}) = 42.0f;
    EXPECT_FLOAT_EQ(linear.weight().at({0, 0}), 42.0f);
}

TEST(LinearTest, MutableBiasAccess) {
    Linear linear(2, 3);
    linear.bias().at({0}) = 7.0f;
    EXPECT_FLOAT_EQ(linear.bias().at({0}), 7.0f);
}

// ===========================================================================
// const-correctness
// ===========================================================================

TEST(LinearTest, ConstForward) {
    const Linear linear(3, 2);
    // Zero-initialized weights → output should be all zeros
    auto X = Tensor::ones({2, 3});
    auto Y = linear.forward(X);

    EXPECT_EQ(Y.shape(), (std::vector<std::size_t>{2, 2}));
    for (std::size_t i = 0; i < Y.size(); ++i) {
        EXPECT_FLOAT_EQ(Y[i], 0.0f);
    }
}

TEST(LinearTest, InputNotModified) {
    Linear linear(3, 2);
    linear.weight() = Tensor::ones({3, 2});
    linear.bias()   = Tensor::zeros({2});

    auto X = Tensor::ones({2, 3});
    const auto* x_ptr_before = X.data();
    auto Y = linear.forward(X);

    // X pointer and content must be unchanged
    EXPECT_EQ(X.data(), x_ptr_before);
    EXPECT_FLOAT_EQ(X.at({0, 0}), 1.0f);
}

// ===========================================================================
// Zero-initialised parameters (default construction)
// ===========================================================================

TEST(LinearTest, DefaultWeightsAreZero) {
    Linear linear(4, 3);
    for (std::size_t i = 0; i < linear.weight().size(); ++i) {
        EXPECT_FLOAT_EQ(linear.weight()[i], 0.0f);
    }
    if (linear.has_bias()) {
        for (std::size_t i = 0; i < linear.bias().size(); ++i) {
            EXPECT_FLOAT_EQ(linear.bias()[i], 0.0f);
        }
    }
}

// ===========================================================================
// Larger matrices
// ===========================================================================

TEST(LinearTest, LargeForward) {
    constexpr std::size_t M = 128;
    constexpr std::size_t K = 256;
    constexpr std::size_t N = 128;

    Linear linear(K, N);
    linear.weight() = Tensor::ones({K, N});
    linear.bias()   = Tensor::ones({N});

    auto X = Tensor::ones({M, K});
    auto Y = linear.forward(X);

    EXPECT_EQ(Y.shape(), (std::vector<std::size_t>{M, N}));
    // Each output element = sum(K ones) + 1 = K + 1
    EXPECT_NEAR(Y.at({0, 0}), static_cast<float>(K + 1), 1e-4f);
    EXPECT_NEAR(Y.at({M - 1, N - 1}), static_cast<float>(K + 1), 1e-4f);
}
