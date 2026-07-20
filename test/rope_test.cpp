#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "rope/rope.h"
#include "tensor/tensor.h"

using mt::RoPE;
using mt::Tensor;

// ===========================================================================
// Basic correctness
// ===========================================================================

TEST(RoPETest, BasicRotationDHead2) {
    // d_head=2, 1 pair. theta=1 so freq=1, angle=pos*1
    RoPE rope(/*d_head=*/2, /*max_seq_len=*/4, /*theta_base=*/1.0f);

    // Q = [[1, 0], [0, 1]]
    auto Q = Tensor::from_data({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    auto K = Tensor::from_data({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});

    rope.forward(Q, K, /*start_pos=*/0);

    // theta=1 → freq=1
    // pos=0: angle=0, cos=1, sin=0 → Q[0] = [1, 0]
    // pos=1: angle=1, cos≈0.5403, sin≈0.8415
    //   Q[1] = [0*0.5403-1*0.8415, 0*0.8415+1*0.5403] = [-0.8415, 0.5403]
    EXPECT_NEAR(Q.at({0, 0}), 1.0f, 1e-4f);
    EXPECT_NEAR(Q.at({0, 1}), 0.0f, 1e-4f);
    EXPECT_NEAR(Q.at({1, 0}), -0.8415f, 1e-3f);
    EXPECT_NEAR(Q.at({1, 1}), 0.5403f, 1e-3f);
}

TEST(RoPETest, StartPosOffset) {
    // d_head=2, theta=1
    RoPE rope(/*d_head=*/2, /*max_seq_len=*/8, /*theta_base=*/1.0f);

    // Single token at position 3
    auto Q = Tensor::from_data({1, 2}, {1.0f, 0.0f});
    auto K = Tensor::from_data({1, 2}, {1.0f, 0.0f});

    rope.forward(Q, K, /*start_pos=*/3);

    // pos=3: angle=3, cos(3)≈-0.9900, sin(3)≈0.1411
    // Q = [1*cos(3)-0*sin(3), 1*sin(3)+0*cos(3)] = [cos(3), sin(3)]
    EXPECT_NEAR(Q.at({0, 0}), std::cos(3.0f), 1e-3f);
    EXPECT_NEAR(Q.at({0, 1}), std::sin(3.0f), 1e-3f);
}

TEST(RoPETest, DHead4TwoPairs) {
    // d_head=4, 2 pairs. theta_base=1
    RoPE rope(/*d_head=*/4, /*max_seq_len=*/4, /*theta_base=*/1.0f);

    // freq_0 = 1/1^(0/4) = 1
    // freq_1 = 1/1^(2/4) = 1
    // So all pairs rotate by the same angle = pos

    auto Q = Tensor::from_data({1, 4}, {1.0f, 0.0f, 0.0f, 1.0f});
    auto K = Tensor::from_data({1, 4}, {1.0f, 0.0f, 0.0f, 1.0f});

    rope.forward(Q, K, /*start_pos=*/0);

    // pos=0, angle=0 for all pairs → no rotation
    EXPECT_NEAR(Q.at({0, 0}), 1.0f, 1e-4f);
    EXPECT_NEAR(Q.at({0, 1}), 0.0f, 1e-4f);
    EXPECT_NEAR(Q.at({0, 2}), 0.0f, 1e-4f);
    EXPECT_NEAR(Q.at({0, 3}), 1.0f, 1e-4f);
}

// ===========================================================================
// Precomputed table: cos² + sin² = 1
// ===========================================================================

TEST(RoPETest, CosSinTableUnit) {
    RoPE rope(/*d_head=*/64, /*max_seq_len=*/16, /*theta_base=*/10000.0f);

    // The table is private, but we can verify through rotation of a
    // unit vector at each position.
    for (std::size_t p = 0; p < 4; ++p) {
        // Unit vectors along each pair
        auto Q = Tensor::zeros({1, 64});
        for (std::size_t i = 0; i < 32; ++i) {
            Q.at({0, 2 * i}) = 1.0f; // x=1, y=0 for each pair
        }
        auto K = Tensor::zeros({1, 64});
        for (std::size_t i = 0; i < 32; ++i) {
            K.at({0, 2 * i}) = 1.0f;
        }

        rope.forward(Q, K, /*start_pos=*/p);

        // Each rotated pair should satisfy x² + y² = 1
        for (std::size_t i = 0; i < 32; ++i) {
            const float x = Q.at({0, 2 * i});
            const float y = Q.at({0, 2 * i + 1});
            EXPECT_NEAR(x * x + y * y, 1.0f, 1e-4f)
                << "pos=" << p << " pair=" << i;
        }
    }
}

// ===========================================================================
// Validation
// ===========================================================================

TEST(RoPETest, OddDHeadThrows) {
    EXPECT_THROW({
        RoPE rope(3, 2048);  // 3 is odd
    }, std::invalid_argument);
}

TEST(RoPETest, WrongDimThrows) {
    RoPE rope(4, 2048);
    auto Q = Tensor::ones({2, 6}); // last dim 6 != 4
    auto K = Tensor::ones({2, 4});
    EXPECT_THROW({
        rope.forward(Q, K);
    }, std::invalid_argument);
}

TEST(RoPETest, ExceedsMaxSeqLenThrows) {
    RoPE rope(4, /*max_seq_len=*/1);
    auto Q = Tensor::ones({2, 4}); // seq=2 with start_pos=0 exceeds max_seq_len=1
    auto K = Tensor::ones({2, 4});
    EXPECT_THROW({
        rope.forward(Q, K);
    }, std::invalid_argument);
}

// ===========================================================================
// Metadata
// ===========================================================================

TEST(RoPETest, Metadata) {
    RoPE rope(128, 2048);
    EXPECT_EQ(rope.d_head(), 128);
    EXPECT_EQ(rope.max_seq_len(), 2048);
}

// ===========================================================================
// Large sequence
// ===========================================================================

TEST(RoPETest, LargeSeqNoCrash) {
    RoPE rope(32, 512);
    auto Q = Tensor::ones({128, 32});
    auto K = Tensor::ones({128, 32});
    rope.forward(Q, K);

    EXPECT_FALSE(std::isnan(Q.at({0, 0})));
    EXPECT_FALSE(std::isnan(K.at({0, 0})));
}
