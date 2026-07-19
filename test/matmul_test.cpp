#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "nn/matmul.h"
#include "tensor/tensor.h"

using mt::Tensor;

namespace {

/// Helper: create a 2-D tensor from a nested initializer-list style vector.
/// Data must be row-major: data[row*cols + col].
auto make_2d(std::vector<std::size_t> shape,
             std::vector<float> data) -> Tensor {
    return Tensor::from_data(std::move(shape), std::move(data));
}

/// Helper: element-wise approximate equality for floating-point comparison.
void expect_near(const Tensor& a, const Tensor& b, float eps = 1e-5f) {
    ASSERT_EQ(a.shape(), b.shape()) << "shape mismatch";
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], b[i], eps) << "at index " << i;
    }
}

} // namespace

// ===========================================================================
// Basic correctness
// ===========================================================================

TEST(MatmulTest, Basic2x3Times3x2) {
    // A = [[1, 2, 3],
    //      [4, 5, 6]]     shape [2, 3]
    auto A = make_2d({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // B = [[1, 2],
    //      [3, 4],
    //      [5, 6]]         shape [3, 2]
    auto B = make_2d({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    auto C = mt::matmul(A, B);

    // C = [[1*1+2*3+3*5, 1*2+2*4+3*6],
    //      [4*1+5*3+6*5, 4*2+5*4+6*6]]
    //   = [[22, 28],
    //      [49, 64]]
    EXPECT_EQ(C.shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ(C.at({0, 0}), 22.0f);
    EXPECT_FLOAT_EQ(C.at({0, 1}), 28.0f);
    EXPECT_FLOAT_EQ(C.at({1, 0}), 49.0f);
    EXPECT_FLOAT_EQ(C.at({1, 1}), 64.0f);
}

TEST(MatmulTest, Square4x4) {
    // A = 4×4 with row i all equal to i+1
    auto A = make_2d({4, 4}, {
        1.0f, 1.0f, 1.0f, 1.0f,  // row 0
        2.0f, 2.0f, 2.0f, 2.0f,  // row 1
        3.0f, 3.0f, 3.0f, 3.0f,  // row 2
        4.0f, 4.0f, 4.0f, 4.0f   // row 3
    });

    // B = identity
    auto B = make_2d({4, 4}, {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    });

    auto C = mt::matmul(A, B);

    // A × I = A
    EXPECT_EQ(C.shape(), (std::vector<std::size_t>{4, 4}));
    expect_near(C, A);
}

TEST(MatmulTest, Rectangular1x100Times100x1) {
    // A: 1×100  all ones
    auto A = make_2d({1, 100}, std::vector<float>(100, 1.0f));
    // B: 100×1  all twos
    auto B = make_2d({100, 1}, std::vector<float>(100, 2.0f));

    auto C = mt::matmul(A, B);

    // C = [100 * (1*2)] = [200]
    EXPECT_EQ(C.shape(), (std::vector<std::size_t>{1, 1}));
    EXPECT_FLOAT_EQ(C.at({0, 0}), 200.0f);
}

TEST(MatmulTest, SingleElementMatrices) {
    auto A = make_2d({1, 1}, {3.0f});
    auto B = make_2d({1, 1}, {7.0f});

    auto C = mt::matmul(A, B);

    EXPECT_EQ(C.shape(), (std::vector<std::size_t>{1, 1}));
    EXPECT_FLOAT_EQ(C.at({0, 0}), 21.0f);
}

// ===========================================================================
// transpose_b
// ===========================================================================

TEST(MatmulTest, TransposeB_Basic) {
    // A: [2, 3]
    auto A = make_2d({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // B: [2, 3]  (same shape as A, but treated as transposed)
    //    B = [[1, 2, 3],
    //         [4, 5, 6]]
    //    B^T logically = [[1, 4],
    //                      [2, 5],
    //                      [3, 6]]
    auto B = make_2d({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    auto C = mt::matmul(A, B, /*transpose_b=*/true);

    // C = A × B^T  = [2,3] × [3,2] = [2,2]
    // C[0][0] = 1*1 + 2*2 + 3*3 = 14
    // C[0][1] = 1*4 + 2*5 + 3*6 = 32
    // C[1][0] = 4*1 + 5*2 + 6*3 = 32
    // C[1][1] = 4*4 + 5*5 + 6*6 = 77
    EXPECT_EQ(C.shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ(C.at({0, 0}), 14.0f);
    EXPECT_FLOAT_EQ(C.at({0, 1}), 32.0f);
    EXPECT_FLOAT_EQ(C.at({1, 0}), 32.0f);
    EXPECT_FLOAT_EQ(C.at({1, 1}), 77.0f);
}

TEST(MatmulTest, TransposeB_EquivalentToExplicitTranspose) {
    // Verify that matmul(A, B, transpose_b=true) gives same result
    // as manually transposing B and calling standard matmul.

    auto A = make_2d({3, 4}, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    });

    // B stored as [2, 4], so B^T is [4, 2]
    auto B = make_2d({2, 4}, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    });

    auto C1 = mt::matmul(A, B, /*transpose_b=*/true);

    // Manually build B_explicit = B^T as [4, 2]
    // B original: row0 = [1,2,3,4], row1 = [5,6,7,8]
    // B^T: col0 = [1,5], col1 = [2,6], col2 = [3,7], col3 = [4,8]
    // In row-major: row0=[1,5], row1=[2,6], row2=[3,7], row3=[4,8]
    auto B_explicit = make_2d({4, 2}, {
        1.0f, 5.0f,
        2.0f, 6.0f,
        3.0f, 7.0f,
        4.0f, 8.0f
    });

    auto C2 = mt::matmul(A, B_explicit, /*transpose_b=*/false);

    expect_near(C1, C2);
}

TEST(MatmulTest, TransposeB_Square) {
    auto A = make_2d({3, 3}, {
        1.0f, 0.0f, 2.0f,
        0.0f, 3.0f, 0.0f,
        4.0f, 0.0f, 5.0f
    });

    // B = A (same), so A × A^T
    auto B = make_2d({3, 3}, {
        1.0f, 0.0f, 2.0f,
        0.0f, 3.0f, 0.0f,
        4.0f, 0.0f, 5.0f
    });

    auto C = mt::matmul(A, B, /*transpose_b=*/true);

    // C[0][0] = 1*1 + 0*0 + 2*2 = 5
    // C[0][1] = 1*0 + 0*3 + 2*0 = 0
    // C[0][2] = 1*4 + 0*0 + 2*5 = 14
    // C[1][0] = 0*1 + 3*0 + 0*2 = 0
    // C[1][1] = 0*0 + 3*3 + 0*0 = 9
    // C[1][2] = 0*4 + 3*0 + 0*5 = 0
    // C[2][0] = 4*1 + 0*0 + 5*2 = 14
    // C[2][1] = 4*0 + 0*3 + 5*0 = 0
    // C[2][2] = 4*4 + 0*0 + 5*5 = 41
    EXPECT_EQ(C.shape(), (std::vector<std::size_t>{3, 3}));
    EXPECT_FLOAT_EQ(C.at({0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(C.at({0, 1}), 0.0f);
    EXPECT_FLOAT_EQ(C.at({0, 2}), 14.0f);
    EXPECT_FLOAT_EQ(C.at({1, 0}), 0.0f);
    EXPECT_FLOAT_EQ(C.at({1, 1}), 9.0f);
    EXPECT_FLOAT_EQ(C.at({1, 2}), 0.0f);
    EXPECT_FLOAT_EQ(C.at({2, 0}), 14.0f);
    EXPECT_FLOAT_EQ(C.at({2, 1}), 0.0f);
    EXPECT_FLOAT_EQ(C.at({2, 2}), 41.0f);
}

// ===========================================================================
// Shape validation — error cases
// ===========================================================================

TEST(MatmulTest, Non2D_A_Throws) {
    auto A = Tensor::ones({2, 3, 4}); // 3-D
    auto B = Tensor::ones({3, 2});
    EXPECT_THROW({ static_cast<void>(mt::matmul(A, B)); }, std::invalid_argument);
}

TEST(MatmulTest, Non2D_B_Throws) {
    auto A = Tensor::ones({2, 3});
    auto B = Tensor::ones({3}); // 1-D
    EXPECT_THROW({ static_cast<void>(mt::matmul(A, B)); }, std::invalid_argument);
}

TEST(MatmulTest, InnerDimensionMismatch_Standard) {
    auto A = Tensor::ones({2, 3}); // K = 3
    auto B = Tensor::ones({4, 2}); // K = 4, mismatch
    EXPECT_THROW({ static_cast<void>(mt::matmul(A, B)); }, std::invalid_argument);
}

TEST(MatmulTest, InnerDimensionMismatch_TransposeB) {
    auto A = Tensor::ones({2, 3}); // K = 3
    auto B = Tensor::ones({2, 4}); // B is [2,4], B^T is [4,2], inner=4 ≠ 3
    EXPECT_THROW({ static_cast<void>(mt::matmul(A, B, true)); }, std::invalid_argument);
}

// ===========================================================================
// Zero / edge cases
// ===========================================================================

TEST(MatmulTest, ZeroMatrixProduct) {
    auto A = Tensor::zeros({3, 4});
    auto B = Tensor::ones({4, 5});

    auto C = mt::matmul(A, B);
    EXPECT_EQ(C.shape(), (std::vector<std::size_t>{3, 5}));
    for (std::size_t i = 0; i < C.size(); ++i) {
        EXPECT_FLOAT_EQ(C[i], 0.0f);
    }
}

TEST(MatmulTest, ResultIsIndependentCopy) {
    auto A = Tensor::ones({2, 3});
    auto B = Tensor::ones({3, 2});

    auto C = mt::matmul(A, B);
    // Mutate the inputs — should not affect C
    A.at({0, 0}) = 999.0f;
    B.at({0, 0}) = 999.0f;

    // Original result: each element = 3 * 1.0 = 3.0
    EXPECT_FLOAT_EQ(C.at({0, 0}), 3.0f);
}

// ===========================================================================
// const-correctness
// ===========================================================================

TEST(MatmulTest, AcceptsConstTensors) {
    const auto A = Tensor::ones({2, 3});
    const auto B = Tensor::ones({3, 4});

    auto C = mt::matmul(A, B);

    EXPECT_EQ(C.shape(), (std::vector<std::size_t>{2, 4}));
    // Each element = K * 1.0 = 3.0
    for (std::size_t i = 0; i < C.size(); ++i) {
        EXPECT_FLOAT_EQ(C[i], 3.0f);
    }
}

// ===========================================================================
// Larger matrix (stress test + basic perf sanity)
// ===========================================================================

TEST(MatmulTest, Large256x256) {
    constexpr std::size_t N = 256;
    auto A = Tensor::ones({N, N});
    auto B = Tensor::ones({N, N});

    auto C = mt::matmul(A, B);

    EXPECT_EQ(C.shape(), (std::vector<std::size_t>{N, N}));
    // Each element = N * 1.0 = 256.0
    EXPECT_NEAR(C.at({0, 0}), static_cast<float>(N), 1e-4f);
    EXPECT_NEAR(C.at({N - 1, N - 1}), static_cast<float>(N), 1e-4f);
}
