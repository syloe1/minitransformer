#include <gtest/gtest.h>

#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "tensor/tensor.h"

using mt::Tensor;

// ===========================================================================
// Construction
// ===========================================================================

TEST(TensorTest, DefaultConstructorYieldsEmptyTensor) {
    Tensor t;
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0);
    EXPECT_EQ(t.ndim(), 0);
    EXPECT_TRUE(t.shape().empty());
    EXPECT_TRUE(t.strides().empty());
}

TEST(TensorTest, ConstructWithShapeAllocatesCorrectSize) {
    Tensor t(std::vector<std::size_t>{2, 3, 4});
    EXPECT_FALSE(t.empty());
    EXPECT_EQ(t.size(), 24);
    EXPECT_EQ(t.ndim(), 3);
    EXPECT_EQ(t.shape(), (std::vector<std::size_t>{2, 3, 4}));
    EXPECT_EQ(t.byte_size(), 24 * sizeof(float));
}

TEST(TensorTest, ConstructWithInitValueFillsAllElements) {
    Tensor t(std::vector<std::size_t>{2, 3}, 3.14f);
    EXPECT_EQ(t.size(), 6);
    for (std::size_t i = 0; i < t.size(); ++i) {
        EXPECT_FLOAT_EQ(t[i], 3.14f);
    }
}

TEST(TensorTest, ConstructWithScalarShape) {
    Tensor t(std::vector<std::size_t>{1}, 42.0f);
    EXPECT_EQ(t.size(), 1);
    EXPECT_EQ(t.ndim(), 1);
    EXPECT_FLOAT_EQ(t[0], 42.0f);
}

TEST(TensorTest, ConstructWithEmptyShape) {
    Tensor t(std::vector<std::size_t>{});
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0);
}

// ===========================================================================
// Static factories
// ===========================================================================

TEST(TensorTest, ZerosFactory) {
    auto t = Tensor::zeros({3, 5});
    EXPECT_EQ(t.size(), 15);
    for (std::size_t i = 0; i < t.size(); ++i) {
        EXPECT_FLOAT_EQ(t[i], 0.0f);
    }
}

TEST(TensorTest, OnesFactory) {
    auto t = Tensor::ones({4, 2});
    EXPECT_EQ(t.size(), 8);
    for (std::size_t i = 0; i < t.size(); ++i) {
        EXPECT_FLOAT_EQ(t[i], 1.0f);
    }
}

TEST(TensorTest, FromDataFactory) {
    auto t = Tensor::from_data({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    EXPECT_EQ(t.shape(), (std::vector<std::size_t>{2, 3}));
    EXPECT_FLOAT_EQ(t[0], 1.0f);
    EXPECT_FLOAT_EQ(t[5], 6.0f);
}

TEST(TensorTest, FromDataSizeMismatchThrows) {
    EXPECT_THROW(
        {
            auto _ = Tensor::from_data({2, 2}, {1.0f, 2.0f, 3.0f});
        },
        std::invalid_argument);
}

TEST(TensorTest, FromDataEmptyShapeAndData) {
    auto t = Tensor::from_data({}, {});
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0);
}

// ===========================================================================
// Copy semantics
// ===========================================================================

TEST(TensorTest, CopyConstructorCreatesIndependentCopy) {
    auto original = Tensor::ones({2, 3});
    auto copy{original}; // NOLINT — intentional copy

    EXPECT_EQ(copy.shape(), original.shape());
    EXPECT_EQ(copy, original); // values equal

    // Mutate the copy — original must NOT see the change
    copy[0] = 99.0f;
    EXPECT_FLOAT_EQ(copy[0], 99.0f);
    EXPECT_FLOAT_EQ(original[0], 1.0f);
}

TEST(TensorTest, CopyAssignmentCreatesIndependentCopy) {
    auto a = Tensor::ones({2, 3});
    auto b = Tensor::zeros({4, 5});
    b = a;

    EXPECT_EQ(b.shape(), a.shape());
    EXPECT_EQ(b, a);

    b[0] = -1.0f;
    EXPECT_FLOAT_EQ(a[0], 1.0f);
}

TEST(TensorTest, SelfCopyAssignmentIsSafe) {
    auto t = Tensor::ones({3});
    const auto* original_data_ptr = t.data();
    t = t; // NOLINT — intentional self-assignment
    EXPECT_EQ(t.data(), original_data_ptr);
    EXPECT_FLOAT_EQ(t[0], 1.0f);
}

// ===========================================================================
// Move semantics
// ===========================================================================

TEST(TensorTest, MoveConstructorTransfersOwnership) {
    auto t = Tensor::ones({2, 3});
    const auto* original_ptr = t.data();
    const auto original_size = t.size();

    Tensor moved{std::move(t)};

    // moved owns the data
    EXPECT_EQ(moved.data(), original_ptr);
    EXPECT_EQ(moved.size(), original_size);
    EXPECT_EQ(moved.shape(), (std::vector<std::size_t>{2, 3}));

    // t is left in a valid-but-empty state
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0);
    EXPECT_EQ(t.data(), nullptr); // vector is empty → data() may be nullptr
}

TEST(TensorTest, MoveAssignmentTransfersOwnership) {
    auto a = Tensor::ones({2, 3});
    auto b = Tensor::zeros({4, 5});
    const auto* a_ptr = a.data();

    b = std::move(a);

    EXPECT_EQ(b.data(), a_ptr);
    EXPECT_EQ(b.shape(), (std::vector<std::size_t>{2, 3}));

    // a is valid-but-empty
    EXPECT_TRUE(a.empty());
}

TEST(TensorTest, SelfMoveAssignmentIsSafe) {
    auto t = Tensor::ones({3});
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-move"
    t = std::move(t); // intentional self-move: must not crash
#pragma GCC diagnostic pop
    // After self-move the tensor should still be usable.
    EXPECT_EQ(t.size(), 3);
    EXPECT_FLOAT_EQ(t[0], 1.0f);
}

// ===========================================================================
// Strides (row-major)
// ===========================================================================

TEST(TensorTest, Strides1D) {
    Tensor t(std::vector<std::size_t>{5});
    EXPECT_EQ(t.strides(), (std::vector<std::size_t>{1}));
}

TEST(TensorTest, Strides2D) {
    Tensor t(std::vector<std::size_t>{3, 4});
    // Row-major: last dim stride = 1, then 4, then 12
    EXPECT_EQ(t.strides(), (std::vector<std::size_t>{4, 1}));
}

TEST(TensorTest, Strides3D) {
    Tensor t(std::vector<std::size_t>{2, 3, 4});
    EXPECT_EQ(t.strides(), (std::vector<std::size_t>{12, 4, 1}));
}

TEST(TensorTest, Strides4D) {
    Tensor t(std::vector<std::size_t>{2, 3, 4, 5});
    // [3*4*5=60, 4*5=20, 5, 1]
    EXPECT_EQ(t.strides(), (std::vector<std::size_t>{60, 20, 5, 1}));
}

// ===========================================================================
// Multi-dimensional access
// ===========================================================================

TEST(TensorTest, At2D) {
    // Fill with i*cols + j so each element has a unique, predictable value
    auto data = std::vector<float>(6);
    std::iota(data.begin(), data.end(), 0.0f); // 0,1,2,3,4,5
    auto t = Tensor::from_data({2, 3}, std::move(data));

    EXPECT_FLOAT_EQ(t.at({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(t.at({0, 2}), 2.0f);
    EXPECT_FLOAT_EQ(t.at({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(t.at({1, 2}), 5.0f);
}

TEST(TensorTest, At3D) {
    auto data = std::vector<float>(24);
    std::iota(data.begin(), data.end(), 0.0f);
    auto t = Tensor::from_data({2, 3, 4}, std::move(data));

    // offset = i*12 + j*4 + k
    EXPECT_FLOAT_EQ(t.at({0, 0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(t.at({0, 0, 3}), 3.0f);
    EXPECT_FLOAT_EQ(t.at({0, 2, 0}), 8.0f);
    EXPECT_FLOAT_EQ(t.at({1, 0, 0}), 12.0f);
    EXPECT_FLOAT_EQ(t.at({1, 2, 3}), 23.0f);
}

TEST(TensorTest, AtMutable) {
    auto t = Tensor::zeros({2, 3});
    t.at({1, 2}) = 7.0f;
    EXPECT_FLOAT_EQ(t.at({1, 2}), 7.0f);
    EXPECT_FLOAT_EQ(t[5], 7.0f); // linear offset = 1*3 + 2 = 5
}

TEST(TensorTest, AtConstTensor) {
    const auto t = Tensor::ones({2, 3});
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
}

TEST(TensorTest, AtWrongDimsThrows) {
    auto t = Tensor::zeros({2, 3});
    EXPECT_THROW({ t.at({0}); }, std::out_of_range);
    EXPECT_THROW({ t.at({0, 1, 2}); }, std::out_of_range);
}

TEST(TensorTest, AtOutOfRangeThrows) {
    auto t = Tensor::zeros({2, 3});
    EXPECT_THROW({ t.at({2, 0}); }, std::out_of_range);  // dim 0 >= 2
    EXPECT_THROW({ t.at({0, 3}); }, std::out_of_range);  // dim 1 >= 3
}

// ===========================================================================
// Offset calculation
// ===========================================================================

TEST(TensorTest, OffsetComputation2D) {
    Tensor t(std::vector<std::size_t>{3, 4});
    // offset = i*4 + j
    EXPECT_EQ(t.offset({0, 0}), 0);
    EXPECT_EQ(t.offset({0, 3}), 3);
    EXPECT_EQ(t.offset({2, 1}), 9);
    EXPECT_EQ(t.offset({2, 3}), 11);
}

TEST(TensorTest, OffsetWrongDimsThrows) {
    Tensor t(std::vector<std::size_t>{3, 4});
    EXPECT_THROW({ static_cast<void>(t.offset({0})); }, std::out_of_range);
}

TEST(TensorTest, OffsetOutOfRangeThrows) {
    Tensor t(std::vector<std::size_t>{3, 4});
    EXPECT_THROW({ static_cast<void>(t.offset({3, 0})); }, std::out_of_range);
}

TEST(TensorTest, OffsetUnchecked) {
    Tensor t(std::vector<std::size_t>{3, 4});
    EXPECT_EQ(t.offset_unchecked({2, 1}), 9);
}

// ===========================================================================
// Data / span views
// ===========================================================================

TEST(TensorTest, DataPointerIsValid) {
    auto t = Tensor::zeros({5});
    float* p = t.data();
    ASSERT_NE(p, nullptr);
    p[0] = 1.0f;
    EXPECT_FLOAT_EQ(t[0], 1.0f);
}

TEST(TensorTest, ConstDataPointerIsValid) {
    const auto t = Tensor::ones({5});
    const float* p = t.data();
    ASSERT_NE(p, nullptr);
    EXPECT_FLOAT_EQ(p[0], 1.0f);
}

TEST(TensorTest, SpanCoversEntirePayload) {
    auto t = Tensor::ones({5});
    auto s = t.span();
    EXPECT_EQ(s.size(), 5);
    EXPECT_EQ(s.data(), t.data());
    s[0] = 42.0f;
    EXPECT_FLOAT_EQ(t[0], 42.0f);
}

TEST(TensorTest, ConstSpanCoversEntirePayload) {
    const auto t = Tensor::ones({5});
    auto s = t.span();
    EXPECT_EQ(s.size(), 5);
    EXPECT_FLOAT_EQ(s[0], 1.0f);
}

// ===========================================================================
// Reshape
// ===========================================================================

TEST(TensorTest, ReshapePreservesData) {
    auto data = std::vector<float>(6);
    std::iota(data.begin(), data.end(), 0.0f);
    auto t = Tensor::from_data({2, 3}, std::move(data));

    t.reshape({3, 2});
    EXPECT_EQ(t.shape(), (std::vector<std::size_t>{3, 2}));
    EXPECT_EQ(t.size(), 6);

    // Element at flat index 3 was {1,0} in [2,3], now is {1,1} in [3,2]
    EXPECT_FLOAT_EQ(t.at({1, 1}), 3.0f);
    // Element at flat index 5 was {1,2} in [2,3], now is {2,1} in [3,2]
    EXPECT_FLOAT_EQ(t.at({2, 1}), 5.0f);
}

TEST(TensorTest, ReshapeSizeMismatchThrows) {
    auto t = Tensor::zeros({2, 3});
    EXPECT_THROW({ t.reshape({2, 2}); }, std::invalid_argument);
    EXPECT_THROW({ t.reshape({12}); }, std::invalid_argument);
}

TEST(TensorTest, ReshapeBetween1DAnd3D) {
    auto data = std::vector<float>(12);
    std::iota(data.begin(), data.end(), 0.0f);
    auto t = Tensor::from_data({12}, std::move(data));

    t.reshape({2, 2, 3});
    EXPECT_EQ(t.ndim(), 3);
    EXPECT_FLOAT_EQ(t.at({0, 0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(t.at({1, 1, 2}), 11.0f);
}

TEST(TensorTest, ReshapePreservesStrides) {
    auto t = Tensor::zeros({2, 3, 4});
    EXPECT_EQ(t.strides(), (std::vector<std::size_t>{12, 4, 1}));
    t.reshape({4, 6});
    EXPECT_EQ(t.strides(), (std::vector<std::size_t>{6, 1}));
}

// ===========================================================================
// Comparison
// ===========================================================================

TEST(TensorTest, EqualTensorsCompareEqual) {
    auto a = Tensor::ones({2, 3});
    auto b = Tensor::ones({2, 3});
    EXPECT_EQ(a, b);
}

TEST(TensorTest, DifferentShapesCompareNotEqual) {
    auto a = Tensor::ones({2, 3});
    auto b = Tensor::ones({3, 2});
    EXPECT_NE(a, b);
}

TEST(TensorTest, DifferentValuesCompareNotEqual) {
    auto a = Tensor::ones({2, 3});
    auto b = Tensor::zeros({2, 3});
    EXPECT_NE(a, b);
}

// ===========================================================================
// Edge / boundary cases
// ===========================================================================

TEST(TensorTest, Large1DTensor) {
    constexpr std::size_t N = 1'000'000;
    auto t = Tensor::ones({N});
    EXPECT_EQ(t.size(), N);
    EXPECT_EQ(t.ndim(), 1);
    EXPECT_FLOAT_EQ(t[0], 1.0f);
    EXPECT_FLOAT_EQ(t[N - 1], 1.0f);
}

TEST(TensorTest, SingleElementTensorAllDimsSize1) {
    auto t = Tensor::ones({1, 1, 1, 1});
    EXPECT_EQ(t.size(), 1);
    EXPECT_EQ(t.ndim(), 4);
    EXPECT_FLOAT_EQ(t.at({0, 0, 0, 0}), 1.0f);
}

TEST(TensorTest, ZerosEmptyShapeReturnsEmptyTensor) {
    auto t = Tensor::zeros({});
    EXPECT_TRUE(t.empty());
}

TEST(TensorTest, OnesEmptyShapeReturnsEmptyTensor) {
    auto t = Tensor::ones({});
    EXPECT_TRUE(t.empty());
}

// ===========================================================================
// const-correctness on accessors
// ===========================================================================

TEST(TensorTest, ConstAccessorsDoNotModify) {
    const auto t = Tensor::zeros({3, 4});
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.size(), 12);
    EXPECT_FALSE(t.empty());
    EXPECT_EQ(t.shape(), (std::vector<std::size_t>{3, 4}));
    EXPECT_NO_THROW({ static_cast<void>(t.offset({0, 0})); });
    EXPECT_FLOAT_EQ(t[0], 0.0f);
    EXPECT_FLOAT_EQ(t.at({0, 0}), 0.0f);
}
