#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "nn/embedding.h"
#include "tensor/tensor.h"

using mt::Embedding;
using mt::Tensor;

// ===========================================================================
// Basic lookup
// ===========================================================================

TEST(EmbeddingTest, BasicLookup) {
    Embedding emb(/*vocab_size=*/5, /*d_model=*/3);

    // Set each row to a known pattern: row[i] = {i*10+0, i*10+1, i*10+2}
    for (std::size_t i = 0; i < 5; ++i) {
        emb.weight().at({i, 0}) = static_cast<float>(i * 10);
        emb.weight().at({i, 1}) = static_cast<float>(i * 10 + 1);
        emb.weight().at({i, 2}) = static_cast<float>(i * 10 + 2);
    }

    // Look up tokens {0, 3, 2}
    std::vector<std::size_t> ids = {0, 3, 2};
    auto out = emb.forward(ids, {3});

    // shape = {3} + {d_model} = {3, 3}
    EXPECT_EQ(out.shape(), (std::vector<std::size_t>{3, 3}));

    // Token 0 → row 0: [0, 1, 2]
    EXPECT_FLOAT_EQ(out.at({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(out.at({0, 1}), 1.0f);
    EXPECT_FLOAT_EQ(out.at({0, 2}), 2.0f);

    // Token 3 → row 3: [30, 31, 32]
    EXPECT_FLOAT_EQ(out.at({1, 0}), 30.0f);
    EXPECT_FLOAT_EQ(out.at({1, 1}), 31.0f);
    EXPECT_FLOAT_EQ(out.at({1, 2}), 32.0f);

    // Token 2 → row 2: [20, 21, 22]
    EXPECT_FLOAT_EQ(out.at({2, 0}), 20.0f);
    EXPECT_FLOAT_EQ(out.at({2, 1}), 21.0f);
    EXPECT_FLOAT_EQ(out.at({2, 2}), 22.0f);
}

TEST(EmbeddingTest, BatchSeqShape) {
    Embedding emb(/*vocab_size=*/4, /*d_model=*/2);

    // Fill with row-major known values
    for (std::size_t i = 0; i < 4; ++i) {
        emb.weight().at({i, 0}) = static_cast<float>(i);
        emb.weight().at({i, 1}) = static_cast<float>(i + 100);
    }

    // shape = {2, 3} → 6 tokens
    std::vector<std::size_t> ids = {0, 1, 2, 3, 0, 3};
    auto out = emb.forward(ids, {2, 3});

    // output shape = {2, 3, 2}
    EXPECT_EQ(out.shape(), (std::vector<std::size_t>{2, 3, 2}));

    // Flat position 0 = token 0: row [0, 100]
    EXPECT_FLOAT_EQ(out.at({0, 0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(out.at({0, 0, 1}), 100.0f);
    // Flat position 3 = token 3: row [3, 103]
    EXPECT_FLOAT_EQ(out.at({1, 0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(out.at({1, 0, 1}), 103.0f);
    // Last token = token 3
    EXPECT_FLOAT_EQ(out.at({1, 2, 0}), 3.0f);
    EXPECT_FLOAT_EQ(out.at({1, 2, 1}), 103.0f);
}

TEST(EmbeddingTest, ZeroVocabSizeWorks) {
    Embedding emb(0, 16);
    std::vector<std::size_t> ids = {};
    auto out = emb.forward(ids, {0});

    EXPECT_EQ(out.shape(), (std::vector<std::size_t>{0, 16}));
    EXPECT_EQ(out.size(), 0);
}

// ===========================================================================
// Validation
// ===========================================================================

TEST(EmbeddingTest, TokenIdOutOfRangeThrows) {
    Embedding emb(/*vocab_size=*/10, /*d_model=*/8);
    std::vector<std::size_t> ids = {0, 5, 10}; // 10 is out of range
    EXPECT_THROW({ static_cast<void>(emb.forward(ids, {3})); }, std::invalid_argument);
}

TEST(EmbeddingTest, SizeMismatchThrows) {
    Embedding emb(10, 8);
    std::vector<std::size_t> ids = {0, 1, 2}; // 3 elements
    EXPECT_THROW({ static_cast<void>(emb.forward(ids, {2, 2})); }, std::invalid_argument); // prod=4
}

TEST(EmbeddingTest, EmptyInputValid) {
    Embedding emb(10, 8);
    std::vector<std::size_t> ids = {};
    auto out = emb.forward(ids, {0});
    EXPECT_EQ(out.shape(), (std::vector<std::size_t>{0, 8}));
}

// ===========================================================================
// Metadata
// ===========================================================================

TEST(EmbeddingTest, Metadata) {
    Embedding emb(32000, 4096);
    EXPECT_EQ(emb.vocab_size(), 32000);
    EXPECT_EQ(emb.d_model(), 4096);
}

TEST(EmbeddingTest, WeightShape) {
    Embedding emb(100, 64);
    EXPECT_EQ(emb.weight().shape(), (std::vector<std::size_t>{100, 64}));
}

// ===========================================================================
// Mutable weight access
// ===========================================================================

TEST(EmbeddingTest, MutableWeightAccess) {
    Embedding emb(2, 3);
    emb.weight().at({0, 0}) = 42.0f;
    EXPECT_FLOAT_EQ(emb.weight().at({0, 0}), 42.0f);
}

// ===========================================================================
// const-correctness
// ===========================================================================

TEST(EmbeddingTest, ConstForward) {
    const Embedding emb(4, 2);
    std::vector<std::size_t> ids = {0, 1};
    auto out = emb.forward(ids, {2});
    // Default zero-initialized weights
    EXPECT_EQ(out.shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ(out.at({0, 0}), 0.0f);
}

// ===========================================================================
// Output is independent of internal state
// ===========================================================================

TEST(EmbeddingTest, OutputIndependentCopy) {
    Embedding emb(3, 2);
    emb.weight().at({0, 0}) = 1.0f;
    emb.weight().at({0, 1}) = 2.0f;

    std::vector<std::size_t> ids = {0};
    auto out = emb.forward(ids, {1});

    // Mutate the weight — should not affect the returned output
    emb.weight().at({0, 0}) = 999.0f;
    EXPECT_FLOAT_EQ(out.at({0, 0}), 1.0f); // still the old value
    EXPECT_FLOAT_EQ(out.at({0, 1}), 2.0f);
}

// ===========================================================================
// Larger scale
// ===========================================================================

TEST(EmbeddingTest, LargeVocabLookup) {
    constexpr std::size_t vocab = 10000;
    constexpr std::size_t d     = 64;
    Embedding emb(vocab, d);

    // Set a known pattern on specific rows
    emb.weight().at({42, 0}) = 3.14f;
    emb.weight().at({9999, 63}) = 2.71f;

    std::vector<std::size_t> ids = {42, 9999};
    auto out = emb.forward(ids, {2});

    EXPECT_FLOAT_EQ(out.at({0, 0}), 3.14f);
    EXPECT_FLOAT_EQ(out.at({1, 63}), 2.71f);
}
