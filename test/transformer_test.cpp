#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "tensor/tensor.h"
#include "transformer/transformer.h"

using mt::Transformer;
using mt::Tensor;

// ===========================================================================
// Forward
// ===========================================================================

TEST(TransformerTest, ForwardOutputShape) {
    // Tiny model: vocab=16, d_model=8, n_heads=2, n_layers=1
    Transformer model(/*vocab=*/16, /*d_model=*/8,
                      /*n_heads=*/2, /*n_layers=*/1, /*max_seq=*/32);

    std::vector<std::size_t> ids = {0, 1, 2, 3};
    auto logits = model.forward(ids, {1, 4});

    EXPECT_EQ(logits.shape(), (std::vector<std::size_t>{1, 4, 16}));
}

TEST(TransformerTest, ForwardMultipleLayers) {
    Transformer model(/*vocab=*/8, /*d_model=*/4,
                      /*n_heads=*/2, /*n_layers=*/2, /*max_seq=*/16);

    std::vector<std::size_t> ids = {0, 1};
    auto logits = model.forward(ids, {1, 2});

    EXPECT_EQ(logits.shape(), (std::vector<std::size_t>{1, 2, 8}));
}

TEST(TransformerTest, BatchInput) {
    Transformer model(/*vocab=*/8, /*d_model=*/4,
                      /*n_heads=*/2, /*n_layers=*/1, /*max_seq=*/16);

    std::vector<std::size_t> ids = {0, 1, 2, 3, 4, 5};
    auto logits = model.forward(ids, {2, 3});

    EXPECT_EQ(logits.shape(), (std::vector<std::size_t>{2, 3, 8}));
}

// ===========================================================================
// Generation
// ===========================================================================

TEST(TransformerTest, GenerateReturnsTokens) {
    Transformer model(/*vocab=*/16, /*d_model=*/8,
                      /*n_heads=*/2, /*n_layers=*/1, /*max_seq=*/64);

    std::vector<std::size_t> prompt = {1, 2, 3};
    auto result = model.generate(prompt, /*max_new_tokens=*/3);

    // Should have prompt + 3 new tokens
    EXPECT_EQ(result.size(), 6);
    EXPECT_EQ(result[0], 1);
    EXPECT_EQ(result[1], 2);
    EXPECT_EQ(result[2], 3);
}

TEST(TransformerTest, GenerateAllTokensInVocabRange) {
    Transformer model(/*vocab=*/16, /*d_model=*/8,
                      /*n_heads=*/2, /*n_layers=*/1, /*max_seq=*/32);

    std::vector<std::size_t> prompt = {0};
    auto result = model.generate(prompt, /*max_new_tokens=*/5);

    for (auto id : result) {
        EXPECT_LT(id, 16); // all tokens < vocab_size
    }
}

// ===========================================================================
// Metadata
// ===========================================================================

TEST(TransformerTest, Metadata) {
    Transformer model(100, 64, 8, 3, 256);
    EXPECT_EQ(model.vocab_size(), 100);
    EXPECT_EQ(model.d_model(), 64);
    EXPECT_EQ(model.n_heads(), 8);
    EXPECT_EQ(model.n_layers(), 3);
}

// ===========================================================================
// Sub-module access
// ===========================================================================

TEST(TransformerTest, SubModuleAccess) {
    Transformer model(32, 16, 2, 2, 64);
    EXPECT_NO_THROW({
        (void)model.embed();
        (void)model.blocks();
        (void)model.final_norm();
        (void)model.lm_head();
    });
    EXPECT_EQ(model.blocks().size(), 2);
}

// ===========================================================================
// Validation
// ===========================================================================

TEST(TransformerTest, WrongTokenCountThrows) {
    Transformer model(8, 4, 2, 1, 16);
    std::vector<std::size_t> ids = {0, 1, 2}; // 3 tokens, shape says 4
    EXPECT_THROW({
        static_cast<void>(model.forward(ids, {2, 2}));
    }, std::invalid_argument);
}

// ===========================================================================
// Zero weights: with default (zero) weights, output logits are all zero
// ===========================================================================

TEST(TransformerTest, ZeroWeightsZeroLogits) {
    Transformer model(8, 4, 2, 1, 16);
    std::vector<std::size_t> ids = {0};
    auto logits = model.forward(ids, {1, 1});

    // All Linear weights are zero → lm_head output is all zeros
    EXPECT_EQ(logits.shape(), (std::vector<std::size_t>{1, 1, 8}));
    for (std::size_t i = 0; i < logits.size(); ++i) {
        EXPECT_FLOAT_EQ(logits[i], 0.0f);
    }
}
