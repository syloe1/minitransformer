#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "tensor/tensor.h"
#include "transformer/kv_cache.h"

using mt::KVCache;
using mt::Tensor;

TEST(KVCacheTest, InitialSizeIsZero) {
    KVCache cache(/*n_layers=*/2, /*n_heads=*/4,
                  /*d_head=*/64, /*max_seq=*/256);
    EXPECT_EQ(cache.size(), 0);
}

TEST(KVCacheTest, StoreAndRetrieve) {
    KVCache cache(1, 1, 4, 8);

    auto K = Tensor::from_data({2, 4}, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    });

    cache.store_k(0, 0, K, /*start_pos=*/0);
    cache.advance(2);

    EXPECT_EQ(cache.size(), 2);
    auto K_out = cache.get_k(0, 0);
    EXPECT_EQ(K_out.shape(), (std::vector<std::size_t>{2, 4}));
    EXPECT_FLOAT_EQ(K_out.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(K_out.at({1, 3}), 8.0f);
}

TEST(KVCacheTest, AppendDecodeStep) {
    KVCache cache(1, 1, 4, 8);

    // Prefill: store 2 tokens
    auto K0 = Tensor::from_data({2, 4}, {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f
    });
    cache.store_k(0, 0, K0, 0);
    cache.advance(2);

    // Decode: append 1 new token at position 2
    auto K1 = Tensor::from_data({1, 4}, {
        0.0f, 0.0f, 1.0f, 0.0f
    });
    cache.store_k(0, 0, K1, 2);
    cache.advance(1);

    EXPECT_EQ(cache.size(), 3);
    auto K = cache.get_k(0, 0);
    EXPECT_FLOAT_EQ(K.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(K.at({1, 1}), 1.0f);
    EXPECT_FLOAT_EQ(K.at({2, 2}), 1.0f);
}

TEST(KVCacheTest, MultiLayerMultiHead) {
    KVCache cache(2, 2, 2, 4);
    auto K = Tensor::from_data({1, 2}, {1.0f, 2.0f});
    auto V = Tensor::from_data({1, 2}, {3.0f, 4.0f});

    // Store per layer/head
    cache.store_k(0, 0, K, 0);
    cache.store_v(0, 0, V, 0);
    cache.store_k(1, 1, K, 1);
    cache.store_v(1, 1, V, 1);
    // advance by what we stored (2 rows for layer0 head0, but each layer advances independently?
    // No — size_ is global. advance manually after all stores.
    cache.advance(2); // we stored 2 positions worth

    EXPECT_EQ(cache.size(), 2);
}

TEST(KVCacheTest, OverflowThrows) {
    KVCache cache(1, 1, 4, /*max_seq=*/2);
    auto K = Tensor::ones({3, 4}); // 3 rows > max_seq
    EXPECT_THROW({
        cache.store_k(0, 0, K, 0);
    }, std::out_of_range);
}

TEST(KVCacheTest, MaxSeqLen) {
    KVCache cache(1, 1, 4, 512);
    EXPECT_EQ(cache.max_seq_len(), 512);
}
