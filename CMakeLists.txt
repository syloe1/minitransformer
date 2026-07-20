cmake_minimum_required(VERSION 3.24)
project(mini-transformer
    VERSION 0.1.0
    DESCRIPTION "Industrial-grade Decoder-only Transformer in C++20"
    LANGUAGES CXX)

# ---------------------------------------------------------------------------
# C++20 — required for std::span, designated initialisers, concepts, ...
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Compiler warnings — industrial standard: -Wall -Wextra -Wpedantic
# Treat warnings as errors in dev; relax for release if needed.
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    # We do NOT use -Werror globally so clangd/exploratory builds don't break.
    # CI / pre-commit should add -Werror.
endif()

# Address Sanitiser (optional, off by default; enable with -DENABLE_ASAN=ON)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

# ---------------------------------------------------------------------------
# Dependencies — fetch GoogleTest automatically
# ---------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
    GIT_SHALLOW    ON
)

# GoogleTest's internal compilation may trigger warnings; suppress them.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# ---------------------------------------------------------------------------
# Library: tensor
# ---------------------------------------------------------------------------
add_library(mt-tensor STATIC
    src/tensor/tensor.h
    src/tensor/tensor.cpp)
target_include_directories(mt-tensor PUBLIC src)
target_compile_features(mt-tensor PUBLIC cxx_std_20)

# ---------------------------------------------------------------------------
# Library: nn  (neural-network operations: matmul, linear, softmax, ...)
# ---------------------------------------------------------------------------
add_library(mt-nn STATIC
    src/nn/matmul.h
    src/nn/matmul.cpp
    src/nn/linear.h
    src/nn/linear.cpp
    src/nn/embedding.h
    src/nn/embedding.cpp
    src/nn/softmax.h
    src/nn/softmax.cpp
    src/nn/mlp.h
    src/nn/mlp.cpp
    src/nn/rms_norm.h
    src/nn/rms_norm.cpp)
target_include_directories(mt-nn PUBLIC src)
target_link_libraries(mt-nn PUBLIC mt-tensor)
target_compile_features(mt-nn PUBLIC cxx_std_20)

# ---------------------------------------------------------------------------
# Library: rope
# ---------------------------------------------------------------------------
add_library(mt-rope STATIC
    src/rope/rope.h
    src/rope/rope.cpp)
target_include_directories(mt-rope PUBLIC src)
target_link_libraries(mt-rope PUBLIC mt-tensor)
target_compile_features(mt-rope PUBLIC cxx_std_20)

# ---------------------------------------------------------------------------
# Library: transformer
# ---------------------------------------------------------------------------
add_library(mt-transformer STATIC
    src/transformer/decoder_block.h
    src/transformer/decoder_block.cpp
    src/transformer/kv_cache.h
    src/transformer/kv_cache.cpp
    src/transformer/transformer.h
    src/transformer/transformer.cpp)
target_include_directories(mt-transformer PUBLIC src)
target_link_libraries(mt-transformer PUBLIC mt-attention mt-nn mt-rope)
target_compile_features(mt-transformer PUBLIC cxx_std_20)

# ---------------------------------------------------------------------------
# Library: attention
# ---------------------------------------------------------------------------
add_library(mt-attention STATIC
    src/attention/attention.h
    src/attention/attention.cpp
    src/attention/multi_head_attention.h
    src/attention/multi_head_attention.cpp)
target_include_directories(mt-attention PUBLIC src)
target_link_libraries(mt-attention PUBLIC mt-nn mt-rope)
target_compile_features(mt-attention PUBLIC cxx_std_20)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

add_executable(tensor_test test/tensor_test.cpp)
target_link_libraries(tensor_test PRIVATE mt-tensor GTest::gtest GTest::gtest_main)
target_compile_features(tensor_test PRIVATE cxx_std_20)

add_executable(matmul_test test/matmul_test.cpp)
target_link_libraries(matmul_test PRIVATE mt-nn GTest::gtest GTest::gtest_main)
target_compile_features(matmul_test PRIVATE cxx_std_20)

add_executable(linear_test test/linear_test.cpp)
target_link_libraries(linear_test PRIVATE mt-nn GTest::gtest GTest::gtest_main)
target_compile_features(linear_test PRIVATE cxx_std_20)

add_executable(embedding_test test/embedding_test.cpp)
target_link_libraries(embedding_test PRIVATE mt-nn GTest::gtest GTest::gtest_main)
target_compile_features(embedding_test PRIVATE cxx_std_20)

add_executable(softmax_test test/softmax_test.cpp)
target_link_libraries(softmax_test PRIVATE mt-nn GTest::gtest GTest::gtest_main)
target_compile_features(softmax_test PRIVATE cxx_std_20)

add_executable(attention_test test/attention_test.cpp)
target_link_libraries(attention_test PRIVATE mt-attention GTest::gtest GTest::gtest_main)
target_compile_features(attention_test PRIVATE cxx_std_20)

add_executable(multi_head_attention_test test/multi_head_attention_test.cpp)
target_link_libraries(multi_head_attention_test PRIVATE mt-attention GTest::gtest GTest::gtest_main)
target_compile_features(multi_head_attention_test PRIVATE cxx_std_20)

add_executable(rope_test test/rope_test.cpp)
target_link_libraries(rope_test PRIVATE mt-rope GTest::gtest GTest::gtest_main)
target_compile_features(rope_test PRIVATE cxx_std_20)

add_executable(mlp_test test/mlp_test.cpp)
target_link_libraries(mlp_test PRIVATE mt-nn GTest::gtest GTest::gtest_main)
target_compile_features(mlp_test PRIVATE cxx_std_20)

add_executable(rms_norm_test test/rms_norm_test.cpp)
target_link_libraries(rms_norm_test PRIVATE mt-nn GTest::gtest GTest::gtest_main)
target_compile_features(rms_norm_test PRIVATE cxx_std_20)

add_executable(decoder_block_test test/decoder_block_test.cpp)
target_link_libraries(decoder_block_test PRIVATE mt-transformer GTest::gtest GTest::gtest_main)
target_compile_features(decoder_block_test PRIVATE cxx_std_20)

add_executable(kv_cache_test test/kv_cache_test.cpp)
target_link_libraries(kv_cache_test PRIVATE mt-transformer GTest::gtest GTest::gtest_main)
target_compile_features(kv_cache_test PRIVATE cxx_std_20)

add_executable(transformer_test test/transformer_test.cpp)
target_link_libraries(transformer_test PRIVATE mt-transformer GTest::gtest GTest::gtest_main)
target_compile_features(transformer_test PRIVATE cxx_std_20)

include(GoogleTest)
gtest_discover_tests(tensor_test)
gtest_discover_tests(matmul_test)
gtest_discover_tests(linear_test)
gtest_discover_tests(embedding_test)
gtest_discover_tests(softmax_test)
gtest_discover_tests(attention_test)
gtest_discover_tests(multi_head_attention_test)
gtest_discover_tests(rope_test)
gtest_discover_tests(mlp_test)
gtest_discover_tests(rms_norm_test)
gtest_discover_tests(decoder_block_test)
gtest_discover_tests(kv_cache_test)
gtest_discover_tests(transformer_test)
