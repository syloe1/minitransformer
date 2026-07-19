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
    src/nn/embedding.cpp)
target_include_directories(mt-nn PUBLIC src)
target_link_libraries(mt-nn PUBLIC mt-tensor)
target_compile_features(mt-nn PUBLIC cxx_std_20)

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

include(GoogleTest)
gtest_discover_tests(tensor_test)
gtest_discover_tests(matmul_test)
gtest_discover_tests(linear_test)
gtest_discover_tests(embedding_test)
