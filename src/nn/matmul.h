#pragma once

#include "tensor/tensor.h"

namespace mt {

// ---------------------------------------------------------------------------
// matmul — standard matrix multiplication  C = A × B
// ---------------------------------------------------------------------------
// This is the single most performance-critical operation in the entire
// Transformer pipeline.  Every Linear projection, every attention-score
// computation (Q×K^T), and every weighted sum (score×V) reduces to matmul.
//
// Semantics:
//   transpose_b=false  →  C[m][n] = Σₖ A[m][k] · B[k][n]
//       A: [M, K]       B: [K, N]       C: [M, N]
//
//   transpose_b=true   →  C[m][n] = Σₖ A[m][k] · B[n][k]
//       A: [M, K]       B: [N, K]       C: [M, N]
//   (B is logically transposed; no data is copied.)
//
// Loop ordering:
//   standard:    i → k → j    (innermost j walks contiguous B[k][j])
//   transpose_b: i → j → k    (innermost k walks contiguous A[i][k] & B[j][k])
//
// Design rationale:
//   - Free function: stateless, pure compute; analogous to std::sort.
//   - transpose_b flag: zero-copy "transpose" — only changes indexing,
//     akin to a database covering index.
//   - Raw pointers in inner loops: avoids the per-element overhead of
//     Tensor::at() which would do bounds-checking + stride multiply on
//     every iteration of the M×K×N innermost loops.
// ---------------------------------------------------------------------------

/// Standard matrix multiplication.
///
/// @param A  Left operand  shape [M, K]
/// @param B  Right operand shape [K, N]  (or [N, K] when transpose_b = true)
/// @param transpose_b  If true, B is treated as [N, K] and transposed
///                     internally (zero-copy).
/// @return C  shape [M, N]
///
/// @throws std::invalid_argument if A.ndim() != 2 or B.ndim() != 2
/// @throws std::invalid_argument if inner dimensions do not match

// tranformer框架最核心， 性能关键的计算算子
// 注意力分数Q*K^T  注意力加权求和score * V全部依赖矩阵乘法

/*
 Amk *Bkn = Cmn
transpose_b = false零拷贝转置

transpose_b = true 逻辑转置， 最内层k同时变量a b 行， 提升cpu缓存效率
*/
[[nodiscard]] auto matmul(const Tensor &A, const Tensor &B,
                          bool transpose_b = false) -> Tensor;
} // namespace mt
