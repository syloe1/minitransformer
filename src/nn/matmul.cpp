#include "matmul.h"

#include <cstddef>
#include <sstream>
#include <stdexcept>

namespace mt {

// ===========================================================================
// Internal helpers
// ===========================================================================
// 算子内部私有工具， 不对外暴露接口
namespace {
// 检查传入张量必须是 二维矩阵
/// Validate that `t` is a 2-D tensor, throw with a readable message otherwise.
void validate_2d(const Tensor &t, const char *name) {
  if (t.ndim() != 2) {
    std::ostringstream oss;
    oss << "matmul: " << name << " must be 2-D, got " << t.ndim()
        << "-D shape [";
    const auto &sh = t.shape();
    for (std::size_t d = 0; d < sh.size(); ++d) {
      if (d > 0)
        oss << ", ";
      oss << sh[d];
    }
    oss << ']';
    throw std::invalid_argument(oss.str());
  }
}
//
} // namespace

// ===========================================================================
// matmul
// ===========================================================================

auto matmul(const Tensor &A, const Tensor &B, const bool transpose_b)
    -> Tensor {

  validate_2d(A, "A");
  validate_2d(B, "B");

  const auto &a_shape = A.shape(); // [M, K]
  const auto &b_shape = B.shape(); // [K, N] or [N, K]

  const std::size_t M = a_shape[0];
  const std::size_t K = a_shape[1];
  const std::size_t B_rows = b_shape[0];
  const std::size_t B_cols = b_shape[1];

  // Validate inner-dimension compatibility.
  //
  // Standard:  A[M,K] × B[K,N]     → inner dim = B_rows (= K)
  // Transpose: A[M,K] × B[N,K]^T   → inner dim = B_cols (= K)
  //                                    (B^T has shape [K,N])
  // false正常
  const std::size_t K_from_B = transpose_b ? B_cols : B_rows;
  if (K != K_from_B) {
    std::ostringstream oss;
    oss << "matmul: inner dimension mismatch — A is [" << M << ", " << K
        << "], B is [" << B_rows << ", " << B_cols << "]"
        << (transpose_b ? " (transposed)" : "") << "; need K == " << K_from_B;
    throw std::invalid_argument(oss.str());
  }

  // Output columns:
  //   Standard:  B[K,N]   → out_cols = B_cols (= N)
  //   Transpose: B[N,K]^T → out_cols = B_rows (= N)
  const std::size_t out_cols = transpose_b ? B_rows : B_cols;

  // Allocate output tensor, zero-initialised because the i,k,j loop
  // accumulates with +=.
  Tensor C(std::vector<std::size_t>{M, out_cols}, 0.0f);

  // Raw pointers — avoid bounds checks and stride arithmetic in hot loops.
  const float *__restrict__ a_data = A.data();
  const float *__restrict__ b_data = B.data();
  float *__restrict__ c_data = C.data();

  // =======================================================================
  // Standard: C[M][N] += A[M][K] × B[K][N]
  // Loop order i → k → j:
  //   - aik is hoisted out of the j-loop (reused N times)
  //   - B[k*N + j] is contiguous as j increases → L1 cache line hit
  // =======================================================================
  // Cij = Aik * Bkj for k from 0 to K - 1
  if (!transpose_b) {
    for (std::size_t i = 0; i < M; ++i) {
      const std::size_t a_row_off = i * K;
      const std::size_t c_row_off = i * out_cols;
      for (std::size_t k = 0; k < K; ++k) {
        const float aik = a_data[a_row_off + k];
        const std::size_t b_row_off = k * out_cols;
        // Inner loop: walk j contiguously over B[k][*] and C[i][*]
        for (std::size_t j = 0; j < out_cols; ++j) {
          c_data[c_row_off + j] += aik * b_data[b_row_off + j];
        }
      }
    }
  }
  // =======================================================================
  // Transpose B: C[M][N] = A[M][K] × B^T[K][N]  (B literal shape [N, K])
  //   C[i][j] = Σₖ A[i][k] · B[j][k]
  // Loop order i → j → k:
  //   - Innermost k walks both A[i][*] and B[j][*] contiguously
  //     (stride K on B is fine because k increments by 1)
  // =======================================================================
  else {
    // Cij = Aik * Bjk for k from 0 to K-1
    const std::size_t K_b = B_cols; // B is [N, K_b] where K_b == K
    for (std::size_t i = 0; i < M; ++i) {
      const std::size_t a_row_off = i * K;
      const std::size_t c_row_off = i * out_cols;
      for (std::size_t j = 0; j < out_cols; ++j) {
        float sum = 0.0f;
        const std::size_t b_row_off = j * K_b;
        // Inner loop: k walks contiguously over A[i][k] and B[j][k]
        for (std::size_t k = 0; k < K; ++k) {
          sum += a_data[a_row_off + k] * b_data[b_row_off + k];
        }
        c_data[c_row_off + j] = sum;
      }
    }
  }

  return C;
}

} // namespace mt
