#include "mlp.h"

#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>

namespace mt {

// ===========================================================================
// SiLU
// ===========================================================================

auto silu(const Tensor &x) -> Tensor {
  Tensor out = x; // copy, then transform in-place
                  // 循环可自动向量化加速
  float *__restrict__ d = out.data();
  const std::size_t n = out.size();

  for (std::size_t i = 0; i < n; ++i) {
    const float v = d[i];
    // SiLU(v) = v * sigmoid(v) = v / (1 + exp(-v))
    // For v < -10, sigmoid ≈ 0 → SiLU ≈ 0  (avoid exp overflow)
    // For v >  10, sigmoid ≈ 1 → SiLU ≈ v
    if (v < -10.0f) {
      d[i] = 0.0f;
    } else if (v > 10.0f) {
      // SiLU ≈ v; compute via v / (1+exp(-v)) is stable here too
      d[i] = v / (1.0f + std::exp(-v));
    } else {
      d[i] = v / (1.0f + std::exp(-v));
    }
  }
  return out;
}

// ===========================================================================
// Construction
// ===========================================================================

MLP::MLP(const std::size_t d_model, std::size_t d_hidden)
    : d_model_(d_model), d_hidden_(d_hidden == 0 ? d_model * 8 / 3 : d_hidden),
      w_gate_(d_model_, d_hidden_, /*has_bias=*/false),
      w_up_(d_model_, d_hidden_, /*has_bias=*/false),
      w_down_(d_hidden_, d_model_, /*has_bias=*/false) {}

// ===========================================================================
// Forward
// ===========================================================================

auto MLP::forward(const Tensor &X) const -> Tensor {

  if (X.ndim() != 2) {
    std::ostringstream oss;
    oss << "MLP::forward: input must be 2-D, got " << X.ndim() << "-D";
    throw std::invalid_argument(oss.str());
  }
  if (X.shape()[1] != d_model_) {
    std::ostringstream oss;
    oss << "MLP::forward: input last dim " << X.shape()[1] << " != d_model "
        << d_model_;
    throw std::invalid_argument(oss.str());
  }
  // 投影 + silu激活
  //  gate branch:  [seq, d_hidden_]
  Tensor gate = silu(w_gate_.forward(X));
  // 单纯线性投影
  //  up branch:    [seq, d_hidden_]
  Tensor up = w_up_.forward(X);

  // Element-wise product (Hadamard)
  float *__restrict__ g = gate.data();
  const float *__restrict__ u = up.data();
  const std::size_t n = gate.size();
  for (std::size_t i = 0; i < n; ++i) {
    g[i] *= u[i]; // gate = gate ⊙ up  (reuse gate buffer)
  }

  // Down projection: [seq, d_hidden_] → [seq, d_model_]
  return w_down_.forward(gate);
}

// ===========================================================================
// Accessors
// ===========================================================================

auto MLP::w_gate() -> Linear & { return w_gate_; }
auto MLP::w_up() -> Linear & { return w_up_; }
auto MLP::w_down() -> Linear & { return w_down_; }

auto MLP::w_gate() const -> const Linear & { return w_gate_; }
auto MLP::w_up() const -> const Linear & { return w_up_; }
auto MLP::w_down() const -> const Linear & { return w_down_; }

auto MLP::d_model() const noexcept -> std::size_t { return d_model_; }
auto MLP::d_hidden() const noexcept -> std::size_t { return d_hidden_; }

} // namespace mt
