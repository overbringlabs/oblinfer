#include "kernels/gdn/gates.h"

#include <cmath>

namespace ingine::kernels::gdn {
namespace {

__global__ void gdn_gates_kernel(GdnGateArgs args) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= args.heads) return;
  const float b = args.b[h];
  args.beta[h] = 1.0f / (1.0f + __expf(-b));
  const float x = args.a[h] + args.dt_bias[h];
  const float softplus = x > 20.0f ? x : __logf(1.0f + __expf(x));
  args.g[h] = -__expf(args.a_log[h]) * softplus;
}

}  // namespace

cudaError_t gdn_gates(const GdnGateArgs& args, cudaStream_t stream) {
  if (!args.a || !args.b || !args.dt_bias || !args.a_log || !args.beta || !args.g || args.heads <= 0) {
    return cudaErrorInvalidValue;
  }
  gdn_gates_kernel<<<(args.heads + 127) / 128, 128, 0, stream>>>(args);
  return cudaGetLastError();
}

void gdn_gates_cpu(const GdnGateArgs& args) {
  for (int h = 0; h < args.heads; ++h) {
    args.beta[h] = float(1.0 / (1.0 + std::exp(-double(args.b[h]))));
    const double x = double(args.a[h]) + double(args.dt_bias[h]);
    const double softplus = x > 20.0 ? x : std::log1p(std::exp(x));
    args.g[h] = float(-std::exp(double(args.a_log[h])) * softplus);
  }
}

}  // namespace ingine::kernels::gdn
