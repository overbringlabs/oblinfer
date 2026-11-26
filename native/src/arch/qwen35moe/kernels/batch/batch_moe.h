// Verify-width routed experts: n tokens, 8n (token, expert) pairs, each expert's weights read once per
// layer instead of once per token that selects it.
//
// Neighbouring tokens in a draft chain keep most of their router experts, so the union U of the 8n ids is
// well below 8n. The path therefore has three stages:
//
//   1. dedup  — compact the union of ids on the device and record, for every pair, which unique expert and
//      which row of that expert's row group it owns (and, per unique expert, the pair list).
//   2. gate/up and down — for one unique expert, one warp reads a weight row once and dots it against all
//      rows of the group, so the expert's bytes move once no matter how many tokens asked for it. Scratch
//      stays pair-major, which keeps the silu step and the per-token reduce free of any indirection.
//   3. reduce — y[t] = sum_k w[t][k] * down_k h[t][k], the same weighted sum the single-token kernel folds
//      in registers, here over the pair scratch.
//
// Bytes per layer drop from 8n * 1.77 MB to U * 1.77 MB. The FLOP count does not change (it is 8n rows of
// the same shape either way), which is exactly the trade this file is for: a decode step is weight-bound, so
// turning bytes into arithmetic is the win.
//
// Contract, as in the rest of the pack: nothing is allocated or synchronised, the ids and weights are read on
// the device (graph-capturable), `y` is overwritten, and an expert whose `stride_bytes` is 0 lives on another
// card and contributes zeros, so both cards can run the same plan and the partials are summed afterwards.
// The plan buffers are sized for BATCH_MAX * MOE_TOP_K pairs, which bounds the grid without a host-visible
// unique count.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "arch/qwen35moe/kernels/moe/moe_apply.h"
#include "arch/qwen35moe/kernels/batch/batch_gemv.h"

namespace ingine::arch::qwen35moe::kernels {

inline constexpr int BATCH_PAIRS = BATCH_MAX * MOE_TOP_K;  // 32 at BATCH_MAX = 4

// Where the dedup put each pair. All arrays are device memory, sized for the worst case.
struct BatchMoePlan {
  int* unique_ids = nullptr;   // [BATCH_PAIRS] expert ids, first-seen order
  int* unique_count = nullptr;  // one int
  int* pair_unique = nullptr;  // [BATCH_PAIRS] -> index into unique_ids
  int* pair_of = nullptr;      // [BATCH_PAIRS * BATCH_MAX] -> pair index of row j of unique expert u, -1 if none
  int* rows = nullptr;         // [BATCH_PAIRS] -> rows held by unique expert u
};

// One expert inside a packed block. Same addressing as kernels/moe/fused_experts.h's FusedExpertLayer
// (`artifact = base + id * stride + offset`), with 64-bit offsets so a shared base can span a >2 GB arena.
struct BatchExpertBlock {
  const std::uint8_t* w_base = nullptr;  // start of this layer's packed expert weights
  const std::uint8_t* s_base = nullptr;  // start of this layer's packed block scales
  std::int64_t stride_bytes = 0;         // bytes per expert in w_base (0: expert lives on another card)
  std::int64_t scale_stride_bytes = 0;
  std::int64_t gate_w = 0, up_w = 0, down_w = 0;
  std::int64_t gate_s = 0, up_s = 0, down_s = 0;
  float gate_scale2 = 1.0f, up_scale2 = 1.0f, down_scale2 = 1.0f;
};

struct BatchMoeArgs {
  const BatchExpertBlock* layers = nullptr;  // [EXPERTS], indexed by expert id
  const float* x = nullptr;                  // [n, MOE_HIDDEN]
  const int* ids = nullptr;                  // [n, MOE_TOP_K]
  const float* weights = nullptr;            // [n, MOE_TOP_K], already normalised
  float* y = nullptr;                        // [n, MOE_HIDDEN]
  float* scratch_gate = nullptr;             // [BATCH_PAIRS, MOE_INTER] -> h after the silu step
  float* scratch_up = nullptr;               // [BATCH_PAIRS, MOE_INTER]
  float* scratch_down = nullptr;             // [BATCH_PAIRS, MOE_HIDDEN]
  BatchMoePlan plan;
  int n = 0;                                 // 1..BATCH_MAX
};

// Four launches: dedup, gate+up, silu, down, then the per-token reduce (five in total).
cudaError_t batch_moe_apply(const BatchMoeArgs& args, cudaStream_t stream);

// The same plan the device builds, computed on the host: first-seen order over the n * MOE_TOP_K pairs, with
// each pair mapped to (unique expert, row inside that expert's group). Tests use it to check the device
// dedup and as the byte count for the cost model.
struct BatchMoePlanHost {
  int unique_count = 0;
  int unique_ids[BATCH_PAIRS];
  int pair_unique[BATCH_PAIRS];
  int pair_row[BATCH_PAIRS];
  int rows[BATCH_PAIRS];
  int pair_of[BATCH_PAIRS * BATCH_MAX];
};

void batch_moe_plan_cpu(const int* ids, int n, BatchMoePlanHost* out);

// FP64 reference over the same bytes: per token, per slot, the full expert GEMV. `y` is overwritten.
void batch_moe_apply_cpu(const BatchMoeArgs& args, const BatchMoePlanHost& host);

// Bytes the batched path reads for one layer, from a host-side plan: the reference for the cost model.
inline std::int64_t batch_moe_weight_bytes(const BatchMoePlanHost& host) {
  return std::int64_t(host.unique_count) * (MOE_INTER * MOE_HIDDEN / 2 * 2 + MOE_HIDDEN * MOE_INTER / 2 +
                                            (MOE_INTER * MOE_HIDDEN / 16) * 3);
}

// ---- router and shared expert at verify width ----

// n tokens, one routing pass. The logits kernel is the single-token kernel's shape — one warp per expert,
// lane-stride-32 column order — with one accumulator per token, so token t's logits are bit-identical to
// kernels/moe/router.cu's and its ids and weights come out the same. The selection is the same rank sweep, one
// block per token, so ties still resolve to the lower expert id and the eight weights are the same softmax.
struct BatchRouterArgs {
  const __nv_bfloat16* weight = nullptr;  // [ROUTER_EXPERTS, ROUTER_HIDDEN], mlp.gate.weight
  const float* x = nullptr;               // [n, ROUTER_HIDDEN]
  int* ids = nullptr;                     // [n, ROUTER_TOP_K], descending logit
  float* weights = nullptr;               // [n, ROUTER_TOP_K], softmax over the selected logits
  float* logits = nullptr;                // [n, ROUTER_EXPERTS] scratch
  int n = 0;
};
cudaError_t route_rows(const BatchRouterArgs& args, cudaStream_t stream);

// y[t] = sigmoid(<gate_weight, x[t]>) * y[t], the shared expert's bf16 scalar gate at verify width. One block
// per token, same dot order as kernels/moe/shared_expert.cu's scale_sigmoid_dot.
cudaError_t scale_sigmoid_rows(float* y, const __nv_bfloat16* gate_weight, const float* x, int rows, int width,
                               cudaStream_t stream);

}  // namespace ingine::arch::qwen35moe::kernels
