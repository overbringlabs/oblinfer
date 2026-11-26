// Per-token CUDA graph for the Qwen3.6-35B-A3B decode step.
//
// One token is a few hundred small launches across two cards, so at this size the launch cost, not the math,
// sets the floor. Capturing the step collapses them into one replay. A graph is captured per `splits` bucket:
// `splits` is the flash-decode width the full-attention layers derive from the position, and inside a bucket
// every position issues the same kernels with the same grids, so one graph serves the whole range.
//
// Two values change per token and neither may be baked into a node:
//   - the token id reaches the embedding through a device int the graph reads (embedding_row_dev below), and
//   - the position reaches the rope and attention kernels through their captured parameters. Both take their
//     argument struct by value, so a captured node holds the position that was current at capture time and
//     decode_graph_run rewrites it in place before each replay. That is why the capture registers the nodes it
//     needs patched instead of the kernels reading pos_dev: the pack's kernels have no pointer to read.
// The host readback at the end of a step (argmax, logits) stays outside the graph.
//
// The second card's experts are captured as their own graph per layer and spliced into the step graph as a
// child node. An event pair split across two graph launches does not order them: a wait node binds to the most
// recent record when it runs, so each graph would consume the other's previous step (measured in
// knowledge/lean-agent-flow/lessons_learned/20261125_cross-device-graph-events.md; reproduced here on sm_89,
// CUDA 13.4). The child node's edges are the cross-device ordering, and the same holds for the pinned staging
// the hop runs through. The events stay in the eager path only.
//
// Capturing on the legacy default stream is not supported, so a capture runs on a named non-blocking stream.
// Measured on sm_89/CUDA 13.4: a synchronous cudaMemset or cudaMemcpy inside a capture issues no node and
// reports no error, so everything inside the step must be stream-ordered on the capture stream.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ingine::arch::qwen35moe {

class Decoder;

// out[i] = float(table[*row * n + i]). dense::embedding_row takes the row on the host, which a captured graph
// cannot vary; this reads it from device memory with the same blocking and the same elementwise conversion, so
// the FP32 output is identical bit for bit.
cudaError_t embedding_row_dev(const __nv_bfloat16* table, const int* row, float* out, int n, cudaStream_t stream);

// Puts both step inputs in device memory in one launch. Called outside the graph, before the replay, so the
// values are kernel arguments rather than host memory the graph would have to alias.
cudaError_t set_step_inputs(int token, int pos, int* token_dev, int* pos_dev, cudaStream_t stream);

// One captured kernel node whose by-value argument holds the position.
struct DecodeGraphPosNode {
  cudaGraphNode_t node = nullptr;
  cudaKernelNodeParams params{};  // func, grid and block as captured; kernelParams is left null
  std::vector<char> bytes;        // the captured argument, with the position rewritten in place
  std::size_t offset = 0;         // where the position sits inside bytes
};

// Collects, while a capture is open, what a replay needs to vary. Every method throws std::runtime_error with
// the reason when the capture does not look like the step it was opened for: an unverified graph is worse than
// no graph, and a failure here is the caller's signal to run eagerly.
class DecodeGraphCapture {
 public:
  // Call around the launch of a kernel whose by-value argument `args` carries the position at `pos_offset`.
  // begin() notes the node count, end() takes the nodes the launch appended and checks each one against
  // `args`, so a kernel that dropped out of the capture cannot pass unnoticed.
  void begin(cudaStream_t stream);
  void end(cudaStream_t stream, const void* args, std::size_t size, std::size_t pos_offset);

  // The second device's half of one MoE layer, spliced into the capture in progress. splice() forks it off the
  // current tail; join() makes the node enqueued next depend on it as well, which keeps the device-0 experts
  // running beside the hop exactly as the eager event pair does.
  void splice(cudaStream_t stream, cudaGraph_t child);
  void join(cudaStream_t stream);

  std::vector<DecodeGraphPosNode> take_pos_nodes() { return std::move(pos_nodes_); }
  int hops() const { return hops_; }

 private:
  std::size_t window_ = 0;
  bool open_ = false;
  int hops_ = 0;
  cudaGraphNode_t hop_ = nullptr;
  std::vector<DecodeGraphPosNode> pos_nodes_;
};

// What ended up inside a captured step, for the log line the decoder prints once per bucket.
struct DecodeGraphStats {
  int nodes = 0;
  int kernels = 0;
  int copies = 0;
  int memsets = 0;
  int hops = 0;
  int pos_nodes = 0;
};

// One captured decode step: the executable, the two device ints of the step inputs, and the nodes the replay
// patches.
struct DecodeGraph {
  cudaGraphExec_t exec = nullptr;
  int splits = 0;
  std::uint64_t token_dev = 0;  // device int32 the captured embedding node reads
  std::uint64_t pos_dev = 0;    // device int32 of the position, written in the same launch as token_dev; no
                                // captured node reads it, since the pack's kernels take pos by value
  cudaStream_t stream = nullptr;
  // cudaGraphExecKernelNodeSetParams names nodes of the graph the executable was instantiated from, so that
  // definition has to outlive every replay.
  cudaGraph_t graph = nullptr;
  std::vector<DecodeGraphPosNode> pos_nodes;
  DecodeGraphStats stats;
};

// Captures one token of `dec`'s work at a position inside the `splits` bucket. Returns false with the reason
// when any part of the step cannot be captured; the caller then keeps stepping eagerly.
bool decode_graph_build(DecodeGraph* g, Decoder& dec, int splits, std::string* error);

// Writes the step inputs, repoints every captured position at `pos` and replays the step. Enqueue-only: the
// caller reads the logits back afterwards. Rewriting a live executable is not defined, so the caller must have
// drained the previous replay on `g->stream` before this call.
bool decode_graph_run(DecodeGraph* g, int token, int pos, std::string* error);

void decode_graph_free(DecodeGraph* g);

}  // namespace ingine::arch::qwen35moe
