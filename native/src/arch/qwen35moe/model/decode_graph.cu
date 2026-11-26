#include "arch/qwen35moe/model/decode_graph.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "arch/qwen35moe/kernels/attn/decode_attn.h"
#include "arch/qwen35moe/model/decoder.h"

namespace ingine::arch::qwen35moe {
namespace {

namespace attn = ingine::arch::qwen35moe::kernels;

constexpr int kEmbedThreads = 256;  // dense_ops.cu's kThreads: same blocking, same result

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

std::uint64_t as_handle(void* p) { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(p)); }

void* as_pointer(std::uint64_t h) { return reinterpret_cast<void*>(static_cast<std::uintptr_t>(h)); }

__global__ void embedding_row_dev_kernel(const __nv_bfloat16* table, const int* row, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = __bfloat162float(table[std::size_t(row[0]) * n + i]);
}

__global__ void set_step_inputs_kernel(int token, int pos, int* token_dev, int* pos_dev) {
  *token_dev = token;
  *pos_dev = pos;
}

// The graph a stream is capturing into, plus how many nodes it holds so far. Captured nodes keep the order the
// launches made, so the range a launch produced is a contiguous window at the end.
cudaGraph_t capturing_graph(cudaStream_t stream, std::size_t* nodes) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  cudaGraph_t graph = nullptr;
  const cudaGraphNode_t* deps = nullptr;
  const cudaGraphEdgeData* edges = nullptr;
  std::size_t n = 0;
  check(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph, &deps, &edges, &n), "capture state");
  if (status != cudaStreamCaptureStatusActive) throw std::runtime_error("the step stream is not capturing");
  if (nodes) {
    std::size_t count = 0;
    check(cudaGraphGetNodes(graph, nullptr, &count), "captured node count");
    *nodes = count;
  }
  return graph;
}

}  // namespace

cudaError_t embedding_row_dev(const __nv_bfloat16* table, const int* row, float* out, int n, cudaStream_t stream) {
  if (!table || !row || !out || n <= 0) return cudaErrorInvalidValue;
  embedding_row_dev_kernel<<<(n + kEmbedThreads - 1) / kEmbedThreads, kEmbedThreads, 0, stream>>>(table, row, out,
                                                                                                  n);
  return cudaGetLastError();
}

cudaError_t set_step_inputs(int token, int pos, int* token_dev, int* pos_dev, cudaStream_t stream) {
  if (!token_dev || !pos_dev) return cudaErrorInvalidValue;
  set_step_inputs_kernel<<<1, 1, 0, stream>>>(token, pos, token_dev, pos_dev);
  return cudaGetLastError();
}

void DecodeGraphCapture::begin(cudaStream_t stream) {
  if (open_) throw std::runtime_error("a capture window is already open");
  capturing_graph(stream, &window_);
  open_ = true;
}

void DecodeGraphCapture::end(cudaStream_t stream, const void* args, std::size_t size, std::size_t pos_offset) {
  if (!open_) throw std::runtime_error("no capture window is open");
  open_ = false;
  std::size_t after = 0;
  cudaGraph_t graph = capturing_graph(stream, &after);
  if (after <= window_) throw std::runtime_error("the position-bearing launch added no node to the capture");
  std::vector<cudaGraphNode_t> nodes(after);
  check(cudaGraphGetNodes(graph, nodes.data(), &after), "captured nodes");
  // Both argument structs open with a pointer member and carry `pos` as an int inside the recorded size, so
  // these two reads identify the node without touching padding.
  const void* want_key = *static_cast<const void* const*>(args);
  const int want_pos = *reinterpret_cast<const int*>(static_cast<const char*>(args) + pos_offset);
  for (std::size_t i = window_; i < after; ++i) {
    cudaGraphNodeType type = cudaGraphNodeTypeKernel;
    check(cudaGraphNodeGetType(nodes[i], &type), "captured node type");
    if (type != cudaGraphNodeTypeKernel)
      throw std::runtime_error("a position-bearing launch captured a node that is not a kernel");
    cudaKernelNodeParams p{};
    check(cudaGraphKernelNodeGetParams(nodes[i], &p), "captured node params");
    if (!p.func || !p.kernelParams || !p.kernelParams[0])
      throw std::runtime_error("a captured kernel node exposes no argument buffer");
    const void* node_key = *static_cast<const void* const*>(p.kernelParams[0]);
    int node_pos = 0;
    std::memcpy(&node_pos, static_cast<const char*>(p.kernelParams[0]) + pos_offset, sizeof(node_pos));
    if (node_key != want_key || node_pos != want_pos)
      throw std::runtime_error("a captured node does not hold the position argument it was recorded with");
    // The rope and attention launchers pass their args struct as the kernel's only parameter (norm_rope_kernel
    // in rope_norm.cu; kv_store_*, decode_kernel and decode_split_kernel in decode_attn.cu), so one
    // kernelParams entry is the whole struct. decode_graph_run rebuilds that entry per replay.
    DecodeGraphPosNode pn;
    pn.node = nodes[i];
    pn.params = p;
    pn.params.kernelParams = nullptr;  // the node's own array is not ours to keep
    pn.params.extra = nullptr;
    pn.bytes.assign(static_cast<const char*>(p.kernelParams[0]),
                    static_cast<const char*>(p.kernelParams[0]) + size);
    pn.offset = pos_offset;
    pos_nodes_.push_back(std::move(pn));
  }
}

void DecodeGraphCapture::splice(cudaStream_t stream, cudaGraph_t child) {
  if (!child) throw std::runtime_error("no second-device graph to splice into the step");
  if (hop_) throw std::runtime_error("a second-device graph is already waiting to be joined");
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  cudaGraph_t graph = nullptr;
  const cudaGraphNode_t* deps = nullptr;
  const cudaGraphEdgeData* edges = nullptr;
  std::size_t n = 0;
  check(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph, &deps, &edges, &n), "capture dependencies");
  if (status != cudaStreamCaptureStatusActive) throw std::runtime_error("the step stream is not capturing");
  cudaGraphNode_t node = nullptr;
  check(cudaGraphAddChildGraphNode(&node, graph, deps, n, child), "splice the second-device graph");
  hop_ = node;
  ++hops_;
}

void DecodeGraphCapture::join(cudaStream_t stream) {
  if (!hop_) throw std::runtime_error("join with no second-device graph spliced");
  check(cudaStreamUpdateCaptureDependencies(stream, &hop_, nullptr, 1, cudaStreamAddCaptureDependencies),
        "join the second-device graph");
  hop_ = nullptr;
}

void decode_graph_free(DecodeGraph* g) {
  if (!g) return;
  if (g->exec) cudaGraphExecDestroy(g->exec);
  if (g->graph) cudaGraphDestroy(g->graph);
  g->exec = nullptr;
  g->graph = nullptr;
  g->pos_nodes.clear();
  g->stats = {};
  g->splits = 0;
  g->stream = nullptr;
}

bool decode_graph_build(DecodeGraph* g, Decoder& dec, int splits, std::string* error) {
  if (!g) return false;
  decode_graph_free(g);
  try {
    if (splits < 1 || splits > attn::kMaxSplits) throw std::runtime_error("attention split bucket out of range");
    const int pos = dec.capture_pos(splits);
    if (pos < 0 || pos >= dec.max_pos()) throw std::runtime_error("no capture position inside the bucket");
    if (!dec.prepare_graph(error)) return false;
    g->splits = splits;
    g->stream = dec.graph_stream();
    g->token_dev = as_handle(dec.token_dev());
    g->pos_dev = as_handle(dec.pos_dev());

    DecodeGraphCapture cap;
    check(cudaStreamBeginCapture(g->stream, cudaStreamCaptureModeThreadLocal), "begin the step capture");
    cudaGraph_t graph = nullptr;
    try {
      dec.enqueue_graph(pos, &cap);
    } catch (...) {
      if (cudaStreamEndCapture(g->stream, &graph) == cudaSuccess && graph) cudaGraphDestroy(graph);
      throw;
    }
    check(cudaStreamEndCapture(g->stream, &graph), "end the step capture");
    g->graph = graph;
    check(cudaGraphInstantiate(&g->exec, graph, 0), "instantiate the step graph");

    std::size_t n = 0;
    check(cudaGraphGetNodes(graph, nullptr, &n), "step graph nodes");
    g->stats.nodes = int(n);
    std::vector<cudaGraphNode_t> nodes(n);
    check(cudaGraphGetNodes(graph, nodes.data(), &n), "step graph node list");
    for (cudaGraphNode_t node : nodes) {
      cudaGraphNodeType type = cudaGraphNodeTypeKernel;
      check(cudaGraphNodeGetType(node, &type), "step graph node type");
      if (type == cudaGraphNodeTypeKernel) ++g->stats.kernels;
      else if (type == cudaGraphNodeTypeMemcpy) ++g->stats.copies;
      else if (type == cudaGraphNodeTypeMemset) ++g->stats.memsets;
      // A spliced second-device graph lands as a node that executes an embedded graph.
      else if (type == cudaGraphNodeTypeGraph) ++g->stats.hops;
    }
    // The recorder says how many hops it spliced; the graph says how many landed. A difference is dropped work.
    if (g->stats.hops != cap.hops())
      throw std::runtime_error("the captured step holds " + std::to_string(g->stats.hops) + " of " +
                               std::to_string(cap.hops()) + " second-device hops");
    g->pos_nodes = cap.take_pos_nodes();
    g->stats.pos_nodes = int(g->pos_nodes.size());
    if (g->pos_nodes.empty()) throw std::runtime_error("the captured step holds no position-bearing node");
    check(cudaGraphUpload(g->exec, g->stream), "upload the step graph");
    return true;
  } catch (const std::exception& e) {
    decode_graph_free(g);
    if (error) *error = e.what();
    return false;
  }
}

bool decode_graph_run(DecodeGraph* g, int token, int pos, std::string* error) {
  const auto fail = [&](const std::string& what) {
    if (error) *error = what;
    return false;
  };
  if (!g || !g->exec) return fail("no captured step for this bucket");
  for (auto& node : g->pos_nodes) {
    if (node.offset + sizeof(pos) > node.bytes.size())
      return fail("the captured argument is shorter than its position field");
    std::memcpy(node.bytes.data() + node.offset, &pos, sizeof(pos));
    void* kp[1] = {node.bytes.data()};  // the array only has to outlive this call
    cudaKernelNodeParams p = node.params;
    p.kernelParams = kp;
    p.extra = nullptr;
    const cudaError_t err = cudaGraphExecKernelNodeSetParams(g->exec, node.node, &p);
    if (err != cudaSuccess) return fail(std::string("rewrite the captured position: ") + cudaGetErrorString(err));
  }
  const cudaError_t in = set_step_inputs(token, pos, static_cast<int*>(as_pointer(g->token_dev)),
                  static_cast<int*>(as_pointer(g->pos_dev)), g->stream);
  if (in != cudaSuccess) return fail(std::string("step inputs: ") + cudaGetErrorString(in));
  const cudaError_t launch = cudaGraphLaunch(g->exec, g->stream);
  if (launch != cudaSuccess) return fail(std::string("replay the step graph: ") + cudaGetErrorString(launch));
  return true;
}

}  // namespace ingine::arch::qwen35moe
