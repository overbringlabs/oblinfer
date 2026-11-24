// ingine-hop: EP step 3 probe (plans/20261124_ingine-vs-maya-ds.md §7.3). One Maya-style expert hop
// GPU0 -> pinned host -> GPU1 -> pinned host -> GPU0, 48 hops per token-shaped visit, no peer access
// even when the driver reports it. Measures copy+sync latency, not math.
//
//     ingine-hop [--visits 200] [--seconds 2]
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kHidden         = 2304; // Inkling-2-Mini model width
constexpr int kRows           = 8;    // per-slot expert rows returned per visit
constexpr int kHopsPerVisit   = 48;   // index visits per token
constexpr size_t kSendBytes   = kHidden * sizeof(__nv_bfloat16);            // 4608 B
constexpr size_t kRecvBytes   = kRows * kHidden * sizeof(__nv_bfloat16);    // 36864 B
constexpr int kThreads        = 256;
constexpr double kGateUs      = 40.0;
constexpr int kMinVisits      = 200;
constexpr int kWarmupVisits   = 10;
constexpr int kExtendedChunk  = 16;

void check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) { return; }
    std::fprintf(stderr, "%s:%d: %s failed: %s: %s\n", file, line, expr, cudaGetErrorName(err),
                 cudaGetErrorString(err));
    std::exit(1);
}

#define CHECK(expr) ::check((expr), #expr, __FILE__, __LINE__)

__global__ void write_hidden(__nv_bfloat16* x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kHidden) { x[i] = __float2bfloat16(1.0f); }
}

// GPU1's stub: consume the hidden the hop carried and emit one BF16 row per expert slot.
__global__ void expand_rows(const __nv_bfloat16* hidden, __nv_bfloat16* rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kHidden) { return; }
    const __nv_bfloat16 v = hidden[i];
#pragma unroll
    for (int r = 0; r < kRows; ++r) { rows[r * kHidden + i] = v; }
}

__global__ void consume_rows(const __nv_bfloat16* rows, __nv_bfloat16* sink) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kHidden) { sink[i] = rows[(kRows - 1) * kHidden + i]; }
}

struct Stats {
    double mean = 0;
    double p50  = 0;
    double p99  = 0;
    double min  = 0;
    double max  = 0;
};

Stats summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) { sum += x; }
    Stats s;
    s.mean = sum / static_cast<double>(v.size());
    s.p50  = v[v.size() / 2];
    s.p99  = v[std::min(v.size() - 1, static_cast<std::size_t>(0.99 * v.size()))];
    s.min  = v.front();
    s.max  = v.back();
    return s;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace

int main(int argc, char** argv) {
    int visits      = kMinVisits;
    double seconds  = 2.0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--visits") == 0 && i + 1 < argc) {
            visits = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--seconds") == 0 && i + 1 < argc) {
            seconds = std::atof(argv[++i]);
        } else {
            std::fprintf(stderr, "usage: ingine-hop [--visits N] [--seconds S]\n");
            return 2;
        }
    }
    if (visits < kMinVisits) { visits = kMinVisits; }

    int count = 0;
    CHECK(cudaGetDeviceCount(&count));
    if (count < 2) {
        std::fprintf(stderr, "ingine-hop needs 2 CUDA devices, found %d\n", count);
        return 1;
    }
    cudaDeviceProp p0{}, p1{};
    CHECK(cudaGetDeviceProperties(&p0, 0));
    CHECK(cudaGetDeviceProperties(&p1, 1));

    int peer01 = -1;
    int peer10 = -1;
    CHECK(cudaDeviceCanAccessPeer(&peer01, 0, 1));
    CHECK(cudaDeviceCanAccessPeer(&peer10, 1, 0));
    // Deliberately never cudaDeviceEnablePeerAccess: the hop under test must survive without P2P.

    __nv_bfloat16* d0_send = nullptr;
    __nv_bfloat16* d0_recv = nullptr;
    __nv_bfloat16* d0_sink = nullptr;
    __nv_bfloat16* d1_in   = nullptr;
    __nv_bfloat16* d1_out  = nullptr;
    cudaStream_t s0        = nullptr;
    cudaStream_t s1        = nullptr;

    CHECK(cudaSetDevice(0));
    CHECK(cudaMalloc(&d0_send, kSendBytes));
    CHECK(cudaMalloc(&d0_recv, kRecvBytes));
    CHECK(cudaMalloc(&d0_sink, kSendBytes));
    // Non-blocking so neither stream serialises against the legacy default stream.
    CHECK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    CHECK(cudaSetDevice(1));
    CHECK(cudaMalloc(&d1_in, kSendBytes));
    CHECK(cudaMalloc(&d1_out, kRecvBytes));
    CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));

    void* h_send = nullptr;
    void* h_recv = nullptr;
    CHECK(cudaHostAlloc(&h_send, kSendBytes, cudaHostAllocPortable));
    CHECK(cudaHostAlloc(&h_recv, kRecvBytes, cudaHostAllocPortable));

    // Cross-device dependency events; timing flags off because these are only for stream ordering.
    // Each event belongs to the device of the stream that records it.
    cudaEvent_t sent_ready = nullptr;
    cudaEvent_t back_ready = nullptr;
    CHECK(cudaSetDevice(0));
    CHECK(cudaEventCreateWithFlags(&sent_ready, cudaEventDisableTiming));
    CHECK(cudaSetDevice(1));
    CHECK(cudaEventCreateWithFlags(&back_ready, cudaEventDisableTiming));

    const int hops = visits * kHopsPerVisit;
    // Timing events are all recorded on GPU0's stream, so they must belong to device 0.
    CHECK(cudaSetDevice(0));
    std::vector<cudaEvent_t> hop_start(hops);
    std::vector<cudaEvent_t> hop_end(hops);
    std::vector<cudaEvent_t> visit_start(visits);
    std::vector<cudaEvent_t> visit_end(visits);
    for (cudaEvent_t& e : hop_start) { CHECK(cudaEventCreate(&e)); }
    for (cudaEvent_t& e : hop_end) { CHECK(cudaEventCreate(&e)); }
    for (cudaEvent_t& e : visit_start) { CHECK(cudaEventCreate(&e)); }
    for (cudaEvent_t& e : visit_end) { CHECK(cudaEventCreate(&e)); }

    const int blocks_hidden = (kHidden + kThreads - 1) / kThreads;

    // The hop: GPU0 producer -> DtoH pinned -> HtoD GPU1 -> GPU1 stub -> DtoH pinned -> HtoD GPU0 -> consumer.
    // Streams hand off through events; the host never blocks inside the sequence.
    auto enqueue_hop = [&](int hop, bool timed) {
        if (timed) { CHECK(cudaEventRecord(hop_start[hop], s0)); }
        write_hidden<<<blocks_hidden, kThreads, 0, s0>>>(d0_send);
        CHECK(cudaMemcpyAsync(h_send, d0_send, kSendBytes, cudaMemcpyDeviceToHost, s0));
        CHECK(cudaEventRecord(sent_ready, s0));

        CHECK(cudaStreamWaitEvent(s1, sent_ready, 0));
        CHECK(cudaMemcpyAsync(d1_in, h_send, kSendBytes, cudaMemcpyHostToDevice, s1));
        expand_rows<<<blocks_hidden, kThreads, 0, s1>>>(d1_in, d1_out);
        CHECK(cudaMemcpyAsync(h_recv, d1_out, kRecvBytes, cudaMemcpyDeviceToHost, s1));
        CHECK(cudaEventRecord(back_ready, s1));

        CHECK(cudaStreamWaitEvent(s0, back_ready, 0));
        CHECK(cudaMemcpyAsync(d0_recv, h_recv, kRecvBytes, cudaMemcpyHostToDevice, s0));
        consume_rows<<<blocks_hidden, kThreads, 0, s0>>>(d0_recv, d0_sink);
        if (timed) { CHECK(cudaEventRecord(hop_end[hop], s0)); }
    };

    auto enqueue_visit = [&](int visit, bool timed) {
        if (timed) { CHECK(cudaEventRecord(visit_start[visit], s0)); }
        for (int h = 0; h < kHopsPerVisit; ++h) {
            enqueue_hop(visit * kHopsPerVisit + h, timed);
        }
        if (timed) { CHECK(cudaEventRecord(visit_end[visit], s0)); }
        CHECK(cudaGetLastError());
    };

    for (int v = 0; v < kWarmupVisits; ++v) { enqueue_visit(v, false); }
    CHECK(cudaStreamSynchronize(s0));

    // One check that the round trip moved the payload: every row of the last hop is bf16 1.0 (0x3F80).
    const auto* recv_words = static_cast<const unsigned short*>(h_recv);
    bool payload_ok        = true;
    for (std::size_t i = 0; i < kRecvBytes / sizeof(unsigned short); ++i) {
        payload_ok = payload_ok && recv_words[i] == 0x3F80u;
    }

    const auto timed_begin = std::chrono::steady_clock::now();
    for (int v = 0; v < visits; ++v) { enqueue_visit(v, true); }
    CHECK(cudaStreamSynchronize(s0));
    const double timed_wall = seconds_since(timed_begin);

    std::vector<double> hop_us(hops);
    std::vector<double> visit_us(visits);
    for (int i = 0; i < hops; ++i) {
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, hop_start[i], hop_end[i]));
        hop_us[i] = 1000.0 * ms;
    }
    for (int v = 0; v < visits; ++v) {
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, visit_start[v], visit_end[v]));
        visit_us[v] = 1000.0 * ms;
    }

    for (cudaEvent_t& e : hop_start) { CHECK(cudaEventDestroy(e)); }
    for (cudaEvent_t& e : hop_end) { CHECK(cudaEventDestroy(e)); }
    for (cudaEvent_t& e : visit_start) { CHECK(cudaEventDestroy(e)); }
    for (cudaEvent_t& e : visit_end) { CHECK(cudaEventDestroy(e)); }

    // Whichever is larger: kMinVisits tokens or `seconds` of wall clock.
    long long extra_visits = 0;
    double extra_seconds   = 0;
    if (timed_wall < seconds) {
        const double target = seconds - timed_wall;
        const auto extra_begin = std::chrono::steady_clock::now();
        while (seconds_since(extra_begin) < target) {
            for (int v = 0; v < kExtendedChunk; ++v) { enqueue_visit(0, false); }
            extra_visits += kExtendedChunk;
        }
        CHECK(cudaStreamSynchronize(s0));
        extra_seconds = seconds_since(extra_begin);
    }

    const Stats hop   = summarize(hop_us);
    const Stats visit = summarize(visit_us);
    const double token_us = hop.mean * kHopsPerVisit;

    std::printf("ingine-hop: EP step 3 hop probe, pinned host staging, no peer access\n");
    std::printf("GPU0  %s  CC %d.%d  SMs %d  CUDA runtime %d.%d\n", p0.name, p0.major, p0.minor,
                p0.multiProcessorCount, CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
    std::printf("GPU1  %s  CC %d.%d  SMs %d\n", p1.name, p1.major, p1.minor,
                p1.multiProcessorCount);
    std::printf("P2P   0->1 %d  1->0 %d  (cudaDeviceCanAccessPeer; never enabled)\n", peer01,
                peer10);
    std::printf("payload  send %zu B, recv %zu B per hop; %d hops per visit\n", kSendBytes,
                kRecvBytes, kHopsPerVisit);
    std::printf("payload check  recv rows all bf16 1.0: %s\n", payload_ok ? "yes" : "NO");
    std::printf("timed  %d visits, %d hops, %.3f s wall, %.1f visits/s\n", visits, hops, timed_wall,
                visits / timed_wall);
    std::printf("hop    mean %.2f us  p50 %.2f  p99 %.2f  min %.2f  max %.2f\n", hop.mean,
                hop.p50, hop.p99, hop.min, hop.max);
    std::printf("visit  mean %.2f us  p50 %.2f  p99 %.2f  (48 hops)\n", visit.mean, visit.p50,
                visit.p99);
    std::printf("token  %.2f us (48 x hop mean)  %.1f tok/s from hop mean, %.1f tok/s timed wall\n",
                token_us, 1e6 / token_us, visits / timed_wall);
    if (extra_visits > 0) {
        std::printf("extra  %lld visits in %.3f s -> %.1f tok/s (untimed stream, 16-visit chunks)\n",
                    extra_visits, extra_seconds, extra_visits / extra_seconds);
    }
    std::printf("gate   %.0f us/hop: %s (hop mean %.2f us, token %.2f us)\n", kGateUs,
                hop.mean <= kGateUs ? "PASS" : "FAIL", hop.mean, token_us);

    CHECK(cudaSetDevice(0));
    CHECK(cudaStreamDestroy(s0));
    CHECK(cudaEventDestroy(sent_ready));
    CHECK(cudaFree(d0_send));
    CHECK(cudaFree(d0_recv));
    CHECK(cudaFree(d0_sink));
    CHECK(cudaFreeHost(h_send));
    CHECK(cudaFreeHost(h_recv));
    CHECK(cudaSetDevice(1));
    CHECK(cudaStreamDestroy(s1));
    CHECK(cudaEventDestroy(back_ready));
    CHECK(cudaFree(d1_in));
    CHECK(cudaFree(d1_out));
    return 0;
}
