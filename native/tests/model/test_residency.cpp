// Expert residency across two devices (EP step 2): the per-layer split table of RuntimeOptions places a
// layer's four expert tensors (and their factors) whole on the second device, and the default table leaves
// today's one-device load untouched. Per moved layer: the second device holds the checkpoint's bytes, the
// primary holds none of them, each device's free memory moved by its share, and a plain one-device
// load_layer of the same layer holds the same bytes. No decode: step 2 is load plumbing.
// Two GPUs and INGINE_RESIDENCY_TEST=1 (it loads the whole model, ~16.5 GB across both cards); skips (77).
#include "common/test_util.h"

#include "core/device.h"
#include "core/safetensors.h"
#include "model/runtime.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace ingine;
using namespace ingine::model;

namespace {

constexpr int kFirstShardLayer = 19;  // 19 is NVFP4 (its factors move too), 20-23 are MXFP8

std::size_t free_bytes(int device) {
  int saved = 0;
  INGINE_TEST_CUDA(cudaGetDevice(&saved));
  INGINE_TEST_CUDA(cudaSetDevice(device));
  std::size_t free = 0, total = 0;
  INGINE_TEST_CUDA(cudaMemGetInfo(&free, &total));
  INGINE_TEST_CUDA(cudaSetDevice(saved));
  return free;
}

// Allocates and frees before a measurement, so lazy context growth is not charged to the load.
void warm_up(int device) {
  int saved = 0;
  INGINE_TEST_CUDA(cudaGetDevice(&saved));
  INGINE_TEST_CUDA(cudaSetDevice(device));
  void* p = nullptr;
  INGINE_TEST_CUDA(cudaMalloc(&p, std::size_t(256) << 20));
  INGINE_TEST_CUDA(cudaFree(p));
  INGINE_TEST_CUDA(cudaSetDevice(saved));
}

int device_of(const void* p) {
  cudaPointerAttributes attr{};
  INGINE_TEST_CUDA(cudaPointerGetAttributes(&attr, p));
  return attr.device;
}

std::vector<std::byte> download(const Tensor& t) {
  int saved = 0;
  INGINE_TEST_CUDA(cudaGetDevice(&saved));
  INGINE_TEST_CUDA(cudaSetDevice(device_of(t.data)));
  std::vector<std::byte> host(t.bytes());
  INGINE_TEST_CUDA(cudaMemcpy(host.data(), t.data, t.bytes(), cudaMemcpyDeviceToHost));
  INGINE_TEST_CUDA(cudaSetDevice(saved));
  return host;
}

}  // namespace

int main() {
  if (ingine::test::skip_without_gpu()) return ingine::test::kSkip;
  int devices = 0;
  INGINE_TEST_CUDA(cudaGetDeviceCount(&devices));
  if (devices < 2) {
    std::printf("skip: needs two CUDA devices, found %d\n", devices);
    return ingine::test::kSkip;
  }
  if (std::getenv("INGINE_RESIDENCY_TEST") == nullptr) {
    std::printf("skip: set INGINE_RESIDENCY_TEST=1 (loads the whole model, ~16.5 GB across both cards)\n");
    return ingine::test::kSkip;
  }
  const std::filesystem::path dir = INGINE_MODEL_DIR;
  if (!std::filesystem::exists(dir / "model.safetensors.index.json")) {
    std::printf("skip: no model at %s\n", INGINE_MODEL_DIR);
    return ingine::test::kSkip;
  }

  const int primary = 0, peer = 1;
  INGINE_TEST_CUDA(cudaSetDevice(primary));
  cudaStream_t stream = nullptr;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceContext second(peer);
  INGINE_TEST_CUDA(cudaSetDevice(primary));  // a DeviceContext leaves its own device current

  warm_up(primary);
  warm_up(peer);
  const std::size_t free0 = free_bytes(primary), free1 = free_bytes(peer);
  if (free0 < (std::size_t(13) << 30) || free1 < (std::size_t(6) << 30)) {
    std::printf("skip: %.1f / %.1f GB free on the two devices, need 13 / 6\n", double(free0) / (1 << 30),
                double(free1) / (1 << 30));
    return ingine::test::kSkip;
  }

  ExpertResidency table;
  for (int l = kFirstShardLayer; l < LAYERS; ++l) table.split[l] = 0;  // 4.55 GiB on the second device

  const SafetensorsModel ckpt(dir);
  const Weights w = load_weights(dir, true, stream, &second, table);

  bool ok = true;
  auto expect = [&](bool c, const std::string& what) {
    if (!c) std::printf("FAIL %s\n", what.c_str());
    ok &= c;
  };
  expect(!w.residency.all_primary(), "the table is recorded as not all-primary");
  expect(w.residency.split[kFirstShardLayer] == 0 && w.residency.split[kFirstShardLayer - 1] == EXPERTS,
         "residency split per layer");

  std::size_t shard_bytes = 0;
  for (int l = 0; l < LAYERS; ++l) {
    const Experts& ex = w.layers[std::size_t(l)].experts;
    const std::string p = "model.llm.layers." + std::to_string(l) + ".mlp.experts.";
    if (l < kFirstShardLayer) {
      expect(ex.split == EXPERTS && ex.w13_hi.data == nullptr && ex.w13.data != nullptr &&
                 device_of(ex.w13.data) == primary,
             "layer " + std::to_string(l) + " keeps its experts on the primary device");
      continue;
    }
    expect(ex.split == 0 && ex.w13.data == nullptr && ex.w13_scale.data == nullptr && ex.w2.data == nullptr &&
               ex.w2_scale.data == nullptr,
           "layer " + std::to_string(l) + " has no expert bytes on the primary device");
    const Tensor hi[4] = {ex.w13_hi, ex.w13_scale_hi, ex.w2_hi, ex.w2_scale_hi};
    const char* names[4] = {"w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"};
    for (int i = 0; i < 4; ++i) {
      const std::string what = p + names[i];
      if (hi[i].data == nullptr) {
        expect(false, what + " is loaded");
        continue;
      }
      expect(device_of(hi[i].data) == peer, what + " is on the second device");
      expect(download(hi[i]) == ckpt.read(what), what + " equals the checkpoint");
      shard_bytes += hi[i].bytes();
    }
    if (ex.fmt == ExpertFormat::NVFP4) {  // moe reads wfactor[e] next to the codes: the factors follow
      expect(ex.w13_factor.data != nullptr && device_of(ex.w13_factor.data) == peer && ex.w2_factor.data != nullptr &&
                 device_of(ex.w2_factor.data) == peer,
             "layer " + std::to_string(l) + " factors follow the shard");
      shard_bytes += 2 * std::size_t(EXPERTS) * sizeof(float);  // w13_factor, w2_factor
    }
  }

  expect(w.bytes_second == shard_bytes, "bytes_second counts the second device's shards");
  const std::size_t lost1 = free1 - free_bytes(peer);
  expect(lost1 >= shard_bytes && lost1 <= shard_bytes + (std::size_t(128) << 20),
         "the second device lost the shard bytes (" + std::to_string(lost1 >> 20) + " MiB for " +
             std::to_string(shard_bytes >> 20) + " MiB of shards)");
  // the BF16 fp8-conversion temporaries (~1 GiB) are freed before load_weights returns
  const std::size_t lost0 = free0 - free_bytes(primary);
  expect(lost0 <= w.bytes_primary && lost0 + (std::size_t(2) << 30) >= w.bytes_primary,
         "the primary device lost its own allocations (" + std::to_string(lost0 >> 20) + " of " +
             std::to_string(w.bytes_primary >> 20) + " MiB)");

  {  // a plain one-device load of the same layer holds those bytes too
    std::vector<DeviceBuffer> plain_storage;
    const Layer plain = load_layer(dir, kFirstShardLayer, true, stream, plain_storage);
    expect(plain.experts.split == EXPERTS && plain.experts.w13_hi.data == nullptr &&
               device_of(plain.experts.w13.data) == primary,
           "load_layer keeps the experts on this device");
    expect(plain.experts.w13.bytes() == w.layers[std::size_t(kFirstShardLayer)].experts.w13_hi.bytes(),
           "w13 has the same size on either path");
    const std::vector<std::byte> here = download(plain.experts.w13);
    const std::vector<std::byte> there = download(w.layers[std::size_t(kFirstShardLayer)].experts.w13_hi);
    expect(here == there, "the shard equals a plain load_layer's w13");
  }

  INGINE_TEST_CUDA(cudaSetDevice(primary));
  INGINE_TEST_CUDA(cudaStreamDestroy(stream));
  std::printf("%s expert residency across two devices\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
