# ingine (C++): design

ingine is a single C++20/CUDA program: `ingine-serve` loads Inkling-2-Mini (+ the ablation file) and serves the
OpenAI-compatible API with no Python or PyTorch at run time. It replaced ingine's first engine, written in Python
on PyTorch with custom CUDA extensions (`engine/`, in the history up to commit 44bc891), and reproduces its model
math, decode kernels, CLI flags and HTTP behaviour. Hardware: RTX 5090 class only (`sm_120a`), CUDA 13.4, 32 GB.

## Origin of each part

| C++ | Ported from the Python engine |
|---|---|
| `src/model/model_data.h`, `weights.cpp`, `state.cpp` | `weights.py`: checkpoint names, constants, visit plan (48 visits: layers 0-5, 6-17 three times, 18-23), expert formats (NVFP4 layers 0-19, MXFP8 20-23); `model.py` `State` (caches), ablation, taps |
| `src/model/decoder.cpp`, `runtime.cpp` | `fast.py`: the fused decode step (kernel sequence per visit, buffers per call width, projection layout, FP8 dense conversion) |
| `src/kernels/{dense,moe,attn}`, `KERNELS.md` | `csrc/**` and `SPEC.md`: the decode kernels and their contracts, ported unchanged |
| `src/model/prefill.cpp`, `src/kernels/prefill` | `model.py` `Model.prefill` (chunked prompt processing); new kernels (flash attention, grouped MoE GEMMs) |
| `src/model/drafter.*`, `src/kernels/draft` | `dspark.py`: the DSpark drafter (draft/commit, context buckets, top-1024 Markov candidates) |
| `src/model/generator.cpp`, `spec.*`, `src/kernels/head` | `engine.py`: prefix reuse, decode graphs per width (1, 2, 4, 8), speculative loop (draft, adaptive verify width, accept, rollback, commit), sampling |
| `src/text` | `chat.py`: chat rendering (the model's `chat_template.jinja`), reasoning/tool output parsing (`StreamParser`), effort levels |
| `src/serve`, `src/engine` | `server.py`: `/health`, `/v1/models`, `/v1/chat/completions` (streaming, `reasoning_content`, tools), FIFO scheduler |
| `src/image`, `src/kernels/vision`, `src/model/vision.*` | none (the Python engine was text only): image input, written against the transformers reference (`Inkling2ImageProcessor`, `Inkling2VisionModel`); see "Images" below |

The serving profile (262,144 tokens, `--dense-fp8 --kv-fp8 --spec --spec-threshold 0.8`, 127.0.0.1:8003) comes
from `runtime-manifest.json`.

## Base: fngine

Infrastructure comes from fngine (github.com/satellitedown/fngine, Apache-2.0, itself derived from Cinference):
build skeleton, vendored third-party libraries, core utilities, HTTP/OpenAI serving and the tokenizer/Jinja text
stack, with the namespace renamed `fngine` -> `ingine`. Every copied or adapted file carries the header

    // Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.

No Qwen-specific code (GDN/QSA/PLE/MTP kernels, CPU expert tier, n-gram store, pack format) was taken. See
`NOTICE`.

## Layout and targets

```
native/
  CMakeLists.txt, CMakePresets.json, cmake/          build (from fngine: arch gate 120a, no separable compilation,
                                                     vendored deps, one target per component)
  third_party/                                       nlohmann json, cpp-httplib, spdlog(+fmt), utf8proc, llama-jinja,
                                                     stb_image, libwebp (decoder)
  include/ingine/                                    public API used by apps (generator.h)
  src/core/        -> ingine_core                    device, arena, host memory, Tensor/DType, decode graphs, PDL,
                                                     logging glue, safetensors reader (mmap)
  src/text/        -> ingine_text                    tokenizer, Jinja adapter, ChatFormat, StreamParser (CUDA-free)
  src/image/       -> ingine_image                   image decoding (stb_image, libwebp), the reference
                                                     processor's pixels, reduction, content hash and prompt ids
                                                     (CUDA-free)
  src/kernels/<a>/ -> ingine_kernels_<a>             dense, moe, attn, head (ported decode kernels + sampling),
                                                     prefill (new), draft (new), vision (new); aggregated by
                                                     ingine_kernels
  src/model/       -> ingine_model                   weights, state, decode step + graphs, prefill driver,
                                                     drafter, speculative loop, vision tower, Generator
                                                     implementation
  src/engine/      -> ingine_engine                  request scheduler (1 active + bounded FIFO), cancellation
  src/serve/       -> ingine_serve                   HTTP server, OpenAI chat request/response, SSE
  apps/serve/      -> ingine-serve                   the server binary
  apps/bench/      -> ingine-bench                   decode/prefill/spec benchmarks
  apps/validate/   -> ingine-validate                parity against the golden data
  tests/                                             CTest: text goldens (CPU), kernel tests, model parity (GPU)
```

Build: `cmake --preset release && cmake --build --preset release` (`build/`); `cmake --preset dev` adds the tests;
the installer uses `install` (`build-install/`, the private toolkit in `../.cuda-toolkit` and an RPATH into it).
Without a preset, nvcc 13.4 comes from `../.cuda-toolkit` or `/opt/cuda`; GCC 12-16 work as host compiler.

## Interfaces

### Text (`src/text`, namespace `ingine::text`), mirror of `chat.py`

```cpp
class Tokenizer {               // tokenizer.json: byte-level BPE, o200k-style split regex (see below), no
 public:                        // normalizer, ignore_merges=true, 60 added tokens
  static Tokenizer load(const std::filesystem::path& model_dir);
  std::vector<int32_t> encode(std::string_view text, bool special) const;   // special: recognise added tokens
  std::string decode(std::span<const int32_t> ids) const;                    // raw UTF-8 (may be partial)
  std::string token_bytes(int32_t id) const;
  std::optional<int32_t> token_id(std::string_view added_token) const;
};
struct ChatError : std::runtime_error { std::string param; };              // -> HTTP 400 with "param"
struct ChatOptions {                       // reasoning_effort: "none".."max" or a number in [0, 0.99]
  std::optional<std::variant<std::string, double>> reasoning_effort;
  std::optional<bool> enable_thinking;
};
struct RenderedPrompt { std::vector<int32_t> ids; std::string text; std::vector<int32_t> prefill_ids; };
struct ToolCallDelta { int index; std::optional<std::string> id, name; std::string arguments; };
struct Delta { std::string reasoning_content, content; std::optional<ToolCallDelta> tool_call; };
struct ToolCall { std::string id, name, arguments; };
struct Completion { std::string reasoning_content, content; std::vector<ToolCall> tool_calls;
                    std::optional<std::string> finish_reason; };
class StreamParser {                       // chat.StreamParser: feed sampled tokens, get deltas
 public:
  std::vector<Delta> feed(int32_t token);
  std::vector<Delta> finish();             // end of generation (length/cancel)
  bool finished() const;                   // end of turn, stop string, or end-of-sampling token seen
  Completion result() const;
};
class ChatFormat {
 public:
  static ChatFormat load(const std::filesystem::path& model_dir);   // tokenizer + chat_template.jinja
  RenderedPrompt render(const nlohmann::ordered_json& messages, const nlohmann::ordered_json* tools,
                        const ChatOptions& options) const;          // chat.normalize_messages + template
  StreamParser parser(std::span<const int32_t> prefill_ids, std::vector<std::string> stop = {}) const;
  std::vector<int32_t> stop_ids() const;                            // end-of-turn ids the generator stops at
  const Tokenizer& tokenizer() const;
};
```

The split regex (tokenizer.json `pre_tokenizer`) is
`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`;
implement it as a hand-written matcher with utf8proc categories (fngine's `qwen_word_end` approach).

### Generator (`src/model`, namespace `ingine::model`), mirror of `engine.Engine.generate`

```cpp
struct Sampling { float temperature = 1.0f; float top_p = 1.0f; int top_k = 0; std::optional<uint64_t> seed; };
struct GenerateStats { int prompt_tokens = 0, reused_tokens = 0, generated_tokens = 0;
                       double prefill_seconds = 0, decode_seconds = 0; int spec_rounds = 0, spec_accepted = 0; };
struct RuntimeOptions {
  std::filesystem::path model_dir, ablation;      // empty ablation: original model
  int max_ctx = 65536; int prefill_chunk = 2048;
  bool dense_fp8 = true, kv_fp8 = false, spec = false; float spec_threshold = 0.8f;
};
class Generator {
 public:
  virtual ~Generator() = default;
  virtual std::string model_id() const = 0;       // "inkling-2-mini-abliterated" with an ablation, else "inkling-2-mini"
  virtual int max_model_len() const = 0;          // = max_ctx
  // One request at a time. Emits sampled tokens in order through on_tokens (a speculative round emits several);
  // stops after a token in stop_ids, after max_new_tokens, or when on_tokens returns false (cancel).
  // Throws std::invalid_argument when the prompt does not fit max_ctx (reserve as engine.py does).
  virtual GenerateStats generate(std::span<const int32_t> prompt, int max_new_tokens, const Sampling& sampling,
                                 std::span<const int32_t> stop_ids,
                                 const std::function<bool(std::span<const int32_t>)>& on_tokens) = 0;
};
std::unique_ptr<Generator> make_generator(const RuntimeOptions& options);   // the model
```

### Model data (`src/model/model_data.h`)

C++ mirrors of `weights.Weights`/`Layer`/`Experts`, `model.State` and `fast.Buffers`, using `ingine::Tensor`
(fngine's non-owning view: data, DType incl. FP8_E4M3FN, ne/nb) over device memory. Field names follow the
Python names (`t["wq"]` -> `layer.wq`, `state.k_ring`, `buffers.taps`, ...), shapes and dtypes identical, plus
the visit plan and constants of `weights.py`. The decode step, the prefill and the drafter share this header.

Expert residency across two devices (EP step 2) is a load-time table. `ExpertResidency` in
`include/ingine/generator.h` (CUDA-free, like the rest of `RuntimeOptions`) holds, per layer, how many of that
layer's 128 experts stay on the primary device; `load_weights(dir, dense_fp8, stream, second, residency)`
places a layer's four expert tensors and their factors whole on the device the table names (`Experts::split`,
the `_hi` views, `Weights::storage_hi`, freed with their own device current), one granule of 546.75 MiB NVFP4 /
1002.4 MiB MXFP8 per layer. The default table is all-primary: no second uploader, no second allocation, the same
allocations in the same order on the same stream. Only `0` and `EXPERTS` are loadable - the moe kernels take one
base pointer per expert tensor and address rows as `e*R`, so a split inside a layer is step 3.

Decode then reads the table (EP step 4). A visit whose layer has `split == 0` runs its experts on the second
device (`Decoder::expert_hop`): GPU0's `xm` (bf16 `[m, H]`, 4.6 KiB at m = 1) and `idx` go to pinned host
staging, `moe_up` and `moe_down_rows` run there, one BF16 row per slot comes back (36 KiB at m = 1), and this
device's `moe_sum` does `moe_down`'s slot-ordered FP32 weighted sum and post-MLP conv. The rows are the values
the fused epilogue sums, so the split is bit-identical to the one-device path (the two kernels are compared in
`tests/kernels/test_bitlock.cpp`); the hop is the protocol `apps/hop` measured at 27.1 us per round trip with no
peer access, ordered by one non-timing event pair per visit and never by a host wait. Layers with
`split == EXPERTS` keep the fused `moe_up`/`moe_down` pair, so the default table copies nothing.

That step spans two devices, and a CUDA graph belongs to one, so an expert split decodes eagerly
(`Runtime::init` skips capture, `run_step` falls back to `run_step_eager`) and DSpark is refused until dual
graphs exist; the prefill's grouped expert GEMMs take one base per expert tensor on this device, so `Prefiller`
refuses a table that moves any layer's experts.

### Prefill (`src/model/prefill.{h,cpp}`)

```cpp
// Processes prompt tokens [T] (device int32) at positions pos0..pos0+T-1 on top of `state` (exactly
// model.Model.prefill for one chunk): writes caches/conv state, returns the last token's logits (device fp32 [V],
// softcapped like the head) and, if taps != nullptr, every row's DSpark taps (bf16 [T, 4, H], model.TAP_VISITS
// + final normed hidden).
class Prefiller {
 public:
  Prefiller(const Weights& w, const Ablation& abl, int max_chunk, cudaStream_t stream);
  // rows: embedding rows computed elsewhere (image tokens) for rows of the chunk
  void run(State& state, const int32_t* tokens, int T, int pos0, float* logits, Tensor* taps,
           std::span<const EmbedRows> rows = {});
};
```

### Images (`src/image`, `src/kernels/vision`, `src/model/vision.{h,cpp}`)

The checkpoint holds Inkling-2's vision tower (`model.visual.*`, 422 BF16 tensors, 636 MB): a 23-layer ViT over
20x20-pixel patches with a class token, learned position embeddings resampled to the patch grid, 2-D rotary
attention and layer scale, and a projector that merges 2x2 patches into one row of the language model's width.
The language model sees an image as one token per 40x40-pixel tile, `ceil(H/40) * ceil(W/40)` of them, whose
embeddings are those rows; positions stay ordinary sequence positions. `KERNELS.md` has the arithmetic.

```cpp
namespace ingine::image {                    // CPU: a request thread decodes
struct Image { int width, height; std::vector<uint8_t> rgb; uint64_t hash; int tokens() const; };
Image decode_data_uri(std::string_view uri, const Limits& limits);   // PNG, JPEG, GIF, BMP (stb_image), WebP (libwebp)
int32_t token_id(uint64_t hash, int k);      // prompt id of image token k: negative, from content and k
}
namespace ingine::model {
struct PromptImage { int pos, height, width; std::shared_ptr<const std::vector<uint8_t>> rgb; };
struct Prompt { std::span<const int32_t> ids; std::span<const PromptImage> images; };
class VisionEncoder {                        // GPU: resident weights, working memory per call
 public:
  VisionEncoder(const std::filesystem::path& model_dir, cudaStream_t stream);
  DeviceBuffer encode(const uint8_t* rgb, int height, int width);   // bf16 [tokens (padded), 2304]
};
}
```

Path of an image through a request:

1. `serve::parse_request` collects the image parts of the messages (base64 `data:` URIs; nothing is fetched).
   The chat template renders each as `<|content_image|><|unused_200054|><|end_message|>`: one placeholder
   token. A request whose text spells the placeholder as well as carrying images is rejected (the count of
   placeholders must equal the count of images), so the k-th placeholder is the k-th image.
2. `image::decode_data_uri` produces the pixels of the reference processor: transparency composited onto its
   checkerboard, no resizing, the first frame of an animation (PNG, GIF and WebP pixels are identical to the
   reference's, JPEG within 2 levels). An image above `--image-max-tokens` (4096) is reduced by area averaging, which
   the reference does not do (it takes any size up to 4096x4096 pixels, 10,609 tokens).
3. Each placeholder is replaced by the image's tokens as `image::token_id(hash, k)`: negative ids, a function
   of the pixels and the tile index. The Generator's prompt caches compare token ids, so with these ids they
   tell images apart without knowing about images: the same image gives the same prompt (extend, checkpoint
   and parked reuse work, and an image inside the reused prefix is not encoded again), another image at the
   same place ends the common prefix at its first token, and a prefix that ends inside one image never
   matches another. A false match needs 31 random bits to agree per token. The context check and the usage
   count the expanded prompt.
4. `EngineGenerator::prefill` encodes an image when the first chunk holding some of its tokens is reached
   and passes its rows to `Prefiller::run`, which writes them over the embedding block after the token
   embedding (negative ids are embedded as the placeholder first, so the lookup stays in its table). The model
   feeds image rows in as they are (no embedding norm); every later read of the embedding block, the taps
   of the drafter included, sees them. An image may span chunks. Decoding is unchanged: generated tokens are
   never image tokens.

Precision: matrix products take BF16 operands with FP32 accumulation, the residual stream, LayerNorm, softmax
and GELU are FP32. Against transformers in FP32, the output is about twice as close as transformers' own BF16
model on every fixture (relative RMS error 0.3-3%, cosine 0.9995-0.99999; the tower amplifies rounding noise
about a hundredfold over its 23 layers). Memory: the weights are resident (609 MiB, loaded before the park
budget is measured); one encode holds about 36 KB per patch plus up to 256 MiB of attention scores (0.85 GB
for 4096 tokens), released on return. Time on an RTX 5090: 3 ms up to 64 tokens, 21 ms for 576, 0.17 s for
1849, 1.1 s for 4096 (attention is a GEMM pair per head over chunks of queries; a fused kernel would be the
next step). `RuntimeOptions::vision` (ingine-serve: on unless `--no-vision`) loads the tower.

### Speculative decoding (`src/model/drafter.*`, `src/model/spec.*`)

`Drafter` mirrors `dspark.Drafter` (`draft(anchor, P, temperature) -> tokens[7], logits[7, V]`,
`commit(feats [n,4,H], start)`, graphs per context bucket and one for small commits). `SpecLoop::generate(...)`
mirrors `engine.Engine._generate_spec` and is called by the Generator when `options.spec`. It uses the
runtime's decode graphs for widths 2, 4 and 8 (history + taps on): the verify width is chosen on the device
from the draft's confidences (a CUDA graph with a switch node over the widths: backup, decode), a second
switch graph accepts and rolls back, and a third (`Drafter::advance`, per context bucket) commits the
accepted rows and drafts for the next anchor from device values, so a round and the next draft are launched
back to back and the host synchronises once per round, after the acceptance. The drafter scores its base
logits over a shortlist head (the 32,768 most frequent tokens, the specials and up to 16,384 tokens of the
context, `note_tokens`), runs its block attention on tensor cores and keeps its global ring in E4M3: any
draft distribution is valid, because drafts are verified by the full model (greedy) or accepted with the
probabilities they were drawn from. It keeps checkpoints of its context for prompt-prefix reuse
(`save_checkpoint`/`restore_checkpoint`; `checkpoint_regions`/`adopt_checkpoint` for parked conversations).

`src/model/runtime.h` is the model runtime the Generator, Prefiller and speculative loop share: stream, weights,
state, decode graphs per width with history and taps, the state accessors for rollback, and pipelined width-1
steps (`chain_*`: each step's input token is the previous step's sampled token on the device, so plain decoding
launches step N+1 before the host reads token N; a step launched past a stop token or a cancel is undone) and
prompt checkpoints (`save_checkpoint`/`restore_checkpoint`: the sliding rings, ring positions and conv states;
full-attention caches are position-addressed).

Prefix reuse (Generator): a prompt that extends the tokens in the state (previous prompt + processed generated
tokens) prefills only its suffix; otherwise, if the prompt of the last prefill is a prefix of it, the checkpoint
taken right after that prefill (target state, drafter context, last logits; one device buffer, ~300 MB at
max_ctx 262144) is restored and only the rest is prefilled. That is the agent-loop case: the next request
re-renders the previous answer (canonical tool-call JSON, retokenised text), which differs from the generated
tokens right after the previous prompt. A conversation displaced by another one is parked: its checkpoint and
its full-attention rows [0, n) are copied aside on the device (about 18 KB per token + 220 MB, within
`RuntimeOptions::park_mb`, least recently used evicted, all released when a device allocation fails) and
restored when a later prompt starts with its prompt. Otherwise the state is reset. The server logs the path
per request (`<id>: reuse=extend|checkpoint|parked|none reused=N prefilled=M prefill=<seconds>`).

Third-party CMake targets: `nlohmann_json::nlohmann_json`, `httplib::httplib`, `spdlog::spdlog`, `ingine_utf8proc`,
`ingine_jinja`, `ingine_stb`, `ingine_webp`, `CUDA::cudart`, `CUDA::cublasLt`.

## Conventions

- C++20, CUDA 20, fngine style (keep a file's existing style), no exceptions through CUDA callbacks,
  `cudaLaunchKernelEx` + PDL as in `pdl.cuh`, one stream per Generator.
- The decode kernels keep their numerics: every rounding point in `KERNELS.md` stays unless a change is measured
  against the goldens.
- Each component's `CMakeLists.txt` lives in its directory and defines its target.

## Testing

- `tests/golden/` holds the Python engine's outputs, generated before it was removed (by
  `native/tools/goldens_text.py` and `native/tools/goldens_model.py`, in the history up to commit 44bc891):
  tokenizer encodes, chat renders, StreamParser delta sequences, and model logits (prefill last-token top-256 and
  32 teacher-forced decode steps for 10 prompt/cache/ablation cases).
- Text: exact equality with the goldens (CPU). Kernels: FP64 CPU oracles (`tests/kernels`, `tests/prefill`,
  `tests/spec`, < 2 GB of GPU memory). Model: `ingine-validate` (KL and top-1 per stage, limits at the Python
  reference's own reproducibility; see `apps/validate/main.cpp`) and exact greedy continuations, with and without
  speculation, on the tie-free cases.
- Images: `tests/golden/vision` holds outputs of the transformers reference (image processor, vision tower
  and projector in FP32) for the fixtures in `tests/golden/vision/fixtures`, written by `tools/vision_ref.py`
  (CPU only; `--check-spec` also compares an independent NumPy implementation of `KERNELS.md`'s description).
  `image` (CPU): decoded pixels equal the processor's. `vision_kernels`: FP64 oracles. `vision_model`: the whole
  tower against the goldens, within transformers' own BF16 error (needs the model file, 1.5 GB of GPU memory).
  `model_vision` (full model): colours and text of images, the prompt caches with images, an image over
  several prefill chunks. `ctest -L vision` runs them.
- `ctest --test-dir build -LE model` runs everything but the full-model tests; `-L model` needs about 26 GB of
  GPU memory (stop the server first).
