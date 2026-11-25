# Inkling-2-Mini abliterated: ingine

An abliterated [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) served
by **ingine**, a C++/CUDA engine. It answers harmful test prompts instead of refusing them (refusal phrases in 1
of 104 and 3 of 48 answers, the 3 followed by the requested text), and stays close to the original elsewhere
(KL 0.014 on harmless answers).

The checkpoint is mixed **NVFP4** (layers 0–19) and **MXFP8** (20–23), about 20 GB. That does not fit one 16 GB
card, and 16 GB of host RAM cannot hold the spill. Serving uses **two Blackwell GPUs** (compute capability 12.0):
expert-parallel 64/64, three of eight full-attention KV heads on the second card, DSpark, FP8 KV, 262,144 tokens
of context, OpenAI-compatible HTTP. There is no NVLink/P2P; activations hop through pinned host memory.

Measured on an RTX 5070 Ti 16 GB + RTX 5060 Ti 16 GB (PCIe 5.0 x8/x8, no P2P), live chat with speculative
decoding, prefix cache on:

| Cached context | Decode | Prefill (new tail) |
|---|---:|---:|
| ~4k | 140–170 tok/s | ~1.5–3k tok/s |
| ~50k | ~150–220 tok/s | ~2–2.8k tok/s |
| ~237k | ~108–122 tok/s | ~1.1–2.5k tok/s |

A follow-up only prefills what is new. `--no-vision` is the 16 GB layout (saves 0.6 GiB). `--park-mb 0`: there is
no spare VRAM for parked conversations.

The model files are
[satellitedown/Inkling-2-Mini-abliterated-ingine](https://huggingface.co/satellitedown/Inkling-2-Mini-abliterated-ingine)
(20.2 GB, pinned in [runtime-manifest.json](runtime-manifest.json)).

## Hardware

- Linux x86_64, **two** NVIDIA GPUs, compute capability **12.0**, **16 GB each** (32 GB split). A single 16 GB
  card cannot load the weights.
- Driver for CUDA 13 (580+). Kernels are `sm_120a` (Blackwell FP4 conversion). 16-CTA clusters fall back to 8 on
  smaller GPCs (5060-class).
- C++20 compiler (GCC 12–16), ~24 GB disk for the checkpoint and build.
- Two-GPU flags: `--second-gpu 1 --expert-split 64 --kv-split --dense-fp8 --kv-fp8 --spec --no-vision --park-mb 0 --max-ctx 262144`.

The installer below still assumes a build machine with CUDA 13. It does not install GPU drivers. The server is
C++/CUDA: no Python or PyTorch at runtime.

## Run

```bash
curl -fsSL https://raw.githubusercontent.com/satellitedown/inkling-2-mini-abliterated-ingine/main/get.sh | bash
```

Or:

```bash
git clone https://github.com/satellitedown/inkling-2-mini-abliterated-ingine
cd inkling-2-mini-abliterated-ingine
bash setup.sh
```

Menu: **1** install (tools, private CUDA 13.4, build `ingine-serve`, download the model), **2** download/resume
model, **3** foreground serve, **4** update, **5** install the `ingine` command. `bash setup.sh --help` lists
flags. After the build, run the two-GPU profile:

```bash
ingine-serve --engine --host 127.0.0.1 --port 8003 \
  --model-dir models/Inkling-2-Mini \
  --ablation models/Inkling-2-Mini/inkling-2-mini-abliterated.safetensors \
  --max-ctx 262144 --second-gpu 1 --expert-split 64 --kv-split \
  --dense-fp8 --kv-fp8 --spec --no-vision --park-mb 0
```

`scripts/ingine` / `ingine start` still exist for a foreground/systemd wrapper; pass the flags above in
`INGINE_EXTRA_ARGS` (or `~/.config/ingine/env`). Put the checkpoint on fast local disk (NVMe). Loading ~20 GB
through a SATA volume and 16 GB of host RAM is slow; once resident, weights stay in VRAM.

| Path | Contents |
|---|---|
| `.tools/` | uv / a small venv (CMake, Ninja, `huggingface_hub`) for the installer only |
| `.cuda-toolkit/` | private CUDA 13.4 used to build; the server loads CUDA libs from here |
| `native/build-install/` | `ingine-serve` and `profile.env` |
| `models/Inkling-2-Mini/` | Hugging Face files + abliteration (`INGINE_MODEL_DIR` to reuse a copy) |

| Variable | Default | Meaning |
|---|---|---|
| `INGINE_MODEL_DIR` | `models/Inkling-2-Mini` | model folder |
| `INGINE_ABLATION` | `$INGINE_MODEL_DIR/inkling-2-mini-abliterated.safetensors` | empty serves the original (`inkling-2-mini`) |
| `INGINE_CTX` | 262144 | context length |
| `HOST`, `PORT` | 127.0.0.1, 8003 | bind |
| `INGINE_EXTRA_ARGS` | empty | extra `ingine-serve` flags (two-GPU set, sampling, …) |

## API

`http://127.0.0.1:8003/v1`, model **`inkling-2-mini-abliterated`**. No auth: keep it on loopback. One request at
a time; 4 wait; then HTTP 503. `GET /health`, `GET /v1/models`. Streaming last chunk includes llama.cpp-style
`timings` (`prompt_ms`, `predicted_ms`, `cache_n`, `draft_n`) when `stream_options.include_usage` is set.

`POST /v1/chat/completions`: `reasoning_effort` (`none` … `max` or 0–0.99), `enable_thinking: false`, `tools`,
`stop`, `max_tokens`, `temperature` / `top_p` / `top_k` / `seed`, `stream`. Reasoning arrives as
`reasoning_content`. Images work if you drop `--no-vision` (see below).

```bash
curl -s http://127.0.0.1:8003/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini-abliterated",
  "messages": [{"role": "user", "content": "Explain how hash maps handle collisions."}],
  "reasoning_effort": "low",
  "max_tokens": 4096
}'
```

### Images

Off in the two-16 GB profile (`--no-vision`). With vision enabled, one token per 40×40 pixels; `--image-max-tokens`
caps large images; 16 images / 32,768 image tokens per request. Follow-ups reuse image tokens like text.

## Oh My Pi (OMP)

```yaml
providers:
  ingine:
    baseUrl: http://127.0.0.1:8003/v1
    auth: none
    api: openai-completions
    models:
      - id: inkling-2-mini-abliterated
        name: Inkling 2 Mini abliterated 256K (ingine)
        contextWindow: 262144
        maxTokens: 32768
        reasoning: true
        thinking:
          mode: effort
          efforts: [minimal, low, medium, high, xhigh, max]
          defaultLevel: medium
          requiresEffort: false
        input: [text]
        cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0}
        compat:
          supportsDeveloperRole: true
          supportsStore: false
          supportsStrictMode: false
          supportsToolChoice: false
          supportsReasoningEffort: true
          reasoningEffortMap: {minimal: none}
          reasoningContentField: reasoning_content
          maxTokensField: max_tokens
          streamIdleTimeoutMs: 600000
```

`omp --model ingine/inkling-2-mini-abliterated`. OMP `minimal` is sent as `none`.

## Abliteration

Refusals on harmful test prompts (greedy; refusal phrase in
[`results/verify-abliterated.json`](results/verify-abliterated.json)):

| | original | abliterated |
|---|---|---|
| `harmful_test` (104), direct | 104/104 | 1/104 |
| JailbreakBench (100), direct | 100/100 | 5/100 |
| `harmful_test` (48), reasoning first (effort low) | 48/48 | 3/48 |
| through ingine: `harmful_test` (104), direct | 104/104 | 3/104 |
| through ingine: effort `none` | 104/104 | 0/104 |
| through ingine: `harmful_test` (48), reasoning first | 48/48 | 1/48 |
| through ingine: JailbreakBench (100), direct | 100/100 | 5/100 |

KL 0.014 on 64 harmless answers; 0.005 mean KL and 96.4% top-1 on 8,192 unused technical tokens. Math/code/MC
matched the original on the small sets in [`results/`](results). GSM8K 40/40; AIME 2024 14/15 abliterated vs 13/15
original.

## License

Apache-2.0, see [LICENSE](LICENSE). Base model
[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Thinking Machines) is Apache-2.0;
files are in the Hugging Face repo, not this git tree. CUDA keeps NVIDIA terms. Abliteration follows
[Heretic](https://github.com/p-e-w/heretic); you are responsible for how you use the uncensored weights. Unofficial,
not endorsed by Thinking Machines.
