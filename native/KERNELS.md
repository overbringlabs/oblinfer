# ingine decode kernels: contracts

ingine runs Inkling-2-Mini on one RTX 5090 (sm_120a, 32 GB, ~1.79 TB/s). The decode step (one token, or up to 8
consecutive tokens for speculative verification) is a fixed sequence of custom kernels captured in one CUDA graph
per call width (`src/model/decoder.cpp`, `src/model/runtime.cpp`). This file is the contract between those
kernels (`src/kernels/{moe,attn,dense}`) and the host code. The prefill (`src/kernels/prefill`) and the DSpark
drafter (`src/kernels/draft`) compute the same model math; their headers document their own precision.

The kernels were written as PyTorch extensions, qualified against ingine's PyTorch model (FP32 activations, BF16
matrix inputs; itself checked against the transformers reference) and then ported unchanged to C++ launchers.
That engine was removed once the C++ engine matched it; it is in the history up to commit 44bc891 (`engine/`).
Weight layouts and the visit plan are in `src/model/model_data.h` and `src/model/weights.cpp`.

## Model facts used by kernels

* H = 2304, 24 query heads, 8 KV heads, head_dim 128 (GQA 3:1), d_rel 16, rel_extent 1024, window 1024.
* 48 layer visits per token: layers 0-5, layers 6-17 three times, layers 18-23. Full-attention layers are
  5, 11, 17, 23; all others use a 1024-token sliding window.
* MoE: 128 experts, top-8, expert intermediate I = 1152. Experts of layers 0-19 are NVFP4, 20-23 MXFP8
  (layouts in `src/model/model_data.h`). `w13` rows are interleaved: row 2i = gate i, row 2i+1 = up i.
* All short convolutions have kernel 4 and keep the previous 3 inputs as state ``[3, C]`` (oldest first,
  zeros before the first token).

## Conventions (all kernels)

* One static library per area: `src/kernels/<area>/` -> `ingine_kernels_<area>`, raw-pointer launchers declared
  in `<area>.h`.
* Every launcher takes preallocated outputs and workspaces and a stream, never allocates, never synchronises, and
  reads positions/lengths from device memory, so it can be captured in a CUDA graph and replayed with different
  contents.
* `M` = tokens per call (1 ≤ M ≤ 8), consecutive positions. Correctness is required for every M in that
  range; performance matters first for M = 1.
* Matrix inputs are BF16; accumulation and activations are FP32. Where the reference rounds an
  intermediate to BF16 (noted below), kernels round it too; everything else may stay FP32.
* Programmatic dependent launch: a kernel may issue loads of weights (and of data written two or more
  kernels back) before `griddepcontrol.wait`, so they overlap its predecessor. ptxas sinks such loads
  below a plain wait (it only orders later accesses), so the wait goes in a loop with an opaque zero
  trip bound (`pdl_wait_fenced` in `dense/cluster.cuh`); check the SASS (`ACQBULK` after the `LDG`s).
* L2 policy (`common/l2.cuh`): streams read once per step (dense and expert weights, the head, cached K/V)
  are loaded evict-first (+1-2% end to end). The small weights re-read every step (gates, router) carry no
  hint: evict-last keeps them resident but measured about 2% slower end to end.
* Qualification: FP64 CPU oracles at real model shapes (`tests/kernels/test_<area>.cpp`, NVFP4 and MXFP8,
  sliding and full, every M), real layers in `tests/model/test_layer.cpp`, and the whole model against the
  goldens in `tests/golden/model` (`ingine-validate`).
* Bit locks: every output bit of a call of M = 1..8 rows is hashed on fixed inputs: mix, message_gates and
  router in `tests/kernels/test_dense.cpp`; proj, proj_conv, head, moe_up / moe_down (both formats) and
  sliding attention in `tests/kernels/test_bitlock.cpp`. A speed change to these kernels must keep the hashes
  (a verify call of M rows keeps its bits).

## Area `moe` (routed experts)

`moe_up(x, idx, w13, w13_scale, w13_factor, fmt, out[, ebase, ecount])`

* `x` bf16 `[M, H]` (normalised MLP input), `idx` int32 `[M, 8]`, `out` bf16 `[M, 8, I]`.
* `fmt` 0 = NVFP4 (`w13` uint8 `[n, 2I, H/2]`, `w13_scale` float8_e4m3 `[n, 2I, H/16]`, `w13_factor`
  fp32 `[n]`), 1 = MXFP8 (`w13` float8_e4m3 `[n, 2I, H]`, `w13_scale` uint8 E8M0 `[n, 2I, H/32]`,
  factor unused). `n = ecount` (default 128): the weights hold experts `[ebase, ebase + ecount)`, addressed by
  the local id `e - ebase`; slots whose expert is outside that shard are skipped (the other device runs them).
* Per (m, e = idx[m, k]): `gu = W13_e · x[m]` (decoded weight value = code × block scale × factor, as
  `model.decode_experts`); gate = gu[2i], up = gu[2i+1] each rounded to BF16 (reference BF16 linear
  output); `out[m, k, i] = bf16(softcap(gate, 4) * sigmoid(gate) * softcap(up, 25))`,
  `softcap(x, c) = c * tanh(x / c)`.

`moe_down(act, idx, wts, w2, w2_scale, w2_factor, fmt, conv_w, conv_state, out[, ebase, ecount])`

* `act` bf16 `[M, 8, I]`, `wts` fp32 `[M, 8]`, `w2` NVFP4 uint8 `[128, H, I/2]` + e4m3 `[128, H, I/16]` +
  factor, or MXFP8 e4m3 `[128, H, I]` + E8M0 `[128, H, I/32]`.
* `s[m] = bf16( Σ_k wts[m, k] * bf16(W2_e · act[m, k]) )` (per-expert output rounded to BF16, FP32 sum,
  sum rounded to BF16).
* Post-MLP convolution per channel c (`conv_w` fp32 `[H, 4]`, conv1d order: tap 3 multiplies the current
  token): `ext = [conv_state[0..2], s[0..M-1]]`, `out[m, c] = (Σ_j conv_w[c, j] * ext[m + j] + s[m]) / √5`.
  Write `conv_state = ext[M..M+2]` (the last three inputs). `out` fp32 `[M, H]`.
* Target: ≥ 80% of DRAM bandwidth on the 8 experts' bytes per token (NVFP4 ≈ 4.5 MB per expert, MXFP8 ≈
  8.2 MB) for M = 1.

`moe_down_rows(act, idx, w2, w2_scale, w2_factor, fmt, rows[, ebase, ecount])` +
`moe_sum(rows, wts, conv_w, conv_state, out)` + `moe_rows_merge(rows, rows_hi, idx, ebase)`

* The same arithmetic as `moe_down`, split at the weighted sum so the experts can run on another device (expert
  parallelism, `src/model/decoder.cpp`'s `expert_hop` / `expert_split_hop`). `moe_down_rows` is `moe_down`'s
  expert pass alone: it writes `rows[(m * 8 + k) * H + c] = bf16(W2_e · act[m, k])` for every slot its shard
  holds (`[ebase, ebase + ecount)`; the other slots are left untouched), the value `moe_down` sums, and reads no
  `wts`, conv state or output. `moe_sum` is its epilogue on the device that owns the state:
  `s[m] = bf16(Σ_k wts[m, k] * rows[(m * 8 + k)])` in slot order, then the post-MLP convolution above.
* A 64/64 split runs `moe_down_rows` on both devices; `moe_rows_merge` copies the rows of the slots whose expert
  is `>= ebase` from the second device's buffer over this device's, then one `moe_sum` sums all 8 slots in order.
* Bit-identical to one `moe_down` on the same `act`/`idx`/weights/state (`tests/kernels/test_bitlock.cpp`
  compares them for M = 1..8, both formats).

## Area `attn` (attention core)

`attn_decode(proj, layer params, conv_k_state, conv_v_state, k_cache, v_cache, kpos, pos, window,
log_scaling, workspace, out)`

* `proj` fp32 `[M, R]`, segments in this row order: q 3072 | gate 3072 | rel 384 | sink 24 | k 1024 |
  v 1024 | (full layers only) k_dyn 32 | v_dyn 32. Each value is a BF16-rounded projection output.
* Per token (positions `p = pos[0] + m`):
  * `q = clip(q.view(24,128) * q_scale, 22.627)`, rounded to BF16; full layers then `q *= tau(p)` and round
    again. `clip(x, t) = x * min(1, t / (‖x‖ + 1e-6))` per head; `tau(p) = 1 + 0.1 * ln(max((p+1)/8192, 1))`.
  * `rel = clip(rel.view(24,16) * r_scale, 8.0)`, BF16; full layers `rel *= tau(p)`, BF16;
    `rel_logits[h, d] = bf16(Σ_j rel[h, j] * bf16(rel_proj[j, d]))`, d < 1024.
  * Keys/values: sliding layers use the static conv (`k_conv`/`v_conv` fp32 `[1024, 4]`, conv1d order,
    `(conv + x)/√5`); full layers use the dynamic conv: tap weight for lag ℓ (0 = current) of channel c in
    head h is `k_static[ℓ, c] + k_dyn[h*4 + ℓ]`, output `(Σ_ℓ w_ℓ x_{t-ℓ} + x_t)/3`. Conv states as above
    (`[3, 1024]`, inputs are the BF16-rounded projections). Then round to BF16; keys:
    `clip(k.view(8,128), 22.627)`; values: per-head RMS norm without weight (eps 1e-6). Store into the cache
    (dtype below).
  * Cache: full layers write position p at `k_cache[p]`; sliding layers write slot `p % 1024` and set
    `kpos[slot] = p` (`kpos` int64 `[1024]`, -1 = empty).
  * Cache dtypes: `k_cache`/`v_cache` `[rows, 8, 128]` are both BF16, or, on full layers only, both
    float8_e4m3fn (`kv_fp8`: no scale). The kernel dispatches on the dtype; the call signature is the same.
    FP8 rule: the BF16 key/value above is rounded to E4M3, nearest even, saturated to ±448 (= torch
    `x.clamp(-448, 448).to(torch.float8_e4m3fn)`; real keys have norm ≤ 22.627 and values are RMS-normed,
    so saturation does not occur in practice). Every read of a cached entry, the call's own new keys/values
    included, uses that E4M3 value. Sliding rings are BF16 only (an FP8 ring is rejected).
  * KV-head range: a launch covers KV heads `[head_begin, head_begin + head_count)` of the 8; `cache_heads`
    is the head stride of `k_cache`/`v_cache` and of the conv state on that device (its own shard), and `out`
    starts at query head `out_head0` with `out_stride` elements per row. Defaults: 0, 8, 8, 0, 3072. A
    KV-head split runs heads 0-4 on one device and 5-7 on the other, each writing its own cache shard and its
    disjoint `out` heads (three query heads per KV head); Q/K/V of a head stay on one device, so the outputs
    concatenate along the head axis with no partial-softmax combine.
  * Scores over cached keys j (positions `kp`, distance `d = p - kp`): valid if `0 ≤ d` and, for sliding
    layers, `d < 1024` and `kp ≥ 0`. `s_j = (q_h · k_j)/√128 + (d < 1024 ? rel_logits[h, d] : 0)`.
    Softmax over valid j plus one sink logit `sink[h]` with zero value:
    `o_h = Σ_j e^{s_j} v_j / (Σ_j e^{s_j} + e^{sink_h})`. For M > 1 a token sees earlier tokens of the
    same call (write the call's keys first).
  * `out[m] = bf16( bf16(o) * sigmoid(gate) )`, bf16 `[M, 3072]` (head-major, as `o.reshape(3072)`).
* Full layers hold up to `rows` = `max_ctx` positions, at most 2^30 (positions and loop bounds are int32 on
  the device, row offsets 64-bit). The served profile uses 262,144 (FP8: 256 MiB per cache tensor, 512 MiB
  K+V per full-attention visit, 4 GiB for the 8 visits; BF16 doubles that). Split the key range across
  blocks (flash-decoding) with a combine pass, using `workspace`. Read the context length from `pos`, not
  from the host; a call with `pos + M > rows` does nothing.
* Target: close to DRAM bandwidth on the K/V bytes read for long contexts; small fixed overhead for the
  1024-key sliding layers.

## Area `dense` (projections, mixing, router, head)

`proj(xin, w, w_scale, segments, out)` — multi-segment GEMV.
* `xin` bf16 `[3, M, H]` (three possibly different input vectors per token), `w` bf16 `[R, H]` (or
  float8_e4m3 with fp32 per-row `w_scale [R]`), `segments` int32 `[S, 3]` = (row_begin, row_end,
  input_index), `out` fp32 `[M, R]` = BF16-rounded `w · xin[input_index]`.

`proj_conv(x, w, w_scale, conv_w, conv_state, out)` — attention output projection + post-attention conv.
* `x` bf16 `[M, 3072]`, `w` `[H, 3072]` (bf16 or fp8 + row scales), `y = bf16(w · x)`, then the same conv
  as `moe_down`'s epilogue with `conv_w [H, 4]`, `conv_state [3, H]`; `out` fp32 `[M, H]`.

`router(x, w, bias, idx, wts)` — `x` bf16 `[M, H]`, `w` `[128, H]` (fp32 or bf16 holding BF16 values),
`bias` fp32 `[128]`. `logits = x·w` (FP32), `s = sigmoid(logits)`, `idx` = the 8 experts with the largest
`s + bias`, `wts = 8 * s[idx] / (Σ s[idx] + 1e-20)`. Outputs int32 `[M, 8]`, fp32 `[M, 8]`.

`mix(blocks, base, write, abl_r, abl_w, partial_out, queries, norm_w, out_bf16, out_f32)` — residual
update + attention residual + RMS norm.
* `blocks`: int64 tensor of device pointers to completed block states, each fp32 `[M, H]` (0-5 of them).
* `new = base + write - w * abl_r * max(abl_r · write - offset, floor)` per token (`base` and/or `write` may be
  absent: pass empty tensors; `abl_r` fp32 `[H]` unit, `abl_w` fp32 `[4]` on device = `w, offset, floor,
  threshold`; offset 0 and floor `-FLT_MAX` give the plain projection). With a detector `abl_d` (fp32 `[H]`) the
  write is left unchanged unless `abl_d · write > threshold`. Written to `partial_out`.
* Stack `S = [blocks..., new]`, each state normalised `n_i = s_i / sqrt(mean(s_i²) + 1e-6)`. For each query
  `q` (fp32 `[Q, H]`, Q ∈ {1, 3}): `a_i = softmax_i(q · n_i / √H)`, `mix_q = Σ_i a_i n_i`.
  `out_f32[q, m] = mix_q` (optional), `out_bf16[q, m] = bf16(rms_norm(mix_q) * norm_w[q])` (optional).
* With no blocks and a single query the mix is skipped and `new` itself is normalised (layer 0).
* `mix_pair(p, msg)` runs two independent mixes in one launch (a message visit's attention read and its
  message mix: one block, four queries); same results as two `mix` calls.
* Side outputs written by the same launch, so the decode step needs no separate conversion or copy nodes
  (they would break the PDL chain): `new_bf16` = `bf16(new)` at a row stride (DSpark taps), `out_copy` = a
  strided copy of `out_bf16` (Q = 1; the final tap), and up to two plain FP32 row copies (`copies`: the K/V
  projection segments kept for speculative rollback).

`message_gates(h, msg, w_down, w_forget, w_gain, norm_w, out_bf16)` — loop passes 2 and 3.
* `h`, `msg` fp32 `[G, M, H]` (G ≤ 3 gates), `w_down` bf16 `[G, 64, 2H]`, `w_forget`/`w_gain` fp32
  `[G, H, 64]`, `norm_w` fp32 `[G, H]`. `h_shared`: `h` is `[M, H]` and every gate reads it (sliding
  layers mix one query for the three gates).
* `g = bf16(w_down · bf16([n(h), n(msg)]))` (n = RMS norm without weight), then
  `gated = (1 - tanh(w_forget · g)) * h + tanh(w_gain · g) * msg`, `out_bf16[g, m] = bf16(rms_norm(gated)
  * norm_w[g])`. For the MLP-side gate (G = 1) the host also needs `gated` in FP32: provide an
  optional `out_f32`.

`head(x, w, w_scale, out)` — `x` bf16 `[M, H]`, `w` `[200058, H]` bf16 or fp8 + row scales,
`out[m, v] = 30 * tanh(bf16(w_v · x)/30)` fp32 `[M, 200058]`. Target: DRAM bandwidth (0.92 GB BF16).

FP8 option (all GEMVs above): `fp8_rows(w_bf16) -> (w_fp8, scale)` with `scale[r] = max|w_r| / 448`; the
serving profile uses it (`--dense-fp8`).

## Area `vision` (image tower)

Not part of the decode step: `model::VisionEncoder` runs these once per image, before the prefill chunk that
holds the image's tokens. Reference: transformers `Inkling2ImageProcessor` and `Inkling2VisionModel`
(`tools/vision_ref.py`; its `--check-spec` implements this section in NumPy and compares it with transformers).
Matrix products are `kernels::prefill::Gemm` (BF16 operands, FP32 accumulation); the kernels of this area are
the elementwise work between them (`src/kernels/vision/vision.h`).

Pixels. RGB bytes `[H, W, 3]`; transparency is composited before (`src/image`): `(c * a + bg * (255 - a) + 127)
// 255` with `bg = 204` where `(y // 8 + x // 8)` is even, else `153`. No resizing. The image is padded right
and bottom to whole 40x40 tiles with the value -1 (on the 0-255 scale), then `x = (v - 127.5) / 127.5`.

Tokens. `rows x cols` tiles (`ceil(H/40)`, `ceil(W/40)`) in raster order, one language-model token each. Tile
`t` owns patches `4t .. 4t+3`: its (0,0), (0,1), (1,0), (1,1) quadrants of 20x20 pixels, each flattened
`(y, x, channel)` to 1200 values. The tower's sequence is the class token followed by the `4 rows cols` patches
(`S = 1 + 4 rows cols`); buffers pad `S` to a multiple of 64 (`Sp`), and no valid row reads a padding row.

* `patchify`: `out[p] = bf16(x)` of patch `p` (bf16 `[Sp, 1200]`).
* Patch embedding: GEMM with `patch_embedding.weight` `[1024, 1200]` (no bias).
* `embed_assemble`: `h[0] = class_embedding + cls_position_embedding`; `h[1 + p] = patch[p] + pos(p)`, where
  `pos` resamples the learned `[32, 32, 1024]` table bilinearly to the `2 rows x 2 cols` patch grid at the
  patch's row and column: `src = (i + 0.5) * 32 / size - 0.5`, taps `floor(src)` and `floor(src) + 1` clamped
  to `[0, 31]`, weights `1 - frac` and `frac`, per axis (FP32 weights, as the reference).
* `layernorm` (`ln_pre`): `(x - mean) / sqrt(var + 1e-5) * w + b` over the 1024 columns; `h` stays FP32.

23 layers, pre-LayerNorm with layer scale:

* `x = bf16(layernorm(h, ln_1))`; one GEMM with the stacked `[q; k; v]` weights `[3072, 1024]`.
* `qkv_post`: adds the biases; turns `q` and `k` by the token's 2-D rotary angles: 16 heads of 64, pair `j < 16`
  by `(col + 1) * f_j`, pair `16 + j` by `(row + 1) * f_j`, `f_j = 10000^(-2j/32)`, the class token at (0, 0),
  `(a, b) -> (a cos - b sin, b cos + a sin)`; writes `q`, `k` bf16 `[16, Sp, 64]` and `v` transposed bf16
  `[16, 64, Sp]`.
* Attention, full within the image: per head and chunk of queries, GEMM `q k^T` (FP32 scores), `softmax_rows`
  (`softmax(scores / 8)` over the `S` valid keys in FP32, rounded to BF16, zeros for padding keys), GEMM with
  `v^T` (bf16 `[Sp, 1024]`, heads side by side).
* `out_proj` GEMM; `bias_scale_residual`: `h += ls_1 * (y + bias)`.
* `x = bf16(layernorm(h, ln_2))`; `c_fc` GEMM `[4096, 1024]`; `bias_act`: `bf16(gelu(y + bias))`, GELU exact
  (`x * Phi(x)`); `c_proj` GEMM; `h += ls_2 * (y + bias)`.

There is no final norm. Projector: `merge_rows` drops the class token and places each tile's four patch states
side by side (bf16 `[N, 4096]`, contiguous by the token order above); `proj_in` GEMM `[4096, 4096]` (no bias),
exact GELU, `proj_out` GEMM `[2304, 4096]` (no bias) into BF16: one row per image token.

In the language model these rows replace the rows of the embedding block at the image's tokens
(`prefill::put_rows`, after `dense::embed`): the reference scatters them into the normed embeddings, so they
get no embedding norm, and they are BF16 values like the embedding rows.

Rounding points: the BF16 GEMM operands (pixels, LayerNorm outputs, `q`/`k`/`v`, attention probabilities and
outputs, GELU outputs, merger rows); everything else is FP32. The reference in BF16 rounds the residual stream
as well.
