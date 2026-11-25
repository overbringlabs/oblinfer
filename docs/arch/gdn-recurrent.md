# Qwen3.5/3.8 gated delta-rule (decode, seq_len=1)

Reference: Hugging Face `transformers` `torch_recurrent_gated_delta_rule` in
`modeling_qwen3_5.py` (Qwen3_5GatedDeltaNet). Decode uses this; prefill uses the chunked variant.

Shapes after in_proj + depthwise conv + split:

- `q, k`: `[B, S, num_k_heads, k_head_dim]`  (16 × 128 for 27B/9B/4B)
- `v`: `[B, S, num_v_heads, v_head_dim]`     (48×128 on 27B, 32×128 on 9B/4B)
- `beta`: sigmoid(b), `[B, S, num_v_heads]`
- `g` (decay logits): `-exp(A_log) * softplus(a + dt_bias)`, `[B, S, num_v_heads]`
- state `S`: `[B, num_v_heads, k_head_dim, v_head_dim]`  FP32

If `num_v_heads != num_k_heads`, repeat_interleave q and k by `num_v_heads / num_k_heads`.

Per token (FP32):

```
q = l2norm(q); k = l2norm(k); q = q / sqrt(k_head_dim)
S = S * exp(g)                         # decay, g already includes -A*softplus
kv_mem = (S * k[..., None]).sum(-2)    # [v_heads, v_dim]
delta = (v - kv_mem) * beta
S = S + k[..., None] * delta[..., None, :]
out = (S * q[..., None]).sum(-2)
```

Then RMSNorm-gated on `out` with `z`, then `out_proj` to hidden.

Conv decode: depthwise causal conv update, kernel 4, groups = conv_dim = 2*key_dim + value_dim.
