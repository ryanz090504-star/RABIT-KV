"""
RABIT-KV canonical-quality-v2: ONE shared logical-quality implementation of the frozen RABIT policy
K3 / V2 / G32 / R4 / META8g64 with CANONICAL semantics (the state defined by kvquant_k3.quantize_rabit2_kv_ref /
Rabit2OnlineStateRef and served by the physical rabit_kv2 pages). Replaces, for NEW runs only, the legacy logical
evaluator (benchmarks/quality/*.py, left unchanged), which differed in two ways (semantic audit e88f460):
  1. V META8g64 metadata was flattened (head, token, V-group) from the HF layout; canonical is (token, head, V-group);
  2. only the prefix was quantized once; later tokens stayed BF16; canonical tokens keep aging (R4 residual -> open
     group -> closed 32-token page) as the sequence grows.

Canonical state at sequence length N (the oracle definition, independent of how N was reached):
  * recent = the newest min(N, 4) tokens, exact BF16;
  * old = the first N - 4 tokens, quantized as ONE region from token 0:
      K3: sequence-axis affine per (head, channel) over 32-token groups; the incomplete last group is zero-padded
          (padding values take part in min / max, exactly like the oracle); codes = round((x - min) / scale) in [0, 7],
          scale = (max - min) / 7 (1 if |scale| < 1e-8);
      V2: per (token, head) affine over 32-channel groups; scale = (max - min) / 3; codes in [0, 3];
      META8g64 (for each of K-min, K-scale, V-min, V-scale): the primary FP32 values flattened in CANONICAL order
          (K: [1, H, G, 1, D] -> (head, token-group, channel); V: [T, H, D/32, 1] -> (token, head, V-group)), grouped by
          64 (last value repeated as padding), uint8 affine codes with BF16 group min / scale;
      codes use the PRIMARY FP32 min / scale; dequantization uses the META8-decoded min / scale:
          value = code * scale_dec + min_dec   (float32; two roundings, same order as the oracle).
Because 32 * H * D / 32 and H * D are multiples of 64 for D = 128, every META8 group of a closed page lies inside that
page, so closed pages are independent of later tokens and are cached once (no cumulative requantization: the open
group and the residual are always recomputed from the RAW BF16 tokens).

HF integration: prefill runs in BF16 with a normal DynamicCache (dense BF16 prompt attention, as the physical initial
prefill); CanonicalRabitCache.from_prefill() then holds the canonical state, and every later forward must add exactly
ONE token (each token attends to the canonical state at its own sequence length, as the physical decode / causal-chunk
path). Attention-visible K / V are returned in the model dtype (BF16) in HF layout [1, H_kv, N, D].
Geometry-agnostic: all shapes come from the tensors (any H_kv, D with D % 32 == 0 and (H * D) % 64 == 0).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

GROUP = 32
RESIDUAL = 4
META_GROUP = 64
K_LEVELS = 7.0
V_LEVELS = 3.0
POLICY = {"k_bits": 3, "v_bits": 2, "group_size": GROUP, "residual_tokens": RESIDUAL, "metadata_bits": 8,
          "metadata_group_size": META_GROUP}


# ------------------------------------------------------------------------------------------------ META8g64
def meta8_roundtrip(primary: torch.Tensor) -> dict:
    """META8g64 of `primary` flattened in ITS OWN (already canonical) order. Returns codes / min / scale (BF16) and the
    decoded FP32 values reshaped to primary.shape."""
    data = primary.detach().float().contiguous()
    shape = tuple(data.shape)
    flat = data.reshape(-1)
    n = flat.numel()
    pad = (-n) % META_GROUP
    if pad:
        flat = torch.cat([flat, flat[-1:].expand(pad)], dim=0)
    g = flat.reshape(-1, META_GROUP)
    mn = g.amin(dim=-1, keepdim=True)
    mx = g.amax(dim=-1, keepdim=True)
    sc = (mx - mn) / 255.0
    sc = torch.where(sc.abs() < 1e-12, torch.ones_like(sc), sc)
    codes = torch.round((g - mn) / sc).clamp(0, 255).to(torch.uint8)
    mn16, sc16 = mn.to(torch.bfloat16), sc.to(torch.bfloat16)
    dec = (codes.float() * sc16.float() + mn16.float()).reshape(-1)
    if pad:
        dec = dec[:-pad]
    return {"codes": codes, "min": mn16, "scale": sc16, "pad": pad, "decoded": dec.reshape(shape)}


# ------------------------------------------------------------------------------------------------ K3 / V2
def k3(raw_k: torch.Tensor) -> dict:
    """K3 sequence-affine G32 of raw K [T, H, D] (canonical layout). Returns codes [Tp, H, D], metadata, decoded
    FP32 [T, H, D]."""
    t, h, d = raw_k.shape
    x = raw_k.detach().float().permute(1, 0, 2).unsqueeze(0)  # [1, H, T, D]
    pad = (-t) % GROUP
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    tp = t + pad
    g = x.reshape(1, h, tp // GROUP, GROUP, d)
    mn = g.amin(dim=3, keepdim=True)
    mx = g.amax(dim=3, keepdim=True)
    sc = (mx - mn) / K_LEVELS
    sc = torch.where(sc.abs() < 1e-8, torch.ones_like(sc), sc)
    codes = torch.round((g - mn) / sc).clamp(0, 7).to(torch.uint8)
    m_min, m_sc = meta8_roundtrip(mn), meta8_roundtrip(sc)  # [1, H, G, 1, D] order = (head, group, channel)
    vals = codes.float() * m_sc["decoded"] + m_min["decoded"]
    vals = vals.reshape(1, h, tp, d)[:, :, :t, :].squeeze(0).permute(1, 0, 2).contiguous()
    return {"codes": codes.reshape(1, h, tp, d).squeeze(0).permute(1, 0, 2).contiguous(), "pad_seq": pad,
            "min_meta": m_min, "scale_meta": m_sc, "decoded": vals}


def v2(raw_v: torch.Tensor) -> dict:
    """V2 per-token last-dimension affine G32 of raw V [T, H, D]; metadata in canonical (token, head, group) order."""
    t, h, d = raw_v.shape
    x = raw_v.detach().float()
    pad = (-d) % GROUP
    if pad:
        x = F.pad(x, (0, pad))
    dp = d + pad
    g = x.reshape(t, h, dp // GROUP, GROUP)
    mn = g.amin(dim=-1, keepdim=True)
    mx = g.amax(dim=-1, keepdim=True)
    sc = (mx - mn) / V_LEVELS
    sc = torch.where(sc.abs() < 1e-8, torch.ones_like(sc), sc)
    codes = torch.round((g - mn) / sc).clamp(0, 3).to(torch.uint8)
    m_min, m_sc = meta8_roundtrip(mn), meta8_roundtrip(sc)  # [T, H, G, 1] order = (token, head, group): CANONICAL
    vals = (codes.float() * m_sc["decoded"] + m_min["decoded"]).reshape(t, h, dp)[..., :d].contiguous()
    return {"codes": codes.reshape(t, h, dp), "min_meta": m_min, "scale_meta": m_sc, "decoded": vals}


def canonical_state(raw_k: torch.Tensor, raw_v: torch.Tensor) -> dict:
    """Full (non-incremental) canonical state for N raw tokens [N, H, D]: membership, K3 / V2 details, decoded FP32."""
    n = raw_k.shape[0]
    recent = min(n, RESIDUAL)
    old = n - recent
    out = {"n": n, "recent_count": recent, "old_count": old, "closed_pages": old // GROUP,
           "open_count": old % GROUP, "k": None, "v": None}
    parts_k, parts_v = [], []
    if old:
        out["k"], out["v"] = k3(raw_k[:old]), v2(raw_v[:old])
        parts_k.append(out["k"]["decoded"])
        parts_v.append(out["v"]["decoded"])
    parts_k.append(raw_k[old:].detach().to(torch.bfloat16).float())
    parts_v.append(raw_v[old:].detach().to(torch.bfloat16).float())
    out["decoded_k"], out["decoded_v"] = torch.cat(parts_k), torch.cat(parts_v)
    return out


# ------------------------------------------------------------------------------------ incremental layer state
class CanonicalLayerState:
    """Canonical RABIT state of one layer, grown token by token. Closed 32-token pages are decoded once and cached;
    the open group and the R4 residual are recomputed from the RAW BF16 tokens at every length."""

    def __init__(self, keep_fp32: bool = False):
        self.keep_fp32 = keep_fp32
        self.closed_k = self.closed_v = None  # decoded closed pages, FP32 canonical [P*32, H, D]
        self.tail_k = self.tail_v = None  # raw BF16 tokens not yet in a closed page [L, H, D]

    @property
    def n(self) -> int:
        c = 0 if self.closed_k is None else self.closed_k.shape[0]
        return c + (0 if self.tail_k is None else self.tail_k.shape[0])

    def append(self, raw_k: torch.Tensor, raw_v: torch.Tensor) -> None:
        """Append raw tokens [t, H, D] (BF16 model output)."""
        k, v = raw_k.detach().to(torch.bfloat16), raw_v.detach().to(torch.bfloat16)
        self.tail_k = k if self.tail_k is None else torch.cat([self.tail_k, k])
        self.tail_v = v if self.tail_v is None else torch.cat([self.tail_v, v])
        while self.tail_k.shape[0] - RESIDUAL >= GROUP:  # a full 32-token old group closes into a page
            pk, pv = k3(self.tail_k[:GROUP])["decoded"], v2(self.tail_v[:GROUP])["decoded"]
            self.closed_k = pk if self.closed_k is None else torch.cat([self.closed_k, pk])
            self.closed_v = pv if self.closed_v is None else torch.cat([self.closed_v, pv])
            self.tail_k, self.tail_v = self.tail_k[GROUP:], self.tail_v[GROUP:]

    def decoded(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention-visible FP32 K / V [N, H, D] of the current canonical state."""
        lt = self.tail_k.shape[0]
        recent = min(lt, RESIDUAL)
        old = lt - recent
        pk = [] if self.closed_k is None else [self.closed_k]
        pv = [] if self.closed_v is None else [self.closed_v]
        if old:
            pk.append(k3(self.tail_k[:old])["decoded"])
            pv.append(v2(self.tail_v[:old])["decoded"])
        pk.append(self.tail_k[old:].float())
        pv.append(self.tail_v[old:].float())
        return torch.cat(pk), torch.cat(pv)


# ------------------------------------------------------------------------------------------- HF integration
def _hf_cache_base():
    from transformers.cache_utils import DynamicCache  # noqa: PLC0415  (imported lazily: tests may run without HF)
    return DynamicCache


def make_canonical_cache_class():
    DynamicCache = _hf_cache_base()

    class CanonicalRabitCache(DynamicCache):
        """HF cache holding the canonical RABIT state per layer; each update() must add exactly one token."""

        def __init__(self):
            super().__init__()
            self.states: list[CanonicalLayerState] = []

        @classmethod
        def from_prefill(cls, prefill_cache) -> "CanonicalRabitCache":
            legacy = prefill_cache.to_legacy_cache() if hasattr(prefill_cache, "to_legacy_cache") else prefill_cache
            obj = cls()
            for layer_idx, (k, v) in enumerate(legacy):
                if k.shape[0] != 1:
                    raise ValueError("canonical-quality-v2 supports batch size 1")
                st = CanonicalLayerState()
                st.append(k[0].permute(1, 0, 2), v[0].permute(1, 0, 2))  # HF [1, H, T, D] -> canonical [T, H, D]
                obj.states.append(st)
                ck, cv = st.decoded()
                obj.key_cache.append(ck.permute(1, 0, 2).unsqueeze(0).to(k.dtype))
                obj.value_cache.append(cv.permute(1, 0, 2).unsqueeze(0).to(v.dtype))
            obj._seen_tokens = obj.key_cache[0].shape[-2]
            return obj

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            if key_states.shape[0] != 1 or key_states.shape[-2] != 1:
                raise ValueError("canonical-quality-v2 cache: exactly one new token per forward (batch size 1)")
            if layer_idx == 0:
                self._seen_tokens += 1
            st = self.states[layer_idx]
            st.append(key_states[0].permute(1, 0, 2), value_states[0].permute(1, 0, 2))
            ck, cv = st.decoded()
            self.key_cache[layer_idx] = ck.permute(1, 0, 2).unsqueeze(0).to(key_states.dtype)
            self.value_cache[layer_idx] = cv.permute(1, 0, 2).unsqueeze(0).to(value_states.dtype)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

    return CanonicalRabitCache


def logical_bytes(n: int, layers: int, kv_heads: int, head_dim: int) -> dict:
    """Logical storage of the canonical state at length n (packed payload bits + META8 codes + BF16 group min / scale +
    BF16 residual), per the canonical representation; NOT allocator capacity."""
    recent = min(n, RESIDUAL)
    old = n - recent
    per = {"k_payload": 0, "v_payload": 0, "k_meta": 0, "v_meta": 0, "residual": recent * kv_heads * head_dim * 2 * 2}
    if old:
        tp = old + (-old) % GROUP
        per["k_payload"] = (tp * kv_heads * head_dim * 3 + 7) // 8

        def meta(count):
            padded = count + (-count) % META_GROUP
            return padded + (padded // META_GROUP) * 4  # uint8 codes + BF16 min + BF16 scale per group

        per["k_meta"] = 2 * meta(kv_heads * (tp // GROUP) * head_dim)
        dp = head_dim + (-head_dim) % GROUP
        per["v_payload"] = (old * kv_heads * dp * 2 + 7) // 8
        per["v_meta"] = 2 * meta(old * kv_heads * (dp // GROUP))
    out = {k: layers * v for k, v in per.items()}
    out["total"] = sum(out.values())
    return out
