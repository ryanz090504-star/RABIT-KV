# KVQuant — INT3 KV Cache Quantization for vLLM

包含两套项目: **量化评估框架** (`Quantization_v2`) + **vLLM Fork** (`vllm-kvquant`，基于 f329ce4，分支 `kvquant-k3-local`)

---

## 快速安装

```bash
pip install -e Quantization_v2/
# vLLM fork 需要从完整 git 目录安装:
pip install -e vllm-kvquant/ --torch-backend=auto
```

## 环境验证

```bash
bash Quantization_v2/scripts/check_env.sh
```

## 评测运行

```bash
# 单策略完整管道 (Quality + Kernel + Memory + Tasks)
bash Quantization_v2/scripts/run_single_pipeline.sh

# 全矩阵实验 (8 策略 × 4 bit-width)
bash Quantization_v2/scripts/run_full_matrix.sh

# A100 上的当前最佳结果 (kvquant_int3 @ 3bit):
# PPL=69.4, 4.0x 压缩, 有效 3.62 bits, Needle 任务 100% 恢复
# 完整报告: Quantization_v2/reports/paper_single_A100_final.md
```

## 添加新量化算法

在 `Quantization_v2/kvquant/policies.py` 中:

```python
class MyPolicy(KVQuantPolicy):
    name = "my_policy"
    nbits: int = 4

    def quantize(self, keys: np.ndarray, values: np.ndarray,
                 context: QuantizationContext) -> QuantizedKVBlock:
        k = UniformQuantizedArray.from_float(keys, self.nbits, axis_strategy="per_head")
        v = UniformQuantizedArray.from_float(values, self.nbits, axis_strategy="per_head")
        return QuantizedKVBlock(key=k, value=v, layer_idx=context.layer_idx)
```

注册到 `_POLICIES` 字典:
```python
_POLICIES = {
    ...
    "my_policy": MyPolicy,
}
```

然后评测:
```bash
kvq quality --model TinyLlama/TinyLlama_v1.1 --policy my_policy --nbits 4 \
    --text-file data/wikitext2_test.txt --max-tokens 128 --num-windows 2
```

## vLLM Fork 改动 (6 files modified, 2 new)

| 文件 | 改动 |
|------|------|
| `vllm/config/cache.py` | 注册 `kvquant_k3` CacheDType |
| `vllm/utils/torch_utils.py` | `kvquant_k3` → uint8, per-token-head scales |
| `vllm/v1/kv_cache_interface.py` | INT3_PER_TOKEN_HEAD 量化模式, packed dim 计算 |
| `vllm/v1/attention/backends/triton_attn.py` | shape 含 INT3 packed dim + scale padding |
| `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` | INT3 cache write path dispatch |
| `vllm/v1/attention/ops/triton_unified_attention.py` | INT3 attention read path dispatch |
| `vllm/v1/attention/ops/kvquant_k3.py` | **新** — 完整的 INT3 Triton kernel (806 lines) |
| `tests/quantization/test_kvquant_k3.py` | **新** — 15 个单元测试 (pack/unpack/kernel smoke) |

基线 commit: `f329ce4` — [ROCm][CI][Bugfix] Use VllmRunner for voxtral_realtime tests

## 已知问题

- **turbo_int3** PPL=2974 (不收敛) — `ResidualSignQuantizedArray` 的 QJL residual 逻辑需要调试
- **kernel latency 不是真实部署加速** — 当前 PyTorch reference path (`torch_unpack_dequant_attention`)，Triton kernel 需要接入评测管线
- model download 在大陆服务器需要 `export HF_ENDPOINT=https://hf-mirror.com`

## 评测维度

| 命令 | 指标 |
|------|------|
| `kvq quality` | PPL, MSE/RMSE, cosine, SNR, KL divergence, top-1/5 accuracy, per-layer error |
| `kvq latency` | TPOT, quant/dequant/attention 分段计时, baseline comparison |
| `kvq memory` | 压缩比, effective bits, memory breakdown (packed/scale/codebook/QJL/page) |
| `kvq tasks` | Exact match, accuracy recovery (Needle/LongBench/RULER) |
| `kvq report` | Markdown 报告 + Policy Comparison Matrix |
| `kvq validate` | 论文证据审计 checklist |
