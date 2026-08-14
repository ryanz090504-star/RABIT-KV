from __future__ import annotations

import base64,os,shutil,subprocess,sys,tempfile,time,zipfile,zlib
from pathlib import Path

ROOT=Path.cwd().resolve()
VLLM=ROOT/"vllm-kvquant"
RABIT=VLLM/"vllm/v1/attention/ops/rabit_kv2.py"
TRITON=VLLM/"vllm/v1/attention/backends/triton_attn.py"
TEST=VLLM/"tests/quantization/test_rabit_kv2_stage4b4_causal_prefill.py"

BACK=ROOT/"rabit2_stage4b4_backup"
SNAP=Path(tempfile.gettempdir())/"rabit2_stage4b4_integrated_snapshot.zip"
RUNNER=Path(tempfile.gettempdir())/"modal_rabit2_stage4b4_integrated.py"
LOG=ROOT/"rabit2_stage4b4_integrated.log"

P='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'
T='eNq9Vk1v4zYQvftXED5RCKuNHMd1DRhokZ6LYhc9GQZBSaOYEU3JIiUk++s7JGV9pG7TxRY1bJMiZ94MH2eevSia6kw4L1rbNsA5kee6aiwRWldWWFlps1j0a/WbBWOvT7ZqstNiscihIFwoVWW0YbhagjaMnPZbRvJ9stpGuwXBF3rsSRN/Fqm0qy9SPyv4ApcWdAafW23lGegJPfAdeftaPINBl7N4pcmGERqQyR15SCLy6RN5WOE82QTrTGQnQGufVPwVmspQv+Fe1GMxklzfjY2VeKtaG7sdnrpzRWywz+1bDfsA1Uptt5Mt6GQG+2XW5mIZlkMCqR2ii0boZ7gGnYIh1sOKzUGCewNIv8bEWDgKQ8Art0acgRu8DKDCbeMeDmnPqzAG3H3FmaoM5LznbU/S2Yq3LaqGaEQjUpORnmVVg+blkvWzDj2yEvLJwlnqyZPJhILlSMqygQy0DRD9vLuysxvMXjHz1xRpegY8jG3caVw2ERtW0n5l8JEFuhFpyG+VBoLpI0D/NAJPaJgYC51PrAPw4ALK3AYIF4WFKRQNCUcTV0xnTvSIccM/E4fd3PzoLu6wm1/NMcKb/jk0V3wWTRmbUtayoNiBPZ6rlFgaLjohlUgVUMyqAWEqvV8+/fHrL/hwaWWDlxbNoWqsxjPYRn6FcOPLuoFCvrLs1Oqyv6QDvWfbiNEH/70O8yQMjwx7jNHNPdus3bgO40+bMCarH93kyBaRr1YX2RXrM6zTNc9Ea4TitRKa+/7l8CoyS/sciE+iL5FeVjqlznGXxFgPWEeoP3FVm7hxssHLboU8k2bhHQI1Z6GRa24AcrpN7u/vURUCPE4CvrcO4nI5Yf2hMqEuMVSQHscKhctzL79VDk2NPZ1rSr0tO7E8mnd2WqhKWKdTN5q7m6NwJUugfVqX9xF8cHb5hhC9gBRIp/9OLQ4IOxVlzDrEK4RxGhOG1Lrxlum12gMnY5XjYyxqlIGclodd2D2ybpz6FHwGYxO7IB94+XxCOn1wVzOTX4wnX0lPjpzfcYfOwBkpD32m2GHdZD47aC/XgxS+OB0MYj0txHfnnKnENc7dy+46uUuObGbT/QsbT9NsxXM2Lo3sOSJcLuoN+wpq+jJuTX8anLdH9SfOJkx6dfJU+i5a8UorqYHnkFU58KHROKqErfT8wJfDy86nP14sw8+NPNPvDzGpAjZc2DzKRA/Nm85OTaWdtEX/pMOCORH37UOpiMMk+iG9zqJYpIZGsfurEf1fYrx0v6iHhK3ZI0Otxf8FD05SmRPU1ePmeEtO01aVXGpppRNVV1tK8UJqp3+uCHp11f+hoq5QUe/0VUJz/FeH0hm0cT/TLR1U8aZi3dLE/d8p4lDIvtj2gzTpiYBNK2VmsXgvUqz7iyQNRerpDIY9cx7W+8wE6cPC+7AR/wSh83c4'
R='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'
MARK="# === RABIT2_STAGE4B4_CAUSAL_PREFILL_BEGIN ==="
TRITON_MARK="# === RABIT2_STAGE4B4_CAUSAL_PREFILL_TRITON_BEGIN ==="
IGNORE={".git",".github","__pycache__",".pytest_cache",".mypy_cache",
         ".ruff_cache","build","dist",".venv","venv","node_modules"}

def dec(s):
    return zlib.decompress(base64.b64decode(s)).decode("utf-8")

OLD_IMPORT="""from vllm.v1.attention.ops.rabit_kv2 import (
    Rabit2SingleSequenceRuntime,
    rabit2_get_active_batch,
    rabit2_online_decode_attention_triton,
    rabit2_register_attention_impl,
)"""

NEW_IMPORT="""from vllm.v1.attention.ops.rabit_kv2 import (
    Rabit2CausalChunkPlan,
    Rabit2SingleSequenceRuntime,
    rabit2_bulk_append_exact,
    rabit2_get_active_batch,
    rabit2_online_decode_attention_triton,
    rabit2_register_attention_impl,
)"""

OLD_INITIAL="""            if context_len == 0 and q_len > 1:
                runtime.append(k_seq, v_seq, kv_cache, block_table_row)
                # One-sequence local metadata for dense exact initial prompt attention."""

NEW_INITIAL="""            if context_len == 0 and q_len > 1:
                # Stage4B4: dense initial-prefill attention reads k_seq/v_seq
                # directly, so the runtime can install the chunk's final exact
                # sidecar state in one bulk operation.
                rabit2_bulk_append_exact(
                    runtime, k_seq, v_seq, kv_cache, block_table_row
                )
                # One-sequence local metadata for dense exact initial prompt attention."""

OLD_LOOP="""            # Decode and non-initial chunked prefill share one exact online
            # primitive. Append before attention so the current query attends
            # to itself and all previous tokens.
            for local_idx in range(q_len):
                runtime.append(
                    k_seq[local_idx : local_idx + 1],
                    v_seq[local_idx : local_idx + 1],
                    kv_cache,
                    block_table_row,
                )
                rabit_out = rabit2_online_decode_attention_triton(
                    q_seq[local_idx : local_idx + 1],
                    kv_cache,
                    block_table_row,
                    runtime,
                    softmax_scale=self.scale,
                )
                output[q0 + local_idx : q0 + local_idx + 1].copy_(rabit_out)"""

NEW_LOOP="""            # Stage4B4: decode stays on the exact one-token append path. For a
            # non-initial chunk, precompute exact future sidecar representation
            # once, then expose only the causally legal state before each
            # attention call.
            chunk_plan = (
                Rabit2CausalChunkPlan(
                    runtime, k_seq, v_seq, kv_cache, block_table_row
                )
                if q_len > 1
                else None
            )
            for local_idx in range(q_len):
                if chunk_plan is None:
                    runtime.append(
                        k_seq[local_idx : local_idx + 1],
                        v_seq[local_idx : local_idx + 1],
                        kv_cache,
                        block_table_row,
                    )
                else:
                    chunk_plan.apply_step(local_idx)

                rabit_out = rabit2_online_decode_attention_triton(
                    q_seq[local_idx : local_idx + 1],
                    kv_cache,
                    block_table_row,
                    runtime,
                    softmax_scale=self.scale,
                )
                output[q0 + local_idx : q0 + local_idx + 1].copy_(rabit_out)"""

def backup(path):
    # New Stage4B4 test file may not exist yet on a clean tree.
    # RABIT/TRITON must exist; optional generated test files can be skipped.
    if not path.exists():
        print(f"Backup skipped (file does not exist yet): {path}")
        return
    dst=BACK/path.relative_to(ROOT)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():
        shutil.copy2(path,dst)

def install():
    if not RABIT.is_file(): raise FileNotFoundError(RABIT)
    if not TRITON.is_file(): raise FileNotFoundError(TRITON)

    rt=RABIT.read_text(encoding="utf-8")
    tt=TRITON.read_text(encoding="utf-8")

    for need in (
        "RABIT2_STAGE4B2_EXACT_V2_FIXED_BEGIN",
        "RABIT2_STAGE4B3_GQA4_BEGIN",
    ):
        if need not in rt:
            raise RuntimeError(f"Missing required RABIT stage: {need}")

    # Strict preflight BEFORE modifying either source file.
    if MARK not in rt:
        if OLD_IMPORT not in tt:
            raise RuntimeError("Stage4B4 preflight: exact triton_attn import block not found")
        if OLD_INITIAL not in tt:
            raise RuntimeError("Stage4B4 preflight: exact initial-prefill block not found")
        if OLD_LOOP not in tt:
            raise RuntimeError("Stage4B4 preflight: exact token-loop block not found")

    backup(RABIT); backup(TRITON); backup(TEST)

    if MARK not in rt:
        rt=rt.rstrip()+"\n\n"+dec(P).strip()+"\n"
        compile(rt,str(RABIT),"exec")
        RABIT.write_text(rt,encoding="utf-8")
        print("Installed Stage4B4 causal/bulk runtime helpers")
    else:
        print("Stage4B4 runtime helpers already present")

    tt=TRITON.read_text(encoding="utf-8")
    if TRITON_MARK not in tt:
        tt=tt.replace(OLD_IMPORT,NEW_IMPORT,1)
        tt=tt.replace(OLD_INITIAL,NEW_INITIAL,1)
        tt=tt.replace(OLD_LOOP,NEW_LOOP,1)
        tt=tt.rstrip()+f"\n\n{TRITON_MARK}\n# Stage4B4 triton integration active.\n# === RABIT2_STAGE4B4_CAUSAL_PREFILL_TRITON_END ===\n"
        compile(tt,str(TRITON),"exec")
        TRITON.write_text(tt,encoding="utf-8")
        print("Installed Stage4B4 triton chunked-prefill integration")
    else:
        print("Stage4B4 triton integration already present")

    TEST.parent.mkdir(parents=True,exist_ok=True)
    TEST.write_text(dec(T),encoding="utf-8")
    compile(TEST.read_text(encoding="utf-8"),str(TEST),"exec")
    print("Stage4B4 local integration preflight: PASSED")

def include(p):
    rel=p.relative_to(VLLM)
    return p.is_file() and not any(x in IGNORE for x in rel.parts) and p.suffix.lower() not in {".pyc",".pyo",".log"}

def snapshot():
    tmp=SNAP.with_suffix(".zip.tmp")
    for p in (tmp,SNAP):
        try:p.unlink()
        except FileNotFoundError:pass
    n=0
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(VLLM.rglob("*")):
            if include(p):
                z.write(p,p.relative_to(VLLM).as_posix());n+=1
    os.replace(tmp,SNAP)
    print(f"Created frozen snapshot: {SNAP} ({n} files, {SNAP.stat().st_size/1024**2:.1f} MB)")

def restore():
    for path in (RABIT,TRITON):
        src=BACK/path.relative_to(ROOT)
        if src.is_file(): shutil.copy2(src,path)
    print("Verification failed; restored pre-Stage4B4 RABIT + Triton sources")

def run():
    RUNNER.write_text(dec(R),encoding="utf-8");time.sleep(2)
    env=os.environ.copy();env["PYTHONUTF8"]="1";env["PYTHONIOENCODING"]="utf-8"
    cmd=[sys.executable,"-m","modal","run",str(RUNNER)]
    print("Running:"," ".join(cmd))
    with LOG.open("w",encoding="utf-8") as log:
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,encoding="utf-8",errors="replace",env=env,bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            print(line,end="");log.write(line);log.flush()
        return p.wait()

def main():
    print("RABIT-2 Stage 4B4 causal-prefill integration")
    print(f"Project root: {ROOT}")
    install();snapshot();code=run()
    if code:
        restore();raise SystemExit(code)
    print(f"Log saved to: {LOG}")

if __name__=="__main__":
    main()
