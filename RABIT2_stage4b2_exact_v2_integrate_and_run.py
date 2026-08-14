from __future__ import annotations

import base64, os, shutil, subprocess, sys, tempfile, time, zipfile, zlib
from pathlib import Path

ROOT = Path.cwd().resolve()
VLLM = ROOT / "vllm-kvquant"
RABIT = VLLM / "vllm/v1/attention/ops/rabit_kv2.py"
TEST = VLLM / "tests/quantization/test_rabit_kv2_stage4b2_fixed.py"
BACK = ROOT / "rabit2_stage4b2_fixed_backup" / "rabit_kv2.py"
SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4b2_fixed_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4b2_fixed.py"
LOG = ROOT / "rabit2_stage4b2_fixed.log"

P = 'eNqdVm1r20gQ/q5fMTQfTgJZieVQgok+OI0bzHFJcNxSrhSxllb2Elmr7q4S+7gff7MvlqXUCekZDNLus7Mzz8w8I+8EkiSB+eRqtojTh8XkZnp+FafTb5NPi/RrnH6efZtep1fTm9mtBnon8KDIigKCgG5JpuBrDAWRCmqi1pF3gojZX/d388XkdjEGxUW2jjK+qVlJ/Q3PafJB0LzJ6IA/UbGmJP8QQCOphE9friewEqReywjmVDWiojlaU7SSXEjISAU1Z5UC/HOLHGxIhd7kwBtVNwqWTVFQxJIqhyUFfcWzYApNwHIHak3RXkW32sQTz4hivIps7FDjOSaVxDiyR5qfblh1KjNS0tYBVgHDfclymhERguTwrA1mJa+o9cf5IfHygguqb9wBRTO1fmYSiqbKzK1e+jblkMBCNNRLfzakUuwfmj7FaS3YhohdKmiRSp2G8+UwtVlI4DWk53k5LSAVZMlUvD9nHpRML7bD+MJ/ImVD99lamHiDsQf426JlsxsVJSfKD8zqSvCmRtoT2EaCyjXG5w9DuAjhPIRRbEE/U+QQIQ4cEXz1c7ZJBgh9pLTWzzrIFk62PTjZvgG3uUnAt+cG9roATmEUnfUQNqrnNRXUN2sRWUo/gEsY0gG6bPcxhzIt2aPDBKE9by8TphxhG9pbsKCR6VXDG7TjgL211zjXteUoz7AX5FHKM/TZ7L6gdhQjvT1/fPNiznyPoiiEsx/tyr/gu8XhD7i8BJeU/lZsts6PbY3M1ke7FUSK+9bVBrvvIngZ7SvV5Ro/b7OwVwJj9a2SDA2iaMrSNJZJvF07LiKhF3jH6f5NHw4p+j8uuLy/0or2yiPNFh66KpV4bqw1DlpZvpnffblPH2Z/T/EOWySs6GTfNihDvpuctKtaBO1OrnYoQMmegKVp5eHHHlI1NXJi8abqAn3AlR6yEfTQ6J5/cNdAR7EtlXEL3LdLavhzfWJftF69XTLWFWR0b+3EGnOiTFBdnWRTpMqJMmgbNIJrqki21vK7gULwjZH+gyE9agZHB0jNeQlX089386mZEY+sWmHAHM8LaCvJHI1ag3uh6wQbmaHgH0jbi1GHgxbT8exP1DmXJWQXiabonRibKtw5B7RLTOE8xLGZuyFsh9ZguVPdOHPBCgW8AsUo4nHKHpx+l4QbadFS1DaP8cr396cH3aC1+HbiQ5Eoyab2z9DeLwLSDfreOA/aeWkSS0qcrfvs5m50g2iwpzbUDF47c01+MXPAi445jdV82DT9IVEtG0mWSH/vK+FAhSXvWEn2BOTQby0tL5jrtUnwa36dbNv7XHO4tjCHva64v3Pyv1CBsKMiyeHRtqb3ro+96e21+dT7D22YIgU='
T = 'eNrFVd1vmzAQf+evsNjDQMtQoFnTToq09Gvqw17WbqpUVdYBhlgBm9qGJvvrdwaSpl9pV1UdDwb77n6+u9/d4WRKloTSrDa1YpQSXlZSGQJCSAOGS6Edpz+rloZps9oZqZKZ4zjfuuOgBDUP9JxXPPPQtpMHSZ1CwDWFBngBccE8f0AUAy3FxD38dTTFzXXNFUtd30lZRiwW1QZyNoojmvEFSylbQGIE05rmYBDhq0Pw6f1oiqIMmjAAY5iwDgey0oGCmBs6byICmiinNQCtGRrkDAMzylMD4tKf04PT84ienU+/H48OInp8MT08p78jenJ6cXzkDsgJFJr5rf0NqJJM+sAUiFR4XjggewMSRnsYVmqWFZt04jgrJJhwF09ZwxM2cW0m3A5IBfS6BnT2D6NNRCvFMXlLqljm2Ts6pY386aVIZkoK1Pf8LpZMKqIZSwkXpODaeOhPzqw70Zex75NP5DIc4m48HuPRMEJHRsP93asudbf4JYgaCmqhPLv4a/niTUK1D6sGhJUCF53gkiLw0xlYcR92rHuLW5wccXKLk1ucfDvOpuGTqVwp9JVhXZu0wBMb531xB4PlCoVnY8orzIRN2lbF1uFSvES1DUsnK9V36S1Zm6o2mupaNbxh+I41WmEr0USWFS9QJ4Gi0Ku2c133CDETg+i5wqbElmvL0cwYsbd+zhVUMyIbpm4UNwwFYEis5JyRs/bygyhAlNc18cOyDcOwY3IRvknJVmhVClx0ggt9psr6uz+QMwGVniE5FVMaOxLjIH1ygx6Xamja8q/CICmkWJUg3rYW4fcdGTqxluH3Wtbf+gPEkhQ4FRVpWSJlrQ2xNVLWOOZYy0rGFR5a+UfMI8NJLxAPPdRS9c5tzpN+lIQRpm409P/HyNje186zTf1Id1laVxQ8qdTyvqbjSbW2MtbM+O/TqRnyF0MypzVGaylKX/0nfEjk/n7fQ/cpxCIYDcju6F8YhNdM+PhF43wr67aI+W0F72yU7iMkwiW/GpAY1zs8w+XOlf0BxPh2/gKmMOnm'
R = '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'
MARK = "# === RABIT2_STAGE4B2_EXACT_V2_FIXED_BEGIN ==="
OLD_BAD_MARK = "# === RABIT2_STAGE4B2_EXACT_V2_BEGIN ==="
IGNORE = {
    ".git", ".github", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "build", "dist", ".venv", "venv", "node_modules"
}

def dec(s):
    return zlib.decompress(base64.b64decode(s)).decode("utf-8")

def install():
    if not RABIT.is_file():
        raise FileNotFoundError(RABIT)
    text = RABIT.read_text(encoding="utf-8")
    if "RABIT2_STAGE4B1_EXACTMETA_BEGIN" not in text:
        raise RuntimeError("Stage4B1 exactmeta is required")

    # The previous installer restores Stage4B1 automatically on failure. Refuse
    # to stack on top of an un-restored failed Stage4B2 block.
    if OLD_BAD_MARK in text:
        raise RuntimeError(
            "Old failed Stage4B2 block is still present. The previous run should "
            "have restored Stage4B1; do not stack patches."
        )

    BACK.parent.mkdir(parents=True, exist_ok=True)
    if not BACK.exists():
        shutil.copy2(RABIT, BACK)

    if MARK not in text:
        text = text.rstrip() + "\n\n" + dec(P).strip() + "\n"
        compile(text, str(RABIT), "exec")
        RABIT.write_text(text, encoding="utf-8")
        print("Installed Stage4B2 persistent-safe exact-V2 path")
    else:
        print("Stage4B2 fixed patch already present.")

    TEST.parent.mkdir(parents=True, exist_ok=True)
    TEST.write_text(dec(T), encoding="utf-8")
    compile(TEST.read_text(encoding="utf-8"), str(TEST), "exec")
    print("Stage 4B2 fixed local source preflight: PASSED")

def include(p):
    rel = p.relative_to(VLLM)
    return (
        p.is_file()
        and not any(x in IGNORE for x in rel.parts)
        and p.suffix.lower() not in {".pyc", ".pyo", ".log"}
    )

def snapshot():
    tmp = SNAP.with_suffix(".zip.tmp")
    for p in (tmp, SNAP):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(VLLM.rglob("*")):
            if include(p):
                z.write(p, p.relative_to(VLLM).as_posix())
    os.replace(tmp, SNAP)
    print("Created frozen snapshot:", SNAP)

def restore():
    if BACK.is_file():
        shutil.copy2(BACK, RABIT)
        print("Verification failed; restored pre-Stage4B2 rabit_kv2.py")

def run():
    RUNNER.write_text(dec(R), encoding="utf-8")
    time.sleep(2)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, "-m", "modal", "run", str(RUNNER)]
    print("Running:", " ".join(cmd))
    with LOG.open("w", encoding="utf-8") as log:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return p.wait()

def main():
    print("RABIT-2 Stage 4B2 fixed exact-V2 integration")
    install()
    snapshot()
    code = run()
    if code:
        restore()
        raise SystemExit(code)
    print("Log saved to:", LOG)

if __name__ == "__main__":
    main()
