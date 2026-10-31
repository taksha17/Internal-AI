"""Report whether a machine can hold the pinned base, and in which dtype.

Turing cards advertise bf16 through emulation, so asking ``is_bf16_supported()``
plainly answers the wrong question: it is asked here with emulation excluded,
and a real allocation and matmul are timed rather than assumed.  Exits zero
whatever happens, because a failed probe is a recorded result, not a crash.
"""

from __future__ import annotations

import json
import platform
import sys
import time

BASE_BYTES = 8_044_982_000  # the three pinned shards of Qwen/Qwen3-4B
GIB = 1024**3


def first_line(exc: BaseException) -> str:
    lines = str(exc).strip().splitlines()
    return lines[0][:160] if lines else type(exc).__name__


def bf16_is_native(torch: object) -> bool | str:
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        return "unknown: torch predates including_emulation"


def allocate(torch: object, nbytes: int) -> str:
    try:
        torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    except (RuntimeError, MemoryError) as exc:
        return f"oom: {first_line(exc)}"
    torch.cuda.empty_cache()
    return "fits"


def matmul_ms(torch: object, dtype: object) -> str:
    try:
        left = torch.randn(1024, 1024, dtype=dtype, device="cuda")
        right = torch.randn(1024, 1024, dtype=dtype, device="cuda")
        product = left @ right
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(3):
            product = left @ right
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) / 3
        finite = bool(torch.isfinite(product).all())
    except (RuntimeError, MemoryError) as exc:
        return f"failed: {first_line(exc)}"
    finally:
        torch.cuda.empty_cache()
    return f"{elapsed * 1000:.1f}ms finite={finite}"


def main() -> int:
    import torch

    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }

    verdict = "no-cuda"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        native = bf16_is_native(torch)
        fits_bf16 = allocate(torch, BASE_BYTES)
        fits_fp32 = allocate(torch, BASE_BYTES * 2)
        report |= {
            "device": props.name,
            "capability": f"{props.major}.{props.minor}",
            "vram_gib": round(total / GIB, 2),
            "vram_free_gib": round(free / GIB, 2),
            "bf16_native": native,
            "base_fits_bf16": fits_bf16,
            "base_fits_fp32": fits_fp32,
            "matmul_bf16": matmul_ms(torch, torch.bfloat16),
            "matmul_fp16": matmul_ms(torch, torch.float16),
            "matmul_fp32": matmul_ms(torch, torch.float32),
        }
        if fits_fp32 == "fits":
            verdict = "fp32"
        elif fits_bf16 == "fits" and native is True:
            verdict = "bf16"
        elif fits_bf16 == "fits":
            verdict = "fp16"
        else:
            verdict = "no-fit"

    print(json.dumps(report, indent=2, default=str))
    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
