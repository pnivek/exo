#!/usr/bin/env python3
"""Comprehensive benchmark matrix for exo disaggregated inference.

Uses /bench/chat/completions (non-streaming) for accurate server-side metrics:
  - prompt_tps     : prefill throughput measured by the runner
  - generation_tps : decode throughput measured by the runner
  - peak_memory_gb : peak GPU memory
  - reasoning_tokens / content_tokens breakdown

Uses /v1/chat/completions (streaming) for client-side TTFT.

Usage:
  # Single config (start cluster first, place instance):
  python3 scripts/benchmark_matrix.py --config disagg --api http://192.168.0.100:52415

  # Full matrix + combine into markdown:
  python3 scripts/benchmark_matrix.py --config mac    --api http://192.168.0.100:52415 --output /tmp/bench_mac.json
  python3 scripts/benchmark_matrix.py --config spark  --api http://192.168.0.112:52415 --output /tmp/bench_spark.json
  python3 scripts/benchmark_matrix.py --config disagg --api http://192.168.0.100:52415 --output /tmp/bench_disagg.json
  python3 scripts/benchmark_matrix.py --config tp-disagg --api http://192.168.0.100:52415 --output /tmp/bench_tp_disagg.json
  python3 scripts/benchmark_matrix.py --combine /tmp
"""

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "mlx-community/gpt-oss-20b-MXFP4-Q8"
DEFAULT_PROMPT_LENGTHS = [512, 2048, 8192, 16384, 32768]
DEFAULT_TRIALS = 3
DEFAULT_MAX_TOKENS = 500  # thinking models need budget for reasoning tokens

# Config-aware skip/warn rules.  tp-disagg 32K may OOM on Mac.
SKIP_RULES: dict[str, set[int]] = {
    "tp-disagg": {32768},
}


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

_PASSAGES = [
    "The quick brown fox jumps over the lazy dog near the riverbank on a sunny afternoon. ",
    "In a distant galaxy far beyond the reach of any telescope, stars are born and die. ",
    "Quantum computing promises to revolutionize the way we process and store information. ",
    "The ancient city rose from the desert sands as archaeologists uncovered its secrets. ",
    "Machine learning models learn patterns from data to make predictions about the future. ",
]


def generate_prompt(target_tokens: int, trial_idx: int) -> str:
    """Generate a unique prompt of approximately target_tokens tokens."""
    base = _PASSAGES[trial_idx % len(_PASSAGES)]
    reps = max(1, (target_tokens * 4) // len(base))
    text = (base * reps)[: target_tokens * 4]
    suffix = f" [Benchmark trial {trial_idx}, target ~{target_tokens} tokens. Respond briefly.]"
    return text + suffix


# ---------------------------------------------------------------------------
# Single trial runner
# ---------------------------------------------------------------------------


def run_bench_trial(
    api_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: int,
) -> dict[str, Any] | None:
    """Call /bench/chat/completions and return server-side metrics."""
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "bench": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{api_url}/bench/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read())
    except Exception as exc:
        print(f"    [bench] ERROR: {exc}")
        return None

    stats = body.get("generation_stats") or {}
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    peak_mem = stats.get("peak_memory_usage") or {}

    return {
        "server_prefill_tps": stats.get("prompt_tps"),
        "server_decode_tps": stats.get("generation_tps"),
        "server_prompt_tokens": stats.get("prompt_tokens")
        or usage.get("prompt_tokens"),
        "server_gen_tokens": stats.get("generation_tokens")
        or usage.get("completion_tokens"),
        "server_reasoning_tokens": details.get("reasoning_tokens"),
        "peak_memory_gb": (peak_mem.get("in_bytes") or 0) / 1024 / 1024 / 1024
        if peak_mem.get("in_bytes")
        else None,
        "finish_reason": (body.get("choices") or [{}])[0].get("finish_reason"),
    }


def run_streaming_trial(
    api_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: int,
) -> dict[str, Any] | None:
    """Call /v1/chat/completions with stream=True and measure client-side TTFT."""
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{api_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t_start = time.monotonic()
    t_first: float | None = None
    token_count = 0
    reasoning_count = 0
    content_count = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            buf = b""
            done = False
            while not done:
                chunk = resp.read(1)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line or not line.startswith(b"data: "):
                        continue
                    data = line[6:]
                    if data == b"[DONE]":
                        done = True
                        break
                    try:
                        obj = json.loads(data)
                        delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                        content = delta.get("content") or ""
                        reasoning = delta.get("reasoning_content") or ""
                        if content or reasoning:
                            if t_first is None:
                                t_first = time.monotonic()
                            token_count += 1
                            if content:
                                content_count += 1
                            if reasoning:
                                reasoning_count += 1
                    except json.JSONDecodeError:
                        pass
    except Exception as exc:
        print(f"    [stream] ERROR: {exc}")
        return None

    t_end = time.monotonic()
    if t_first is None or token_count == 0:
        print(f"    [stream] No tokens ({(t_end - t_start) * 1000:.0f}ms)")
        return None

    ttft_ms = (t_first - t_start) * 1000
    gen_time = t_end - t_first
    tps = token_count / gen_time if gen_time > 0 else 0.0

    return {
        "client_ttft_ms": ttft_ms,
        "client_tps": tps,
        "client_tokens": token_count,
        "client_reasoning_tokens": reasoning_count,
        "client_content_tokens": content_count,
    }


# ---------------------------------------------------------------------------
# Compute per-length timeout
# ---------------------------------------------------------------------------


def compute_timeout(prompt_tokens: int) -> int:
    # Generous budget: 120s base + 1s per 10 prompt tokens + 300s for reasoning
    return max(300, prompt_tokens // 10 + 300)


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------


def run_config(
    config: str,
    api_url: str,
    model: str,
    prompt_lengths: list[int],
    trials: int,
    max_tokens: int,
    skip_prompts: set[int],
) -> dict[str, Any]:
    """Run all prompt lengths and trials for a config."""
    results: dict[str, Any] = {
        "config": config,
        "model": model,
        "trials_requested": trials,
        "max_tokens": max_tokens,
        "prompt_lengths": {},
    }

    # Warmup: one bench call at the shortest prompt length
    warmup_len = prompt_lengths[0]
    print(f"  Warming up with {warmup_len}-token prompt...")
    run_bench_trial(api_url, model, generate_prompt(warmup_len, 99), max_tokens, 300)
    time.sleep(2)

    for pl in prompt_lengths:
        if pl in skip_prompts:
            print(f"  Prompt {pl:>6} — SKIPPED (config rule)")
            results["prompt_lengths"][str(pl)] = {"skipped": True}
            continue

        timeout = compute_timeout(pl)
        print(f"  Prompt {pl:>6} tokens ({trials} trials, timeout={timeout}s)")

        bench_results: list[dict[str, Any]] = []
        stream_results: list[dict[str, Any]] = []

        for trial in range(trials):
            prompt = generate_prompt(pl, trial)
            prompt_b = generate_prompt(pl, trial + 100)  # different prompt for stream

            # Bench call
            b = run_bench_trial(api_url, model, prompt, max_tokens, timeout)
            if b:
                bench_results.append(b)
                tps_str = (
                    f"{b['server_decode_tps']:.1f}"
                    if b.get("server_decode_tps")
                    else "?"
                )
                ptps_str = (
                    f"{b['server_prefill_tps']:.1f}"
                    if b.get("server_prefill_tps")
                    else "?"
                )
                mem_str = (
                    f"{b['peak_memory_gb']:.1f}GB" if b.get("peak_memory_gb") else "?"
                )
                rtoks = b.get("server_reasoning_tokens") or 0
                gtoks = b.get("server_gen_tokens") or 0
                print(
                    f"    T{trial + 1} bench: prefill={ptps_str} tps  decode={tps_str} tps  "
                    f"mem={mem_str}  gen={gtoks}tok (reason={rtoks})"
                )
            else:
                print(f"    T{trial + 1} bench: FAILED")

            time.sleep(1)

            # Streaming call for TTFT
            s = run_streaming_trial(api_url, model, prompt_b, max_tokens, timeout)
            if s:
                stream_results.append(s)
                print(
                    f"    T{trial + 1} stream: TTFT={s['client_ttft_ms']:.0f}ms  "
                    f"TPS={s['client_tps']:.1f}  tokens={s['client_tokens']}"
                )
            else:
                print(f"    T{trial + 1} stream: FAILED")

            time.sleep(2)

        # Aggregate
        def avg(vals: list[float | None]) -> float | None:
            clean = [v for v in vals if v is not None]
            return statistics.mean(clean) if clean else None

        def safe_min(vals: list[float | None]) -> float | None:
            clean = [v for v in vals if v is not None]
            return min(clean) if clean else None

        def safe_max(vals: list[float | None]) -> float | None:
            clean = [v for v in vals if v is not None]
            return max(clean) if clean else None

        pl_result: dict[str, Any] = {
            "successful_bench_trials": len(bench_results),
            "successful_stream_trials": len(stream_results),
            "skipped": False,
        }

        if bench_results:
            pl_result["server_prefill_tps"] = {
                "mean": avg([r.get("server_prefill_tps") for r in bench_results]),
                "min": safe_min([r.get("server_prefill_tps") for r in bench_results]),
                "max": safe_max([r.get("server_prefill_tps") for r in bench_results]),
            }
            pl_result["server_decode_tps"] = {
                "mean": avg([r.get("server_decode_tps") for r in bench_results]),
                "min": safe_min([r.get("server_decode_tps") for r in bench_results]),
                "max": safe_max([r.get("server_decode_tps") for r in bench_results]),
            }
            pl_result["server_gen_tokens"] = {
                "mean": avg([r.get("server_gen_tokens") for r in bench_results]),
            }
            pl_result["server_reasoning_tokens"] = {
                "mean": avg([r.get("server_reasoning_tokens") for r in bench_results]),
            }
            pl_result["peak_memory_gb"] = {
                "mean": avg([r.get("peak_memory_gb") for r in bench_results]),
                "max": safe_max([r.get("peak_memory_gb") for r in bench_results]),
            }

        if stream_results:
            pl_result["client_ttft_ms"] = {
                "mean": avg([r.get("client_ttft_ms") for r in stream_results]),
                "min": safe_min([r.get("client_ttft_ms") for r in stream_results]),
                "max": safe_max([r.get("client_ttft_ms") for r in stream_results]),
            }
            pl_result["client_tps"] = {
                "mean": avg([r.get("client_tps") for r in stream_results]),
            }

        results["prompt_lengths"][str(pl)] = pl_result
        print()

    return results


# ---------------------------------------------------------------------------
# Combine mode: markdown tables
# ---------------------------------------------------------------------------


def _get(results: dict[str, Any], pl: int, metric: str) -> str:
    pl_data = results.get("prompt_lengths", {}).get(str(pl), {})
    if pl_data.get("skipped"):
        return "SKIP"
    m = pl_data.get(metric)
    if m is None:
        return "—"
    if isinstance(m, dict):
        v = m.get("mean")
        if v is None:
            return "—"
        if "ms" in metric:
            return f"{v:.0f}ms"
        if "gb" in metric.lower():
            return f"{v:.1f}GB"
        return f"{v:.1f}"
    return f"{m:.1f}"


def combine(result_dir: Path) -> None:
    """Load all bench_*.json files and print a comparison markdown table."""
    config_order = ["mac", "spark", "disagg", "tp-disagg"]
    all_results: dict[str, dict[str, Any]] = {}

    for path in result_dir.glob("bench_*.json"):
        with open(path) as f:
            data = json.load(f)
        cfg = data.get("config", path.stem.replace("bench_", ""))
        all_results[cfg] = data

    if not all_results:
        print("No bench_*.json files found")
        return

    configs = [c for c in config_order if c in all_results] + [
        c for c in all_results if c not in config_order
    ]

    # Collect all prompt lengths
    all_pls: set[int] = set()
    for data in all_results.values():
        for pl_str in data.get("prompt_lengths", {}):
            all_pls.add(int(pl_str))
    prompt_lengths = sorted(all_pls)

    model = next(iter(all_results.values())).get("model", "unknown")
    print(f"\n## Benchmark Results — {model}\n")

    def table(title: str, metric: str, fmt: str = "") -> None:
        print(f"### {title}\n")
        header = "| Prompt |" + "".join(f" {c:>12} |" for c in configs)
        sep = "|--------|" + "".join("-------------|" for _ in configs)
        print(header)
        print(sep)
        for pl in prompt_lengths:
            row = f"| {pl:>6} |"
            for c in configs:
                row += f" {_get(all_results.get(c, {}), pl, metric):>12} |"
            print(row)
        print()

    table("Prefill TPS (server-side, tokens/sec)", "server_prefill_tps")
    table("Decode TPS (server-side, tokens/sec)", "server_decode_tps")
    table("TTFT — client-side (ms)", "client_ttft_ms")
    table("Peak GPU Memory (GB)", "peak_memory_gb")
    table("Generated Tokens (mean)", "server_gen_tokens")
    table("Reasoning Tokens (mean)", "server_reasoning_tokens")

    print("\n### Notes\n")
    print(
        "- Prefill / Decode TPS measured by the runner (server-side), equivalent to mlx_lm.benchmark quality"
    )
    print("- TTFT measured client-side (includes network and orchestration overhead)")
    print("- `bench=True` disables KV prefix cache for clean per-trial measurements")
    print("- Thinking model (gpt-oss-20b): most generated tokens are reasoning_content")
    print("- SKIP = config rule (tp-disagg 32K may OOM on Mac during all_gather)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark exo disaggregated inference configurations"
    )
    parser.add_argument(
        "--config",
        choices=["mac", "spark", "disagg", "tp-disagg"],
        help="Config to benchmark (cluster must already be running with correct instance placed)",
    )
    parser.add_argument("--api", help="API URL e.g. http://192.168.0.100:52415")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--prompt-lengths",
        type=lambda s: [int(x) for x in s.split(",")],
        default=DEFAULT_PROMPT_LENGTHS,
        metavar="CSV",
        help="Comma-separated prompt lengths e.g. 512,2048,8192",
    )
    parser.add_argument(
        "--output",
        help="Output JSON path (default: /tmp/bench_{config}.json)",
    )
    parser.add_argument(
        "--combine",
        metavar="DIR",
        help="Combine all bench_*.json in DIR into a markdown table and exit",
    )
    args = parser.parse_args()

    if args.combine:
        combine(Path(args.combine))
        return

    if not args.config:
        parser.error("--config is required unless --combine is used")
    if not args.api:
        parser.error("--api is required unless --combine is used")

    skip = SKIP_RULES.get(args.config, set())
    output = args.output or f"/tmp/bench_{args.config.replace('-', '_')}.json"

    print("=" * 65)
    print(f"Benchmarking config: {args.config}")
    print(f"API: {args.api}  Model: {args.model}")
    print(f"Trials: {args.trials}  Max tokens: {args.max_tokens}")
    print(f"Prompt lengths: {args.prompt_lengths}")
    if skip:
        print(f"Skipping: {sorted(skip)} (config rule)")
    print("=" * 65)
    print()

    results = run_config(
        config=args.config,
        api_url=args.api.rstrip("/"),
        model=args.model,
        prompt_lengths=sorted(args.prompt_lengths),
        trials=args.trials,
        max_tokens=args.max_tokens,
        skip_prompts=skip,
    )

    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output}")

    # Print quick summary
    print(f"\n## {args.config} — summary")
    print("| Prompt | Prefill TPS | Decode TPS | TTFT (ms) | Mem (GB) |")
    print("|--------|------------|-----------|----------|---------|")
    for pl in sorted(args.prompt_lengths):
        pl_data = results["prompt_lengths"].get(str(pl), {})
        if pl_data.get("skipped"):
            print(f"| {pl:>6} | SKIP        | SKIP       | SKIP      | SKIP     |")
            continue

        def _mean(d: dict[str, Any], key: str) -> str:
            m = d.get(key)
            if not m:
                return "—"
            v = m.get("mean")
            return f"{v:.1f}" if v is not None else "—"

        def _ttft(d: dict[str, Any], key: str) -> str:
            m = d.get(key)
            if not m:
                return "—"
            v = m.get("mean")
            return f"{v:.0f}" if v is not None else "—"

        print(
            f"| {pl:>6} | {_mean(pl_data, 'server_prefill_tps'):>11} "
            f"| {_mean(pl_data, 'server_decode_tps'):>9} "
            f"| {_ttft(pl_data, 'client_ttft_ms'):>8} "
            f"| {_mean(pl_data, 'peak_memory_gb'):>7} |"
        )


if __name__ == "__main__":
    main()
