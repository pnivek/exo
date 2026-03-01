#!/usr/bin/env bash
# Comparison benchmark for disaggregated vs single-node inference.
# Tests 4 configurations:
#   mac      — Mac Studio M2 Ultra solo
#   spark    — DGX Spark solo
#   disagg   — PD Split: Spark prefill → Mac decode
#   tpdisagg — TP Split: Spark1+Spark2 tensor-parallel prefill → Mac decode
#
# Usage:
#   # Run all configs end-to-end (starts/stops clusters automatically):
#   ./scripts/benchmark_comparison.sh --run-all
#
#   # Run a single config (cluster must already be running with instance placed):
#   ./scripts/benchmark_comparison.sh --config mac   --api http://192.168.0.100:52415
#   ./scripts/benchmark_comparison.sh --config spark --api http://192.168.0.112:52415
#   ./scripts/benchmark_comparison.sh --config disagg   --api http://192.168.0.100:52415
#   ./scripts/benchmark_comparison.sh --config tpdisagg --api http://192.168.0.100:52415
#
#   # Combine saved results into a markdown table:
#   ./scripts/benchmark_comparison.sh --combine
#
# Options:
#   --model ID               Model to benchmark (default: mlx-community/Meta-Llama-3.1-8B-Instruct-4bit)
#   --trials N               Timed trials per depth (default: 3)
#   --mac-ssh USER@HOST      Mac SSH target (default: kevin@192.168.0.100)
#   --spark-ssh USER@HOST    Spark 1 SSH target (default: pnivek@192.168.0.112)
#   --spark2-ssh USER@HOST   Spark 2 SSH target for TP disagg (default: pnivek@192.168.0.172)
#   --mac-api URL            Mac API base URL (default: http://192.168.0.100:52415)
#   --spark-api URL          Spark 1 API base URL (default: http://192.168.0.112:52415)
#   --spark2-api URL         Spark 2 API base URL (default: http://192.168.0.172:52415)
#   --spark-ld-path PATH     LD_LIBRARY_PATH for both Sparks — $HOME expanded on remote
#   --depths "0 4096 ..."    Space-separated context depths (default: "0 4096 8192 16384 32768")
#
# llama-benchy (https://github.com/eugr/llama-benchy) handles HTTP timing,
# warmup, prompt generation, and JSON output. Install via: uvx llama-benchy --help
#
# Results are saved to /tmp/bench_{config}.json.
# Requires: curl, python3, ssh, uvx

set -euo pipefail

# ---------- Defaults ----------
MODEL_ID="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
TRIALS=3
DEPTHS="0 4096 8192 16384 32768"
RESULTS_DIR="/tmp"

MAC_SSH="kevin@192.168.0.100"
SPARK_SSH="pnivek@192.168.0.112"
SPARK2_SSH="pnivek@192.168.0.172"
MAC_API="http://192.168.0.100:52415"
SPARK_API="http://192.168.0.112:52415"
SPARK2_API="http://192.168.0.172:52415"
# Spark 1 uses /snap/bin/uv; Spark 2 has uv at ~/.local/bin/uv
SPARK_UV_CMD="/snap/bin/uv"
SPARK2_UV_CMD="~/.local/bin/uv"
# Default LD_LIBRARY_PATH for both Sparks. $HOME intentionally un-expanded here —
# it will be expanded by the remote shell when the startup script runs.
SPARK_LD_PATH='$HOME/mlx-qmm/build/lib.linux-aarch64-cpython-313/mlx/lib:$HOME/local/openblas_full/usr/lib/aarch64-linux-gnu/openblas-pthread:$HOME/local/gfortran/usr/lib/aarch64-linux-gnu:$HOME/exo/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:$HOME/exo/.venv/lib/python3.13/site-packages/nvidia/nccl/lib'

# ---------- Arg parsing ----------
CONFIG=""
API_URL=""
COMBINE=false
RUN_ALL=false
ONLY_CONFIGS="" # space-separated allow-list for --run-all; empty = all 4

while [[ $# -gt 0 ]]; do
  case "$1" in
  --configs)
    ONLY_CONFIGS="$2"
    shift 2
    ;; # e.g. --configs "mac spark"
  --config)
    CONFIG="$2"
    shift 2
    ;;
  --api)
    API_URL="$2"
    shift 2
    ;;
  --model)
    MODEL_ID="$2"
    shift 2
    ;;
  --trials)
    TRIALS="$2"
    shift 2
    ;;
  --mac-ssh)
    MAC_SSH="$2"
    shift 2
    ;;
  --spark-ssh)
    SPARK_SSH="$2"
    shift 2
    ;;
  --spark2-ssh)
    SPARK2_SSH="$2"
    shift 2
    ;;
  --mac-api)
    MAC_API="$2"
    shift 2
    ;;
  --spark-api)
    SPARK_API="$2"
    shift 2
    ;;
  --spark2-api)
    SPARK2_API="$2"
    shift 2
    ;;
  --spark-ld-path)
    SPARK_LD_PATH="$2"
    shift 2
    ;;
  --depths)
    DEPTHS="$2"
    shift 2
    ;;
  --combine)
    COMBINE=true
    shift
    ;;
  --run-all)
    RUN_ALL=true
    shift
    ;;
  *)
    echo "Unknown argument: $1"
    exit 1
    ;;
  esac
done

# ---------- Combine mode ----------
if $COMBINE; then
  python3 - "$RESULTS_DIR" "$MODEL_ID" "$TRIALS" <<'PYEOF'
import json, sys, os

results_dir, model_id, trials = sys.argv[1], sys.argv[2], sys.argv[3]
configs = {}
for name in ["mac", "spark", "disagg", "tpdisagg"]:
    path = os.path.join(results_dir, f"bench_{name}.json")
    if os.path.exists(path):
        with open(path) as f:
            configs[name] = json.load(f)

if not configs:
    print("No result files found in " + results_dir)
    sys.exit(1)

# Collect all depths across configs
all_depths = set()
for cfg in configs.values():
    for b in cfg.get('benchmarks', []):
        if not b.get('is_context_prefill_phase'):
            all_depths.add(b.get('context_size', 0))
depths = sorted(all_depths)

def find_bench(cfg, depth):
    """Find the non-prefill benchmark entry for a given context_size."""
    for b in cfg.get('benchmarks', []):
        if not b.get('is_context_prefill_phase') and b.get('context_size', 0) == depth:
            return b
    return None

def mean(metric):
    if metric is None: return None
    return metric.get('mean')

def std(metric):
    if metric is None: return None
    return metric.get('std')

def fmt(val, f=".0f"):
    return f"{val:{f}}" if val is not None else "-"

def fmt_pm(m, s, f=".0f"):
    mv = mean(m)
    sv = std(m)
    if mv is None: return "-"
    if sv is not None and sv > 0:
        return f"{mv:{f}} +/-{sv:{f}}"
    return f"{mv:{f}}"

print(f"## Comparison: {model_id} ({trials} trials avg, powered by llama-benchy)")
print()

# TTFT table
print("### TTFT — End-to-End (ms, mean +/- std)")
print("| Depth | Mac Studio | DGX Spark | PD Split | TP Split |")
print("|-------|------------|-----------|----------|----------|")
for d in depths:
    mac_b  = find_bench(configs.get('mac',      {}), d)
    spa_b  = find_bench(configs.get('spark',    {}), d)
    pd_b   = find_bench(configs.get('disagg',   {}), d)
    tp_b   = find_bench(configs.get('tpdisagg', {}), d)
    mac_v  = fmt_pm(mac_b.get('e2e_ttft'), None) if mac_b else "-"
    spa_v  = fmt_pm(spa_b.get('e2e_ttft'), None) if spa_b else "-"
    pd_v   = fmt_pm(pd_b.get('e2e_ttft'),  None) if pd_b  else "-"
    tp_v   = fmt_pm(tp_b.get('e2e_ttft'),  None) if tp_b  else "-"
    print(f"| {d:<5} | {mac_v:>10} | {spa_v:>9} | {pd_v:>8} | {tp_v:>8} |")
print()

# est_ppt table
have_ppt = any(
    find_bench(cfg, d) is not None and mean((find_bench(cfg, d) or {}).get('est_ppt')) is not None and mean((find_bench(cfg, d) or {}).get('est_ppt')) > 0
    for cfg in configs.values()
    for d in depths
)
if have_ppt:
    print("### Server-Side Prefill Time — est_ppt (ms)")
    print("| Depth | Mac Studio | DGX Spark | PD Split | TP Split |")
    print("|-------|------------|-----------|----------|----------|")
    for d in depths:
        mac_b = find_bench(configs.get('mac', {}), d)
        spa_b = find_bench(configs.get('spark', {}), d)
        pd_b  = find_bench(configs.get('disagg', {}), d)
        tp_b  = find_bench(configs.get('tpdisagg', {}), d)
        mac_v = fmt(mean((mac_b or {}).get('est_ppt')))
        spa_v = fmt(mean((spa_b or {}).get('est_ppt')))
        pd_v  = fmt(mean((pd_b  or {}).get('est_ppt')))
        tp_v  = fmt(mean((tp_b  or {}).get('est_ppt')))
        print(f"| {d:<5} | {mac_v:>10} | {spa_v:>9} | {pd_v:>8} | {tp_v:>8} |")
    print()

# Generation speed table
print("### Generation Speed (tok/s, mean / peak)")
print("| Depth | Mac Studio | DGX Spark | PD Split | TP Split |")
print("|-------|------------|-----------|----------|----------|")
for d in depths:
    mac_b = find_bench(configs.get('mac', {}), d)
    spa_b = find_bench(configs.get('spark', {}), d)
    pd_b  = find_bench(configs.get('disagg', {}), d)
    tp_b  = find_bench(configs.get('tpdisagg', {}), d)
    def tps_cell(b):
        if b is None: return "-"
        tg  = mean(b.get('tg_throughput'))
        pk  = mean(b.get('peak_throughput'))
        if tg is None: return "-"
        if pk is not None and pk > 0:
            return f"{tg:.1f} / {pk:.1f}"
        return f"{tg:.1f}"
    print(f"| {d:<5} | {tps_cell(mac_b):>10} | {tps_cell(spa_b):>9} | {tps_cell(pd_b):>8} | {tps_cell(tp_b):>8} |")
print()
PYEOF
  exit 0
fi

# ---------- Cluster management helpers (used by --run-all) ----------

_ssh() {
  # Disable ControlMaster to avoid stale socket issues; swallow non-zero exit codes
  # (pkill returns 1 when nothing matches, which would otherwise abort the script).
  ssh -o ControlMaster=no -o StrictHostKeyChecking=no -o ConnectTimeout=15 "$@" 2>&1 || true
}

_wait_api() {
  local url="$1" label="$2"
  echo -n "  Waiting for ${label} API"
  for i in $(seq 1 40); do
    if curl -sf "${url}/state" >/dev/null 2>&1; then
      echo " ready"
      return 0
    fi
    echo -n "."
    sleep 2
  done
  echo " TIMED OUT"
  return 1
}

_place_and_wait() {
  local api="$1" meta="$2" n_expected="$3"
  echo "  Placing ${meta} instance..."
  curl -s -X POST "${api}/place_instance" \
    -H "Content-Type: application/json" \
    -d "{\"model_id\":\"${MODEL_ID}\",\"instance_meta\":\"${meta}\"}" >/dev/null
  echo -n "  Waiting for ${n_expected} runner(s) to be Ready"
  for i in $(seq 1 90); do
    curl -s "${api}/state" -o /tmp/_bench_state.json 2>/dev/null || {
      sleep 2
      continue
    }
    result=$(python3 -c "
import json, sys
try:
    s = json.load(open('/tmp/_bench_state.json'))
    runners = s.get('runners', {})
    ready  = sum(1 for v in runners.values() if list(v.keys())[0] in ('RunnerReady','RunnerRunning'))
    failed = sum(1 for v in runners.values() if list(v.keys())[0] == 'RunnerFailed')
    print(str(ready) + '/' + str(len(runners)) + (' FAILED' if failed else ''))
except Exception as e:
    print('0/0')
" 2>/dev/null || echo "0/0")
    echo -n " [${result}]"
    [[ $result == *"FAILED"* ]] && {
      echo " RUNNER FAILED — aborting"
      return 1
    }
    [[ $result == "${n_expected}/${n_expected}" ]] && {
      echo " done"
      return 0
    }
    sleep 2
  done
  echo " TIMED OUT"
  return 1
}

_kill_mac() {
  echo "  Killing Mac exo..."
  _ssh "$MAC_SSH" 'pkill -9 -f "python.*exo" 2>/dev/null; pkill -9 -f "exo_runner" 2>/dev/null; true'
  sleep 1
}

_kill_spark() {
  echo "  Killing Spark 1 exo..."
  # pkill -9 -f "python" may kill the SSH-auth helper and return 255 — _ssh swallows that.
  _ssh "$SPARK_SSH" 'pkill -9 -f "python" 2>/dev/null; true'
  sleep 1
}

_kill_spark2() {
  echo "  Killing Spark 2 exo..."
  _ssh "$SPARK2_SSH" 'pkill -9 -f "python" 2>/dev/null; true'
  sleep 1
}

_start_mac() {
  _kill_mac
  # EXO_MEMORY_THRESHOLD=0.0: force KVPrefixCache to evict before every new store,
  # keeping at most 1 entry at a time. Prevents the deepcopy accumulation of multiple
  # 1GB+ KV caches that triggers a Python GC crash at 16K+ context with 3+ trials.
  _ssh "$MAC_SSH" 'cd ~/exo && EXO_MEMORY_THRESHOLD=0.0 nohup .venv/bin/python3 -m exo -v > /tmp/exo.log 2>&1 & disown; echo ok'
}

_start_spark() {
  local dial_arg="${1:-}"
  local ld="$SPARK_LD_PATH"
  local uv="$SPARK_UV_CMD"
  _kill_spark
  # Pass the startup script via stdin (bash -s) so $HOME expands on the remote.
  # The local heredoc expands $ld and $dial_arg; remote shell expands $HOME.
  _ssh "$SPARK_SSH" 'bash -s' <<EOF
export LD_LIBRARY_PATH="$ld"
cd ~/exo
nohup $uv run exo -v $dial_arg > /tmp/exo.log 2>&1 &
disown
echo ok
EOF
}

_start_spark2() {
  local dial_arg="${1:-}"
  local ld="$SPARK_LD_PATH"
  local uv="$SPARK2_UV_CMD"
  _kill_spark2
  _ssh "$SPARK2_SSH" 'bash -s' <<EOF
export LD_LIBRARY_PATH="$ld"
cd ~/exo
nohup $uv run exo -v $dial_arg > /tmp/exo.log 2>&1 &
disown
echo ok
EOF
}

# ---------- Run-all mode ----------
if $RUN_ALL; then
  SCRIPT_PATH="$(realpath "$0")"
  # Flags forwarded to each sub-invocation
  FWD="--model ${MODEL_ID} --trials ${TRIALS}"
  FWD+=" --mac-ssh ${MAC_SSH} --spark-ssh ${SPARK_SSH} --spark2-ssh ${SPARK2_SSH}"
  FWD+=" --mac-api ${MAC_API} --spark-api ${SPARK_API} --spark2-api ${SPARK2_API}"

  # Returns 0 (true) if the given config should be run based on --configs filter
  _should_run() { [[ -z $ONLY_CONFIGS ]] || [[ " $ONLY_CONFIGS " == *" $1 "* ]]; }

  echo "================================================"
  echo "  FULL BENCHMARK SUITE — run-all"
  [[ -n $ONLY_CONFIGS ]] && echo "  Configs:  ${ONLY_CONFIGS}" || echo "  Configs:  mac spark disagg tpdisagg"
  echo "  Model:    ${MODEL_ID}"
  echo "  Trials:   ${TRIALS}"
  echo "  Depths:   ${DEPTHS}"
  echo "  Mac:      ${MAC_SSH} / ${MAC_API}"
  echo "  Spark 1:  ${SPARK_SSH} / ${SPARK_API}"
  echo "  Spark 2:  ${SPARK2_SSH} / ${SPARK2_API}"
  echo "================================================"
  echo ""

  MAC_HOST="${MAC_SSH##*@}"

  # ── 1: Mac Studio solo ───────────────────────────────────────
  if _should_run mac; then
    echo ">>> Mac Studio solo"
    _start_mac
    _wait_api "$MAC_API" "Mac"
    _place_and_wait "$MAC_API" "MlxRing" 1
    bash "$SCRIPT_PATH" $FWD --depths "$DEPTHS" --config mac --api "$MAC_API"
    _kill_mac
    echo ""
  fi

  # ── 2: DGX Spark solo ────────────────────────────────────────
  if _should_run spark; then
    echo ">>> DGX Spark solo"
    _start_spark
    _wait_api "$SPARK_API" "Spark 1"
    _place_and_wait "$SPARK_API" "MlxRing" 1
    bash "$SCRIPT_PATH" $FWD --depths "$DEPTHS" --config spark --api "$SPARK_API"
    _kill_spark
    echo ""
  fi

  # ── 3: PD Split — Spark1 prefill → Mac decode ────────────────
  if _should_run disagg; then
    echo ">>> PD Split (Spark 1 prefill → Mac decode)"
    _start_mac
    _wait_api "$MAC_API" "Mac"
    MAC_PORT=$(_ssh "$MAC_SSH" \
      'lsof -i -P -n 2>/dev/null | awk "/python/ && /LISTEN/ && !/52415/ && !/52416/ {print \$9}" | sed "s/.*://" | head -1')
    echo "  Mac libp2p: /ip4/${MAC_HOST}/tcp/${MAC_PORT}"
    _start_spark "--dial /ip4/${MAC_HOST}/tcp/${MAC_PORT}"
    _wait_api "$SPARK_API" "Spark 1"
    sleep 6
    _place_and_wait "$MAC_API" "Disaggregated" 2
    bash "$SCRIPT_PATH" $FWD --depths "$DEPTHS" --config disagg --api "$MAC_API"
    _kill_mac
    _kill_spark
    echo ""
  fi

  # ── 4: TP Split — Spark1+Spark2 TP prefill → Mac decode ──────
  if _should_run tpdisagg; then
    echo ">>> TP Split (Spark 1 + Spark 2 tensor-parallel prefill → Mac decode)"
    _start_mac
    _wait_api "$MAC_API" "Mac"
    MAC_PORT=$(_ssh "$MAC_SSH" \
      'lsof -i -P -n 2>/dev/null | awk "/python/ && /LISTEN/ && !/52415/ && !/52416/ {print \$9}" | sed "s/.*://" | head -1')
    echo "  Mac libp2p: /ip4/${MAC_HOST}/tcp/${MAC_PORT}"
    _start_spark "--dial /ip4/${MAC_HOST}/tcp/${MAC_PORT}"
    _start_spark2 "--dial /ip4/${MAC_HOST}/tcp/${MAC_PORT}"
    _wait_api "$SPARK_API" "Spark 1"
    _wait_api "$SPARK2_API" "Spark 2"
    sleep 8
    _place_and_wait "$MAC_API" "TensorPrefillDisagg" 3
    bash "$SCRIPT_PATH" $FWD --depths "$DEPTHS" --config tpdisagg --api "$MAC_API"
    _kill_mac
    _kill_spark
    _kill_spark2
    echo ""
  fi

  # ── Combine ──────────────────────────────────────────────────
  echo ">>> Combining results"
  bash "$SCRIPT_PATH" $FWD --combine
  exit 0
fi

# ---------- Individual benchmark mode ----------
if [ -z "$CONFIG" ] || [ -z "$API_URL" ]; then
  echo "Usage:"
  echo "  $0 --run-all                              # full end-to-end suite (4 configs)"
  echo "  $0 --config mac|spark|disagg|tpdisagg --api URL  # single config (cluster must be ready)"
  echo "  $0 --combine                              # combine saved results"
  exit 1
fi

OUTFILE="${RESULTS_DIR}/bench_${CONFIG}.json"

echo "=== Comparison Benchmark: $CONFIG ==="
echo "API:     $API_URL"
echo "Model:   $MODEL_ID"
echo "Trials:  $TRIALS"
echo "Depths:  $DEPTHS"
echo "Output:  $OUTFILE"
echo "Runner:  llama-benchy via uvx"
echo ""

# ---------- Check llama-benchy ----------
if ! command -v uvx >/dev/null 2>&1; then
  echo "Error: uvx not found. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
if ! uvx llama-benchy --help >/dev/null 2>&1; then
  echo "Error: llama-benchy not accessible via uvx. Try: uvx llama-benchy --help"
  exit 1
fi
echo "llama-benchy ready (uvx)"
echo ""

# ---------- Run llama-benchy ----------
# Single call handles all depths, warmup, prompt generation, and JSON output.
# --depth: varies context depth (llama-benchy uses Sherlock Holmes corpus)
# --latency-mode generation: estimates server-side time from generation timing
echo "Running llama-benchy: $TRIALS trials x depths [${DEPTHS}] ..."
echo ""

# Build the depth args (space-separated → separate --depth values not needed,
# llama-benchy accepts: --depth 0 4096 8192 ...)
# shellcheck disable=SC2086
uvx llama-benchy \
  --base-url "${API_URL}/v1" \
  --model "$MODEL_ID" \
  --depth $DEPTHS \
  --runs "$TRIALS" \
  --latency-mode generation \
  --format json \
  --save-result "$OUTFILE"

echo ""
echo "Results saved to $OUTFILE"
echo ""
echo "To combine all results:"
echo "  $0 --combine"
