#!/usr/bin/env bash
# Comprehensive comparison benchmark for disaggregated vs single-node inference.
# Tests 3 configurations at 4 context lengths, measuring TTFT and generation TPS.
#
# Configurations:
#   mac    — Mac Studio M2 Ultra (single node)
#   spark  — DGX Spark NVIDIA GB10 (single node)
#   disagg — PD Split: DGX Spark prefill → Mac Studio decode
#
# Usage:
#   ./scripts/benchmark_comparison.sh --config mac     --api http://192.168.0.114:52415
#   ./scripts/benchmark_comparison.sh --config spark   --api http://192.168.0.112:52415
#   ./scripts/benchmark_comparison.sh --config disagg  --api http://192.168.0.114:52415 \
#       --prefill-ssh pnivek@192.168.0.112 --decode-ssh kevin@192.168.0.114
#   ./scripts/benchmark_comparison.sh --combine
#
# Results are saved to /tmp/bench_{config}.json and combined into a markdown table.
#
# Requires: curl, python3, jq (for --combine), ssh (for disagg)

set -euo pipefail

MODEL_ID="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
TRIALS=3
PROMPT_LENGTHS=(45 512 2048 8192 16384 32768)
MAX_TOKENS=200
RESULTS_DIR="/tmp"

# Parse arguments
CONFIG=""
API_URL=""
PREFILL_SSH=""
DECODE_SSH=""
COMBINE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
  --config)
    CONFIG="$2"
    shift 2
    ;;
  --api)
    API_URL="$2"
    shift 2
    ;;
  --prefill-ssh)
    PREFILL_SSH="$2"
    shift 2
    ;;
  --decode-ssh)
    DECODE_SSH="$2"
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
  --combine)
    COMBINE=true
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
  python3 - "$RESULTS_DIR" <<'PYEOF'
import json
import sys
import os

results_dir = sys.argv[1]
configs = {}

for name in ["mac", "spark", "disagg"]:
    path = os.path.join(results_dir, f"bench_{name}.json")
    if os.path.exists(path):
        with open(path) as f:
            configs[name] = json.load(f)

if not configs:
    print("No result files found in " + results_dir)
    sys.exit(1)

lengths = [45, 512, 2048, 8192, 16384, 32768]

print("## Comparison: Llama 3.1 8B 4-bit (3 trials avg, 200 gen tokens)")
print()

# TTFT table
print("### TTFT (ms)")
header = "| Prompt Tokens | Mac Studio | DGX Spark | PD Split WiFi | PD Split 10GbE |"
sep =    "|---------------|------------|-----------|---------------|----------------|"
print(header)
print(sep)
for l in lengths:
    key = str(l)
    mac_ttft = f"{configs['mac'][key]['ttft_ms']:.0f}" if 'mac' in configs and key in configs['mac'] else "-"
    spark_ttft = f"{configs['spark'][key]['ttft_ms']:.0f}" if 'spark' in configs and key in configs['spark'] else "-"
    disagg_ttft = f"{configs['disagg'][key]['ttft_ms']:.0f}" if 'disagg' in configs and key in configs['disagg'] else "-"
    disagg_10g = "-"
    if 'disagg' in configs and key in configs['disagg']:
        d = configs['disagg'][key]
        if 'phases' in d:
            p = d['phases']
            gbe_net = p.get('kv_size_mb', 0) / 1250.0 * 1000
            disagg_10g = f"{p['prefill_ms'] + p['serialize_ms'] + gbe_net + p['deserialize_ms'] + p['first_tok_ms'] + p['orchestration_ms']:.0f}"
    print(f"| {l:<13} | {mac_ttft:>10} | {spark_ttft:>9} | {disagg_ttft:>13} | {disagg_10g:>14} |")
print()

# Generation speed table
print("### Generation Speed (tok/s)")
header = "| Prompt Tokens | Mac Studio | DGX Spark | PD Split |"
sep =    "|---------------|------------|-----------|----------|"
print(header)
print(sep)
for l in lengths:
    key = str(l)
    mac_tps = f"{configs['mac'][key]['tps']:.1f}" if 'mac' in configs and key in configs['mac'] else "-"
    spark_tps = f"{configs['spark'][key]['tps']:.1f}" if 'spark' in configs and key in configs['spark'] else "-"
    disagg_tps = f"{configs['disagg'][key]['tps']:.1f}" if 'disagg' in configs and key in configs['disagg'] else "-"
    print(f"| {l:<13} | {mac_tps:>10} | {spark_tps:>9} | {disagg_tps:>8} |")
print()

# PD Split phase breakdown (disagg only)
if 'disagg' in configs:
    print("### PD Split Phase Breakdown (ms)")
    header = "| Prompt | Prefill | KV Ser | KV Net WiFi | KV Net 10GbE | KV Deser | 1st Tok | Orch |"
    sep    = "|--------|---------|--------|-------------|--------------|----------|---------|------|"
    print(header)
    print(sep)
    for l in lengths:
        key = str(l)
        if key in configs['disagg'] and 'phases' in configs['disagg'][key]:
            p = configs['disagg'][key]['phases']
            gbe_net = p.get('kv_size_mb', 0) / 1250.0 * 1000
            print(f"| {l:<6} | {p['prefill_ms']:>7.0f} | {p['serialize_ms']:>6.0f} | {p['net_send_ms']:>11.0f} | {gbe_net:>12.0f} | {p['deserialize_ms']:>8.0f} | {p['first_tok_ms']:>7.0f} | {p['orchestration_ms']:>4.0f} |")
        else:
            print(f"| {l:<6} | {'':>7} | {'':>6} | {'':>11} | {'':>12} | {'':>8} | {'':>7} | {'':>4} |")
    print()
PYEOF
  exit 0
fi

# ---------- Benchmark mode ----------
if [ -z "$CONFIG" ] || [ -z "$API_URL" ]; then
  echo "Usage: $0 --config <mac|spark|disagg> --api <URL> [--prefill-ssh SSH] [--decode-ssh SSH]"
  exit 1
fi

if [ "$CONFIG" = "disagg" ]; then
  if [ -z "$PREFILL_SSH" ] || [ -z "$DECODE_SSH" ]; then
    echo "Error: --prefill-ssh and --decode-ssh required for disagg config"
    exit 1
  fi
fi

echo "=== Comparison Benchmark: $CONFIG ==="
echo "API:       $API_URL"
echo "Model:     $MODEL_ID"
echo "Trials:    $TRIALS"
echo "Lengths:   ${PROMPT_LENGTHS[*]}"
echo "Max tokens: $MAX_TOKENS"
[ -n "$PREFILL_SSH" ] && echo "Prefill:   $PREFILL_SSH"
[ -n "$DECODE_SSH" ] && echo "Decode:    $DECODE_SSH"
echo ""

# Generate a prompt of approximately the target token count.
# Takes target_tokens and an optional trial index (0-based) to vary the prompt.
# Returns the prompt string on stdout.
generate_prompt() {
  local target_tokens="$1"
  local trial_idx="${2:-0}"
  python3 -c "
import sys
target = int('$target_tokens')
trial_idx = int('$trial_idx')

# Different base passages to avoid KV prefix cache hits across trials
passages = [
    'The sun rose slowly over the mountain peaks, casting long shadows across the valley below. Birds began their morning chorus as the first rays of light touched the river, turning its surface into liquid gold. A gentle breeze carried the scent of pine and wildflowers through the forest clearing where deer grazed peacefully in the dawn light.',
    'In the depths of the ocean, creatures moved through the darkness with a grace born of millennia of evolution. Bioluminescent jellyfish pulsed with ethereal light as they drifted through the midnight zone. Schools of lanternfish created constellations beneath the waves while ancient nautilus shells spiraled through currents older than memory.',
    'The old library stood at the edge of town, its shelves heavy with books that held centuries of human knowledge. Dust motes danced in the sunlight streaming through tall windows as a scholar carefully turned the pages of a medieval manuscript. The smell of aged paper and leather bindings filled the quiet reading room.',
    'Thunder rolled across the prairie as dark clouds gathered on the western horizon. The wheat fields rippled like a golden ocean under the wind, their stalks bending and swaying in unison. A farmhouse stood resolute against the approaching storm, its windows glowing warm against the darkening afternoon sky.',
    'The marketplace bustled with merchants calling out their wares in a dozen languages. Silk and spices from distant lands filled wooden stalls while craftsmen hammered copper into intricate designs. Children darted between the crowds chasing a stray cat as the aroma of freshly baked bread mingled with exotic perfumes.',
]

base = passages[trial_idx % len(passages)]
# ~1.3 tokens per word; each passage is ~60 tokens
tokens_per_passage = 60
reps = max(1, target // tokens_per_passage)
prompt = ' '.join([base] * reps)
suffix = ' Now write a short creative story continuing from the text above.'
print(prompt + suffix)
"
}

# Run a single trial with a given prompt and timeout.
# Outputs "ttft_ms tps token_count" on stdout.
run_trial() {
  local prompt="$1"
  local timeout="$2"
  python3 - "$API_URL" "$MODEL_ID" "$prompt" "$MAX_TOKENS" "$timeout" <<'PYEOF'
import json
import sys
import time
import urllib.request

api_url = sys.argv[1]
model_id = sys.argv[2]
prompt = sys.argv[3]
max_tokens = int(sys.argv[4])
timeout = int(sys.argv[5])

url = f"{api_url}/v1/chat/completions"
payload = json.dumps({
    "model": model_id,
    "messages": [{"role": "user", "content": prompt}],
    "stream": True,
    "max_tokens": max_tokens,
    "temperature": 0.7,
}).encode()

req = urllib.request.Request(
    url,
    data=payload,
    headers={"Content-Type": "application/json"},
)

t_start = time.monotonic()
t_first = None
token_count = 0

with urllib.request.urlopen(req, timeout=timeout) as resp:
    buf = b""
    while True:
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
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            reasoning = delta.get("reasoning_content", "")
            if content or reasoning:
                if t_first is None:
                    t_first = time.monotonic()
                token_count += 1

t_end = time.monotonic()

if t_first is None:
    print("ERROR 0 0", file=sys.stdout)
    sys.exit(1)

ttft_ms = (t_first - t_start) * 1000
gen_time = t_end - t_first
tps = token_count / gen_time if gen_time > 0 else 0

print(f"{ttft_ms:.1f} {tps:.1f} {token_count}")
PYEOF
}

# Extract a DISAGG_TIMING value from log lines.
extract_timing() {
  local lines="$1"
  local key="$2"
  echo "$lines" | grep -oP "${key}=\K[0-9.]+" | tail -1 || echo "0"
}

# Compute timeout for a given prompt length
compute_timeout() {
  local prompt_tokens="$1"
  python3 -c "print(max(120, int($prompt_tokens / 10 + 120)))"
}

# Initial warmup with shortest prompt (loads model into memory)
echo "Warming up (model load)..."
warmup_prompt=$(generate_prompt "${PROMPT_LENGTHS[0]}" 99)
warmup_timeout=$(compute_timeout "${PROMPT_LENGTHS[0]}")
run_trial "$warmup_prompt" "$warmup_timeout" >/dev/null 2>&1 || true
sleep 2
echo ""

# JSON accumulator — built incrementally
json_results="{}"

for prompt_len in "${PROMPT_LENGTHS[@]}"; do
  echo "=== Prompt length: ~${prompt_len} tokens ==="

  timeout=$(compute_timeout "$prompt_len")

  # Per-context-length warmup (primes KV allocation at this size, uses unique prompt)
  echo "  Warmup at ${prompt_len} tokens..."
  warmup_prompt=$(generate_prompt "$prompt_len" 98)
  run_trial "$warmup_prompt" "$timeout" >/dev/null 2>&1 || true
  sleep 1

  ttft_sum=0
  tps_sum=0
  success=0

  # Phase accumulators for disagg
  prefill_sum=0
  ser_sum=0
  net_sum=0
  deser_sum=0
  ftok_sum=0
  orch_sum=0
  kv_size_sum=0

  for i in $(seq 1 "$TRIALS"); do
    # Generate unique prompt per trial to avoid KV prefix cache hits
    prompt=$(generate_prompt "$prompt_len" "$((i - 1))")

    echo -n "  Trial $i/$TRIALS ... "

    # Record log offsets for disagg
    if [ "$CONFIG" = "disagg" ]; then
      prefill_offset=$(ssh "$PREFILL_SSH" "wc -c < /tmp/exo.log" 2>/dev/null || echo "0")
      decode_offset=$(ssh "$DECODE_SSH" "wc -c < /tmp/exo.log" 2>/dev/null || echo "0")
    fi

    result=$(run_trial "$prompt" "$timeout" 2>/dev/null) || {
      echo "FAILED"
      continue
    }
    ttft=$(echo "$result" | awk '{print $1}')
    tps=$(echo "$result" | awk '{print $2}')
    tokens=$(echo "$result" | awk '{print $3}')

    if [ "$ttft" = "ERROR" ]; then
      echo "FAILED (no tokens)"
      continue
    fi

    echo "TTFT=${ttft}ms  TPS=${tps}  tokens=${tokens}"

    ttft_sum=$(python3 -c "print($ttft_sum + $ttft)")
    tps_sum=$(python3 -c "print($tps_sum + $tps)")
    success=$((success + 1))

    # Extract disagg timings
    if [ "$CONFIG" = "disagg" ]; then
      sleep 2 # Let logs flush

      prefill_lines=$(ssh "$PREFILL_SSH" "tail -c +$((prefill_offset + 1)) /tmp/exo.log | grep DISAGG_TIMING" 2>/dev/null || echo "")
      decode_lines=$(ssh "$DECODE_SSH" "tail -c +$((decode_offset + 1)) /tmp/exo.log | grep DISAGG_TIMING" 2>/dev/null || echo "")

      # Support both pipelined and bulk timing keys
      pipelined_total=$(extract_timing "$prefill_lines" "pipelined_total_ms")
      pipelined_prefill=$(extract_timing "$prefill_lines" "pipelined_prefill_ms")

      if [ "$(echo "$pipelined_total > 0" | bc -l 2>/dev/null || python3 -c "print(1 if $pipelined_total > 0 else 0)")" = "1" ]; then
        # Pipelined mode — prefill and network overlap
        prefill_ms=$pipelined_prefill
        serialize_ms=0               # No separate serialization in pipelined mode
        net_send_ms=$pipelined_total # Total includes overlapped network
        kv_size_mb=$(extract_timing "$prefill_lines" "chunk_mb")
      else
        # Bulk mode (legacy fallback)
        prefill_ms=$(extract_timing "$prefill_lines" "prefill_compute_ms")
        serialize_ms=$(extract_timing "$prefill_lines" "kv_serialize_ms")
        net_send_ms=$(extract_timing "$prefill_lines" "kv_network_send_ms")
        kv_size_mb=$(extract_timing "$prefill_lines" "kv_size_mb")
      fi

      deserialize_ms=$(extract_timing "$decode_lines" "kv_deserialize_ms")
      first_tok_ms=$(extract_timing "$decode_lines" "decode_first_token_ms")

      measured_phases=$(python3 -c "print($prefill_ms + $serialize_ms + $net_send_ms + $deserialize_ms + $first_tok_ms)")
      orchestration=$(python3 -c "print(max(0, $ttft - $measured_phases))")

      echo "    Phases: prefill=${prefill_ms} ser=${serialize_ms} net=${net_send_ms} deser=${deserialize_ms} ftok=${first_tok_ms} orch=${orchestration} kv=${kv_size_mb}MB"

      prefill_sum=$(python3 -c "print($prefill_sum + $prefill_ms)")
      ser_sum=$(python3 -c "print($ser_sum + $serialize_ms)")
      net_sum=$(python3 -c "print($net_sum + $net_send_ms)")
      deser_sum=$(python3 -c "print($deser_sum + $deserialize_ms)")
      ftok_sum=$(python3 -c "print($ftok_sum + $first_tok_ms)")
      orch_sum=$(python3 -c "print($orch_sum + $orchestration)")
      kv_size_sum=$(python3 -c "print($kv_size_sum + $kv_size_mb)")
    fi
  done

  if [ "$success" -eq 0 ]; then
    echo "  All trials failed for ${prompt_len} tokens, skipping."
    echo ""
    continue
  fi

  avg_ttft=$(python3 -c "print($ttft_sum / $success)")
  avg_tps=$(python3 -c "print($tps_sum / $success)")

  echo "  Avg: TTFT=$(python3 -c "print(f'{$avg_ttft:.1f}')")ms  TPS=$(python3 -c "print(f'{$avg_tps:.1f}')")  ($success/$TRIALS)"

  # Build JSON entry for this prompt length
  if [ "$CONFIG" = "disagg" ] && [ "$success" -gt 0 ]; then
    phases_json=$(python3 -c "
import json
s = $success
print(json.dumps({
    'prefill_ms': $prefill_sum / s,
    'serialize_ms': $ser_sum / s,
    'net_send_ms': $net_sum / s,
    'deserialize_ms': $deser_sum / s,
    'first_tok_ms': $ftok_sum / s,
    'orchestration_ms': $orch_sum / s,
    'kv_size_mb': $kv_size_sum / s,
}))
")
    json_results=$(python3 -c "
import json
r = json.loads('''$json_results''')
r['$prompt_len'] = {
    'ttft_ms': $avg_ttft,
    'tps': $avg_tps,
    'trials': $success,
    'phases': json.loads('''$phases_json''')
}
print(json.dumps(r))
")
  else
    json_results=$(python3 -c "
import json
r = json.loads('''$json_results''')
r['$prompt_len'] = {
    'ttft_ms': $avg_ttft,
    'tps': $avg_tps,
    'trials': $success,
}
print(json.dumps(r))
")
  fi

  echo ""
done

# Save results
output_file="${RESULTS_DIR}/bench_${CONFIG}.json"
echo "$json_results" | python3 -m json.tool >"$output_file"
echo "Results saved to $output_file"
echo ""
echo "Run all configs, then combine:"
echo "  $0 --combine"
