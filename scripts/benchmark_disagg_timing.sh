#!/usr/bin/env bash
# Benchmark script for disaggregated inference TTFT timing breakdown.
# Extracts DISAGG_TIMING log lines from prefill and decode nodes via SSH,
# computes a phase-by-phase breakdown, and projects 10GbE performance.
#
# Prerequisites:
#   - exo must be running on both nodes with logs redirected to /tmp/exo.log:
#       uv run exo -v > /tmp/exo.log 2>&1
#   - SSH key-based auth to both nodes (no password prompts)
#
# Usage:
#   ./scripts/benchmark_disagg_timing.sh <API_URL> <PREFILL_SSH> <DECODE_SSH> [MODEL_ID] [TRIALS]
#
# Examples:
#   ./scripts/benchmark_disagg_timing.sh http://192.168.0.114:52415 \
#     pnivek@192.168.0.112 kevin@192.168.0.114 \
#     mlx-community/Meta-Llama-3.1-8B-Instruct-8bit 3

set -euo pipefail

API_URL="${1:?Usage: $0 <API_URL> <PREFILL_SSH> <DECODE_SSH> [MODEL_ID] [TRIALS]}"
PREFILL_SSH="${2:?Usage: $0 <API_URL> <PREFILL_SSH> <DECODE_SSH> [MODEL_ID] [TRIALS]}"
DECODE_SSH="${3:?Usage: $0 <API_URL> <PREFILL_SSH> <DECODE_SSH> [MODEL_ID] [TRIALS]}"
MODEL_ID="${4:-mlx-community/Meta-Llama-3.1-8B-Instruct-8bit}"
TRIALS="${5:-3}"

PROMPT="Write a short poem about the ocean in exactly four lines."

echo "=== Disaggregated Inference Timing Breakdown ==="
echo "API:          $API_URL"
echo "Prefill node: $PREFILL_SSH"
echo "Decode node:  $DECODE_SSH"
echo "Model:        $MODEL_ID"
echo "Trials:       $TRIALS"
echo "Prompt:       \"$PROMPT\""
echo ""

# Run a single trial — outputs "ttft_ms tps token_count" on stdout
run_trial() {
  python3 - "$API_URL" "$MODEL_ID" "$PROMPT" <<'PYEOF'
import json
import sys
import time
import urllib.request

api_url = sys.argv[1]
model_id = sys.argv[2]
prompt = sys.argv[3]

url = f"{api_url}/v1/chat/completions"
payload = json.dumps({
    "model": model_id,
    "messages": [{"role": "user", "content": prompt}],
    "stream": True,
    "max_tokens": 128,
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

with urllib.request.urlopen(req, timeout=60) as resp:
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
            if content:
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
# Usage: extract_timing "$lines" "key_name"
extract_timing() {
  local lines="$1"
  local key="$2"
  echo "$lines" | grep -oP "${key}=\K[0-9.]+" | tail -1 || echo "0"
}

# Warmup
echo "Warming up..."
run_trial >/dev/null 2>&1 || true
sleep 2
echo ""

# Collect results
declare -a all_ttft all_prefill all_serialize all_net_send all_net_recv all_deserialize all_first_tok all_kv_size all_orchestration

for i in $(seq 1 "$TRIALS"); do
  echo "--- Trial $i/$TRIALS ---"

  # Record log file sizes before the trial so we only read new lines after
  prefill_offset=$(ssh "$PREFILL_SSH" "wc -c < /tmp/exo.log" 2>/dev/null || echo "0")
  decode_offset=$(ssh "$DECODE_SSH" "wc -c < /tmp/exo.log" 2>/dev/null || echo "0")

  # Run the request
  result=$(run_trial 2>/dev/null) || {
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

  sleep 2 # Give logs time to flush

  # Extract only NEW timing lines (after the byte offset recorded before this trial)
  prefill_lines=$(ssh "$PREFILL_SSH" "tail -c +$((prefill_offset + 1)) /tmp/exo.log | grep DISAGG_TIMING" 2>/dev/null || echo "")
  decode_lines=$(ssh "$DECODE_SSH" "tail -c +$((decode_offset + 1)) /tmp/exo.log | grep DISAGG_TIMING" 2>/dev/null || echo "")

  # Parse prefill node timings
  prefill_ms=$(extract_timing "$prefill_lines" "prefill_compute_ms")
  serialize_ms=$(extract_timing "$prefill_lines" "kv_serialize_ms")
  net_send_ms=$(extract_timing "$prefill_lines" "kv_network_send_ms")
  kv_size_mb=$(extract_timing "$prefill_lines" "kv_size_mb")

  # Parse decode node timings
  net_recv_ms=$(extract_timing "$decode_lines" "kv_network_recv_ms")
  deserialize_ms=$(extract_timing "$decode_lines" "kv_deserialize_ms")
  first_tok_ms=$(extract_timing "$decode_lines" "decode_first_token_ms")

  # Compute orchestration overhead
  measured_phases=$(python3 -c "print($prefill_ms + $serialize_ms + $net_send_ms + $deserialize_ms + $first_tok_ms)")
  orchestration=$(python3 -c "print(max(0, $ttft - $measured_phases))")

  echo "  Client TTFT:      ${ttft} ms"
  echo "  Prefill compute:  ${prefill_ms} ms"
  echo "  KV serialize:     ${serialize_ms} ms"
  echo "  KV network send:  ${net_send_ms} ms"
  echo "  KV network recv:  ${net_recv_ms} ms"
  echo "  KV deserialize:   ${deserialize_ms} ms"
  echo "  Decode first tok: ${first_tok_ms} ms"
  echo "  KV size:          ${kv_size_mb} MB"
  echo "  Orchestration:    ${orchestration} ms"
  echo "  Gen TPS:          ${tps} tok/s (${tokens} tokens)"
  echo ""

  all_ttft+=("$ttft")
  all_prefill+=("$prefill_ms")
  all_serialize+=("$serialize_ms")
  all_net_send+=("$net_send_ms")
  all_net_recv+=("$net_recv_ms")
  all_deserialize+=("$deserialize_ms")
  all_first_tok+=("$first_tok_ms")
  all_kv_size+=("$kv_size_mb")
  all_orchestration+=("$orchestration")
done

n=${#all_ttft[@]}
if [ "$n" -eq 0 ]; then
  echo "All trials failed."
  exit 1
fi

# Compute averages and 10GbE projection
python3 - "$n" \
  "$(
    IFS=,
    echo "${all_ttft[*]}"
  )" \
  "$(
    IFS=,
    echo "${all_prefill[*]}"
  )" \
  "$(
    IFS=,
    echo "${all_serialize[*]}"
  )" \
  "$(
    IFS=,
    echo "${all_net_send[*]}"
  )" \
  "$(
    IFS=,
    echo "${all_net_recv[*]}"
  )" \
  "$(
    IFS=,
    echo "${all_deserialize[*]}"
  )" \
  "$(
    IFS=,
    echo "${all_first_tok[*]}"
  )" \
  "$(
    IFS=,
    echo "${all_kv_size[*]}"
  )" \
  "$(
    IFS=,
    echo "${all_orchestration[*]}"
  )" \
  <<'PYEOF'
import sys

n = int(sys.argv[1])

def parse_list(s):
    return [float(x) for x in s.split(",")]

ttft = parse_list(sys.argv[2])
prefill = parse_list(sys.argv[3])
serialize = parse_list(sys.argv[4])
net_send = parse_list(sys.argv[5])
net_recv = parse_list(sys.argv[6])
deserialize = parse_list(sys.argv[7])
first_tok = parse_list(sys.argv[8])
kv_size = parse_list(sys.argv[9])
orchestration = parse_list(sys.argv[10])

def avg(lst):
    return sum(lst) / len(lst)

a_ttft = avg(ttft)
a_prefill = avg(prefill)
a_serialize = avg(serialize)
a_net_send = avg(net_send)
a_net_recv = avg(net_recv)
a_deserialize = avg(deserialize)
a_first_tok = avg(first_tok)
a_kv_size = avg(kv_size)
a_orchestration = avg(orchestration)

# 10GbE projection: 1250 MB/s effective throughput
gbe_net_ms = a_kv_size / 1250.0 * 1000
gbe_ttft = a_prefill + a_serialize + gbe_net_ms + a_deserialize + a_first_tok + a_orchestration

# WiFi bandwidth estimate from actual transfer
wifi_bandwidth_mbps = a_kv_size / (a_net_send / 1000) * 8 if a_net_send > 0 else 0

print(f"=== Average Results ({n}/{sys.argv[1]} successful) ===")
print(f"KV cache size: {a_kv_size:.2f} MB")
print(f"WiFi bandwidth: ~{wifi_bandwidth_mbps:.0f} Mbps ({a_kv_size / (a_net_send / 1000):.1f} MB/s)" if a_net_send > 0 else "WiFi bandwidth: N/A")
print()
print("| Phase              | WiFi (ms) | 10GbE (ms) |")
print("|--------------------|-----------|------------|")
print(f"| Prefill compute    | {a_prefill:9.1f} | {a_prefill:10.1f} |")
print(f"| KV serialize       | {a_serialize:9.1f} | {a_serialize:10.1f} |")
print(f"| KV network         | {a_net_send:9.1f} | {gbe_net_ms:10.1f} |")
print(f"| KV deserialize     | {a_deserialize:9.1f} | {a_deserialize:10.1f} |")
print(f"| Decode first token | {a_first_tok:9.1f} | {a_first_tok:10.1f} |")
print(f"| Orchestration      | {a_orchestration:9.1f} | {a_orchestration:10.1f} |")
print(f"| **Total TTFT**     | **{a_ttft:7.1f}** | **{gbe_ttft:8.1f}** |")
print()
if a_net_send > 0:
    speedup = a_ttft / gbe_ttft if gbe_ttft > 0 else 0
    net_pct_wifi = a_net_send / a_ttft * 100 if a_ttft > 0 else 0
    net_pct_gbe = gbe_net_ms / gbe_ttft * 100 if gbe_ttft > 0 else 0
    print(f"10GbE projected speedup: {speedup:.2f}x")
    print(f"Network as % of TTFT: WiFi={net_pct_wifi:.1f}%, 10GbE={net_pct_gbe:.1f}%")
PYEOF
