#!/usr/bin/env bash
# Benchmark script for disaggregated inference on exo.
# Measures TTFT (time to first token) and TPS (tokens per second)
# via streaming SSE from the OpenAI-compatible chat completions API.
#
# Usage:
#   ./scripts/benchmark_disagg.sh <API_URL> [MODEL_ID] [TRIALS]
#
# Examples:
#   ./scripts/benchmark_disagg.sh http://192.168.0.112:52415   # Spark solo
#   ./scripts/benchmark_disagg.sh http://192.168.0.114:52415   # Mac solo / Disaggregated
#
# The script sends a standard prompt, streams the response, and reports:
#   - TTFT: time from request to first SSE data chunk (seconds)
#   - TPS:  total generated tokens / total generation time
#
# Requires: curl, python3 (for JSON parsing and timing)

set -euo pipefail

API_URL="${1:?Usage: $0 <API_URL> [MODEL_ID] [TRIALS]}"
MODEL_ID="${2:-mlx-community/Meta-Llama-3.1-8B-Instruct-8bit}"
TRIALS="${3:-3}"

PROMPT="Write a short poem about the ocean in exactly four lines."

echo "=== exo Disaggregated Inference Benchmark ==="
echo "API:    $API_URL"
echo "Model:  $MODEL_ID"
echo "Trials: $TRIALS"
echo "Prompt: \"$PROMPT\""
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

with urllib.request.urlopen(req, timeout=30) as resp:
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

# Warmup
echo "Warming up..."
run_trial >/dev/null 2>&1 || true
echo ""

# Collect results
ttft_sum=0
tps_sum=0
tokens_sum=0
success=0

for i in $(seq 1 "$TRIALS"); do
  echo -n "Trial $i/$TRIALS ... "
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

  echo "TTFT=${ttft}ms  TPS=${tps}  tokens=${tokens}"
  ttft_sum=$(python3 -c "print($ttft_sum + $ttft)")
  tps_sum=$(python3 -c "print($tps_sum + $tps)")
  tokens_sum=$((tokens_sum + tokens))
  success=$((success + 1))
done

echo ""

if [ "$success" -eq 0 ]; then
  echo "All trials failed."
  exit 1
fi

avg_ttft=$(python3 -c "print(f'{$ttft_sum / $success:.1f}')")
avg_tps=$(python3 -c "print(f'{$tps_sum / $success:.1f}')")

echo "=== Results ($success/$TRIALS successful) ==="
echo ""
echo "| Metric | Value |"
echo "|--------|-------|"
echo "| Model | $MODEL_ID |"
echo "| Avg TTFT | ${avg_ttft} ms |"
echo "| Avg TPS | ${avg_tps} tok/s |"
echo "| Avg Tokens | $((tokens_sum / success)) |"
