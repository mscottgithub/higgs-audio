#!/usr/bin/env bash
set -euo pipefail

# Target the HIGGS server directly (default 127.0.0.1:8010).
HIGGS_HOST="${HIGGS_HOST:-127.0.0.1}"
HIGGS_PORT="${HIGGS_PORT:-8010}"
HIGGS="http://${HIGGS_HOST}:${HIGGS_PORT}"

pass(){ printf "✅ %s\n" "$1"; }
fail(){ printf "❌ %s\n" "$1"; exit 1; }
note(){ printf "ℹ️  %s\n" "$1"; }

# --- Optional root (some builds may not expose /) ---
note "Root"
if curl -fsS -m 3 "$HIGGS/" >/dev/null; then
  pass "/ root reachable"
else
  note "/ root not reachable (not fatal; continuing)"
fi

# --- /health with retries & raw dump if it fails ---
note "Health"
ATTEMPTS=10; SLEEP=0.5; OK=0
for ((i=1; i<=ATTEMPTS; i++)); do
  BODY="$(curl -fsS -m 3 "$HIGGS/health" || true)"
  if echo "$BODY" | grep -qi '"ok"[[:space:]]*:[[:space:]]*true'; then
    pass "/health ok"
    OK=1; break
  fi
  sleep "$SLEEP"
done
if [ "$OK" -ne 1 ]; then
  echo "---- /health raw response ----"
  curl -sv "$HIGGS/health" || true
  echo "------------------------------"
  fail "/health not ok"
fi

# --- Voices ---
note "Voices"
if curl -fsS -m 5 "$HIGGS/voices" | grep -Eq '"ok"[[:space:]]*:[[:space:]]*true|"voices"'; then
  pass "/voices responded"
else
  echo "---- /voices raw ----"
  curl -sv "$HIGGS/voices" || true
  echo "---------------------"
  fail "/voices failed"
fi

# --- SFX ---
note "SFX"
if curl -fsS -m 5 "$HIGGS/sfx" | grep -q '"ok"'; then
  pass "/sfx ok"
else
  echo "---- /sfx raw ----"
  curl -sv "$HIGGS/sfx" || true
  echo "------------------"
  fail "/sfx failed"
fi

# --- Minimal /generate (no conditioning) ---
note "Generate"
GEN_PAYLOAD='{"transcript":"Hello from Higgs smoke test","out_path":"higgs_smoke.wav","temperature":1.0,"top_p":0.95,"top_k":50,"ras_win_len":7,"ras_win_max_num_repeat":2,"seed":42}'
if curl -fsS -m 600 -H 'Content-Type: application/json' -d "$GEN_PAYLOAD" "$HIGGS/generate" | grep -q '"ok":'; then
  pass "/generate responded"
else
  echo "---- /generate raw ----"
  curl -sv -H 'Content-Type: application/json' -d "$GEN_PAYLOAD" "$HIGGS/generate" || true
  echo "------------------------"
  fail "/generate failed"
fi

pass "Higgs smoke test finished"
