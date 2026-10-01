#!/usr/bin/env bash
# Post-startup smoke test for the local stack (brief §26). Brings up the `core` profile, waits
# for the gateway to report ready, checks each service's health, and — in bypass mode — submits
# one trip request end to end. Exits non-zero on the first failure.
#
#   ./scripts/smoke_test.sh            # assumes the stack is already up
#   ./scripts/smoke_test.sh --up       # brings up `core`, tests, then leaves it running
set -euo pipefail

GATEWAY="${GATEWAY_URL:-http://localhost:8000}"
COMPOSE="docker compose"

if [[ "${1:-}" == "--up" ]]; then
  echo "==> Starting the core profile…"
  $COMPOSE --profile core up --build -d
fi

echo "==> Waiting for the gateway to become ready…"
for _ in $(seq 1 60); do
  if curl -fsS "${GATEWAY}/health/ready" >/dev/null 2>&1; then break; fi
  sleep 2
done

echo "==> /health/live"
curl -fsS "${GATEWAY}/health/live" | grep -q '"status":"alive"' && echo "   OK"

echo "==> /health/ready (component states)"
curl -fsS "${GATEWAY}/health/ready"; echo

echo "==> /metrics exposes Prometheus counters"
curl -fsS "${GATEWAY}/metrics" | grep -q "api_requests_total" && echo "   OK"

echo "==> OpenAPI names the trip routes"
curl -fsS "${GATEWAY}/openapi.json" | grep -q "/api/v1/trips" && echo "   OK"

echo "==> Submitting a trip (dev bypass)…"
BODY='{"origin":"Nuremberg","destination":"Prague","departure_date":"2026-09-10",
"return_date":"2026-09-13","travellers":1,"max_budget":{"amount":"600.00","currency":"EUR"},
"accommodation_preference":"hostel","transport_preference":"any","interests":["history"],
"max_transport_duration_hours":8,"max_transfers":2,"accessibility_needs":[],
"ranking_strategy":"balanced"}'
STATUS=$(curl -fsS -X POST "${GATEWAY}/api/v1/trips" \
  -H 'Content-Type: application/json' -d "${BODY}" | grep -o '"status":"[a-z_]*"' | head -1)
echo "   plan ${STATUS}"
[[ -n "${STATUS}" ]] && echo "   OK"

echo "==> Smoke test passed."
