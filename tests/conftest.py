"""Session-wide test configuration.

Disables live telemetry export before any application module is imported. The gateway's
``app.py`` builds a module-level app at import time (uvicorn needs the module attribute), which
configures OpenTelemetry from the environment; without this, importing the gateway in a test
would install a real OTLP exporter and its background thread would spam connection errors trying
to reach a collector that isn't running. Tests never export telemetry — they assert behaviour,
not delivery.

Using ``setdefault`` means a developer can still force it on (``OTEL_ENABLED=true pytest``) to
exercise the export path against a real collector.
"""

from __future__ import annotations

import os

os.environ.setdefault("OTEL_ENABLED", "false")
