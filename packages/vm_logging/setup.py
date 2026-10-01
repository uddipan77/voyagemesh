"""One-call logging configuration for every service.

``configure_logging(settings)`` installs the JSON formatter (or a plain console formatter in
``LOG_FORMAT=console`` for local readability) on the root logger. It is idempotent — safe to
call from each service's startup without stacking handlers.
"""

from __future__ import annotations

import logging

from vm_config.settings import Settings
from vm_logging.json_formatter import JsonFormatter

__all__ = ["configure_logging"]

_CONFIGURED_MARKER = "_vm_logging_configured"


def configure_logging(settings: Settings) -> None:
    root = logging.getLogger()

    # Idempotent: remove any handler we previously installed so a second call does not
    # double-log (which would also double any redaction cost).
    for handler in list(root.handlers):
        if getattr(handler, _CONFIGURED_MARKER, False):
            root.removeHandler(handler)

    handler = logging.StreamHandler()
    setattr(handler, _CONFIGURED_MARKER, True)
    if settings.log_format == "json":
        handler.setFormatter(JsonFormatter(service=settings.service_name))
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
        )

    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())

    # Tame the noisiest third-party loggers so the signal is VoyageMesh's own events.
    for noisy in ("httpx", "httpcore", "hpack", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
