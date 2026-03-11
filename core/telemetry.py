"""Thread-safe runtime error telemetry with cooldown-based log throttling."""

import logging
import threading
import time
from collections import defaultdict, deque

from core.error_taxonomy import category_for_error_code

log = logging.getLogger("emotion-light")


class RuntimeErrorTelemetry:
    """Sammelt Runtime-Fehler strukturiert und drosselt Log-Spam."""

    def __init__(self, max_events: int = 256):
        self._lock = threading.Lock()
        self._counts: defaultdict[str, int] = defaultdict(int)
        self._last_emit: dict[str, float] = {}
        self._events: deque = deque(maxlen=max_events)

    def record(
        self,
        component: str,
        code: str,
        detail: str,
        exc: Exception | None = None,
        level: int = logging.WARNING,
        cooldown_s: float = 5.0,
    ) -> None:
        now = time.time()
        category = category_for_error_code(code)
        key = f"{component}:{code}"

        with self._lock:
            self._counts[key] += 1
            count = self._counts[key]
            self._events.append(
                {
                    "ts": now,
                    "component": component,
                    "category": category,
                    "code": code,
                    "detail": detail,
                    "count": count,
                    "exception": repr(exc) if exc is not None else None,
                }
            )
            last = float(self._last_emit.get(key, 0.0))
            should_emit = (now - last) >= cooldown_s
            if should_emit:
                self._last_emit[key] = now

        if should_emit:
            suffix = f"; exc={exc}" if exc is not None else ""
            log.log(
                level,
                "[ERR:%s] component=%s count=%d detail=%s%s",
                code,
                f"{component}/{category}",
                count,
                detail,
                suffix,
            )

    def summary(self) -> dict:
        with self._lock:
            return dict(self._counts)


ERR_TELEMETRY = RuntimeErrorTelemetry()
