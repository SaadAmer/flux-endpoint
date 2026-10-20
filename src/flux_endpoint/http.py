import time
from collections.abc import Callable
from importlib.metadata import version

import httpx

JOB_API_URL = "https://api.runpod.ai/v2"
REST_API_URL = "https://api.runpod.io/v2"
DEFAULT_TIMEOUT_S = 30.0
CONNECT_RETRIES = 2

_THROTTLED = 429
_TEMPORARY_FAILURES = frozenset({502, 503, 504})
_IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})


class RetryTransport(httpx.BaseTransport):
    """Retries throttled and temporarily failing requests with exponential backoff.

    A 429 is retried for every method, because the API rejected the request
    without acting on it. A 502, 503 or 504 is retried only for idempotent
    methods, so a job is never submitted twice.
    """

    def __init__(
        self,
        transport: httpx.BaseTransport,
        *,
        attempts: int = 4,
        backoff_s: float = 0.5,
        max_delay_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._transport = transport
        self._attempts = attempts
        self._backoff_s = backoff_s
        self._max_delay_s = max_delay_s
        self._sleep = sleep

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self._transport.handle_request(request)
        for attempt in range(self._attempts - 1):
            if not self._is_retryable(request, response):
                break
            response.close()
            self._sleep(self._delay(response, attempt))
            response = self._transport.handle_request(request)
        return response

    def close(self) -> None:
        self._transport.close()

    @staticmethod
    def _is_retryable(request: httpx.Request, response: httpx.Response) -> bool:
        if response.status_code == _THROTTLED:
            return True
        return response.status_code in _TEMPORARY_FAILURES and request.method in _IDEMPOTENT_METHODS

    def _delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After", "")
        delay = float(retry_after) if retry_after.isdigit() else self._backoff_s * 2**attempt
        return min(delay, self._max_delay_s)


def create_runpod_client(
    api_key: str, base_url: str, timeout_s: float = DEFAULT_TIMEOUT_S
) -> httpx.Client:
    """Create an HTTP client for a Runpod API that retries throttled requests.

    Raises:
        ValueError: If the API key is empty.
    """
    if not api_key:
        raise ValueError("Set RUNPOD_API_KEY to a Runpod API key.")
    return httpx.Client(
        base_url=base_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": f"flux-endpoint/{version('flux-endpoint')}",
        },
        timeout=timeout_s,
        transport=RetryTransport(httpx.HTTPTransport(retries=CONNECT_RETRIES)),
    )
