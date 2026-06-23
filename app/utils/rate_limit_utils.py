# app/utils/rate_limit_utils.py
import time
from typing import Deque
from app.core.logger import logger  # Reuse the project-wide logger.


def apply_api_rate_limit(
        request_times: Deque[float],
        max_requests: int,
        window_seconds: int = 60
) -> None:
    """
    Generic sliding-window API rate limiter.
    Maintains a deque of request timestamps and blocks when the number of
    requests inside the active window exceeds the configured limit.
    :param request_times: Deque of request timestamps, initialized externally
    :param max_requests: Maximum number of requests allowed inside the window
    :param window_seconds: Sliding-window duration in seconds, default 60
    :return: None. The function blocks when the limit is exceeded.
    """
    current_time = time.time()

    # 1. Remove expired timestamps outside the sliding window.
    while request_times and current_time - request_times[0] >= window_seconds:
        request_times.popleft()

    # 2. If the limit has been reached, wait for the remaining window time.
    if len(request_times) >= max_requests:
        # Remaining wait time = full window - age of the oldest request.
        sleep_duration = window_seconds - (current_time - request_times[0])
        if sleep_duration > 0:
            logger.debug(
                f"API rate limit reached: max {max_requests} requests in {window_seconds}s. "
                f"Waiting {sleep_duration:.2f}s."
            )
            time.sleep(sleep_duration)
            # Refresh current time and clear out timestamps that expired while waiting.
            current_time = time.time()
            while request_times and current_time - request_times[0] >= window_seconds:
                request_times.popleft()

    # 3. Record the current request inside the active window.
    request_times.append(current_time)
    logger.debug(
        f"Recorded API request timestamp. Current request count in the {window_seconds}s window: {len(request_times)}"
    )
