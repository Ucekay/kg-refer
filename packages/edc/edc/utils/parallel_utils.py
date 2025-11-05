import asyncio
import logging
import os
import time
from typing import Dict, List, Optional

import openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, max_requests_per_second: int):
        self.max_requests = max_requests_per_second
        self.requests = []

    async def acquire(self):
        """AAquire permission to make a request, respecting rate limits."""
        now = time.time()
        self.requests = [req_time for req_time in self.requests if now - req_time < 1.0]

        if len(self.requests) >= self.max_requests:
            sleep_time = 1.0 - (now - self.requests[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.requests.append(time.time())


class OpenAIRateLimitedProcessor:
    """Processor for handling OpenAI requests with rate limiting and concurrency control."""

    def __init__(
        self,
        max_concurrent: int = 5,
        max_requests_per_second: int = 200,
        max_tokens: int = 1000,
    ) -> None:
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limiter = RateLimiter(max_requests_per_second)
        self.max_tokens = max_tokens
