import asyncio
import logging
import os
import time

import openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for OpenAI API requests."""

    def __init__(self, max_reqs_per_sec: int) -> None:
        self.max_reqs_per_sec = max_reqs_per_sec
        self.requests = []

    async def acquire(self) -> None:
        """Aquire permission to make a request."""
        now = time.time()

        self.requests = [req_time for req_time in self.requests if now - req_time < 1.0]

        if len(self.requests) >= self.max_reqs_per_sec:
            sleep_time = 1.0 - (now - self.requests[0])
            await asyncio.sleep(sleep_time)

        self.requests.append(time.time())


class AsyncOpenAIProcessor:
    def __init__(self, max_concurrent=5, max_req_per_sec=80) -> None:
        self.async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limiter = RateLimiter(max_req_per_sec)

    async def openai_responses_async(
        self,
        model: str,
        instructions: str,
        input: str,
        templature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        async with self.semaphore:
            await self.rate_limiter.acquire()

            response = None

            while response is None:
                try:
                    response = await self.async_client.responses.create(
                        model=model,
                        instructions=instructions,
                        input=input,
                        temperature=templature,
                        max_output_tokens=max_tokens,
                        # reasoning={"effort": "minimal"},
                    )

                except openai.APIStatusError as e:
                    logger.warning(f"OpenAI API status error: {e}")
                    await asyncio.sleep(60)
                except openai.APIError as e:
                    logger.warning(f"OpenAI API error: {e}")
                    await asyncio.sleep(3)
                except Exception as e:
                    logger.warning(f"Unexpected error: {e}")
                    await asyncio.sleep(3)

            output_text = response.output_text
            logging.debug(
                f"Model: {model}\nInstructions: {instructions}\nInput: {input}\nResult: {output_text}"
            )
            return output_text
