import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

import openai
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as tqdm_asyncio

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for OpenAI API requests."""

    def __init__(self, max_requests_per_second: int):
        self.max_requests = max_requests_per_second
        self.requests = []

    async def acquire(self):
        """Acquire permission to make a request, respecting rate limits."""
        now = time.time()
        # Remove requests older than 1 second
        self.requests = [req_time for req_time in self.requests if now - req_time < 1.0]

        # Check rate limit
        if len(self.requests) >= self.max_requests:
            # Wait until 1 second has passed since the oldest request
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
    ):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limiter = RateLimiter(max_requests_per_second)
        self.max_tokens = max_tokens

    async def generate_with_limits(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate completion with rate limiting and concurrency control."""
        async with self.semaphore:
            await self.rate_limiter.acquire()

            try:
                # Prepare messages with system prompt if provided
                if system_prompt:
                    full_messages = [{"role": "system", "content": system_prompt}]
                    full_messages.extend(messages)
                else:
                    full_messages = messages

                params = {
                    "model": model,
                    "messages": full_messages,
                    "temperature": temperature,
                }

                if max_tokens:
                    params["max_tokens"] = max_tokens
                elif self.max_tokens:
                    params["max_tokens"] = self.max_tokens

                response = await self.client.chat.completions.create(**params)

                return {
                    "success": True,
                    "data": response.choices[0].message.content,
                    "model": model,
                    "messages": messages,
                    "system_prompt": system_prompt,
                    "prompt": messages[-1]["content"] if messages else "",
                }
            except openai.APIError as e:
                logger.error(f"OpenAI API error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "model": model,
                    "messages": messages,
                    "prompt": messages[-1]["content"] if messages else "",
                }
            except asyncio.TimeoutError:
                logger.error(f"Request timeout for model {model}")
                return {
                    "success": False,
                    "error": "Request timeout",
                    "model": model,
                    "messages": messages,
                    "prompt": messages[-1]["content"] if messages else "",
                }
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "model": model,
                    "messages": messages,
                    "prompt": messages[-1]["content"] if messages else "",
                }

    async def process_prompts(
        self,
        model: str,
        prompts: List[str],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process multiple prompts in parallel with rate limiting."""
        start_time = time.time()

        # Convert prompts to message format with index tracking
        indexed_tasks = []  # Store (original_index, task) pairs
        for i, prompt in enumerate(prompts):
            messages = [{"role": "user", "content": prompt}]
            task = self.generate_with_limits(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )
            indexed_tasks.append((i, task))

        # Show progress bar while processing - maintain order using index tracking
        results = [None] * len(prompts)  # Pre-allocate with None to maintain order
        for completed_task in tqdm_asyncio.as_completed(
            [task for _, task in indexed_tasks],  # Extract only tasks for as_completed
            total=len(prompts),
            desc=f"Processing {len(prompts)} prompts",
        ):
            result = await completed_task
            # Find original index for this completed task
            for original_index, original_task in indexed_tasks:
                if original_task is completed_task:
                    results[original_index] = result
                    break
        end_time = time.time()

        # Categorize results
        successful = []
        failed = []

        for result in results:
            if isinstance(result, dict):
                if result.get("success"):
                    successful.append(result)
                else:
                    failed.append(result)
            else:
                failed.append({"success": False, "error": str(result)})

        total_time = end_time - start_time
        actual_rps = len(prompts) / total_time if total_time > 0 else 0

        stats = {
            "total_requests": len(prompts),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate": len(successful) / len(prompts) * 100 if prompts else 0,
            "total_time": total_time,
            "requests_per_second": actual_rps,
        }

        logger.info(f"Parallel processing completed: {stats}")

        return {
            "successful": successful,
            "failed": failed,
            "stats": stats,
        }

    async def process_messages(
        self,
        model: str,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process multiple message lists in parallel with rate limiting."""
        start_time = time.time()

        tasks = []
        for messages in messages_list:
            task = self.generate_with_limits(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )
            tasks.append(task)

        # Show progress bar while processing
        results = []
        for task in tqdm_asyncio.as_completed(
            tasks, total=len(tasks), desc=f"Processing {len(messages_list)} messages"
        ):
            result = await task
            results.append(result)
        end_time = time.time()

        # Categorize results
        successful = []
        failed = []

        for result in results:
            if isinstance(result, dict):
                if result.get("success"):
                    successful.append(result)
                else:
                    failed.append(result)
            else:
                failed.append({"success": False, "error": str(result)})

        total_time = end_time - start_time
        actual_rps = len(messages_list) / total_time if total_time > 0 else 0

        stats = {
            "total_requests": len(messages_list),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate": len(successful) / len(messages_list) * 100
            if messages_list
            else 0,
            "total_time": total_time,
            "requests_per_second": actual_rps,
        }

        logger.info(f"Parallel processing completed: {stats}")

        return {
            "successful": successful,
            "failed": failed,
            "stats": stats,
        }


# Global processor instance for reuse
_global_processor: Optional[OpenAIRateLimitedProcessor] = None


def get_global_processor(
    max_concurrent: int = 5,
    max_requests_per_second: int = 200,
    max_tokens: int = 1000,
) -> OpenAIRateLimitedProcessor:
    """Get or create a global processor instance."""
    global _global_processor
    if _global_processor is None:
        _global_processor = OpenAIRateLimitedProcessor(
            max_concurrent=max_concurrent,
            max_requests_per_second=max_requests_per_second,
            max_tokens=max_tokens,
        )
    return _global_processor


async def process_openai_requests_parallel(
    model: str,
    prompts: List[str],
    max_concurrent: int = 5,
    max_requests_per_second: int = 200,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function for processing OpenAI requests in parallel."""
    processor = get_global_processor(
        max_concurrent=max_concurrent,
        max_requests_per_second=max_requests_per_second,
        max_tokens=max_tokens,
    )

    return await processor.process_prompts(
        model=model,
        prompts=prompts,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )


async def process_openai_messages_parallel(
    model: str,
    messages_list: List[List[Dict[str, str]]],
    max_concurrent: int = 5,
    max_requests_per_second: int = 200,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function for processing OpenAI messages in parallel."""
    processor = get_global_processor(
        max_concurrent=max_concurrent,
        max_requests_per_second=max_requests_per_second,
        max_tokens=max_tokens,
    )

    return await processor.process_messages(
        model=model,
        messages_list=messages_list,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )
