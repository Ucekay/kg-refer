import asyncio
import logging
import os
import time

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, Field


class Triplet(BaseModel):
    h: str = Field(description="Head entity of the triplet")
    r: str = Field(description="Relation of the triplet")
    t: str = Field(description="Tail entity of the triplet")


class KnowledgeBase(BaseModel):
    triplets: list[Triplet] = Field(description="List of extracted triplets")


class RelationDefinition(BaseModel):
    relation: str = Field(description="Name of the relation")
    definition: str = Field(description="Definition of the relation")


class RelationDefinitions(BaseModel):
    definitions: list[RelationDefinition] = Field(
        description="List of relation definitions"
    )


logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for OpenAI API requests."""

    def __init__(
        self, max_reqs_per_sec: int, rate_limit_cooldown: float = 10.0
    ) -> None:
        self.max_reqs_per_sec = max_reqs_per_sec
        self.requests = []
        self.rate_limit_cooldown = rate_limit_cooldown
        self.lock = asyncio.Lock()
        # グローバルなpause状態を管理するEvent
        self.running_event = asyncio.Event()
        self.running_event.set()  # 初期状態は実行可能

    async def acquire(self) -> None:
        """Aquire permission to make a request."""
        # まずグローバルなpause状態をチェック（ロック不要で待機可能）
        await self.running_event.wait()

        # 通常のレート制限チェック（短時間のロック）
        while True:
            async with self.lock:
                now = time.time()
                self.requests = [
                    req_time for req_time in self.requests if now - req_time < 1.0
                ]

                if len(self.requests) < self.max_reqs_per_sec:
                    self.requests.append(now)
                    return

                sleep_time = 1.0 - (now - self.requests[0])

            # ロックの外で待機（他のタスクもacquire()を進められる）
            await asyncio.sleep(sleep_time)

    async def pause_for_rate_limit(self) -> None:
        """Pause all waiting tasks due to rate limit error."""
        async with self.lock:
            # 既にpause中なら何もしない（重複pause防止）
            if not self.running_event.is_set():
                logger.info("Already paused, skipping duplicate pause")
                return

            logger.warning(
                f"Rate limit error detected. Pausing all requests for {self.rate_limit_cooldown} seconds."
            )
            self.running_event.clear()  # すべての待機中タスクを停止

        # クールダウン期間待機（ロックの外）
        await asyncio.sleep(self.rate_limit_cooldown)

        async with self.lock:
            logger.info("Rate limit cooldown finished. Resuming requests.")
            self.requests.clear()  # リクエスト履歴をクリア
            self.running_event.set()  # タスクを再開


class AsyncOpenAIProcessor:
    def __init__(self, max_concurrent=5, max_req_per_sec=70) -> None:
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
            response = None

            while response is None:
                await self.rate_limiter.acquire()

                try:
                    response = await self.async_client.responses.create(
                        model=model,
                        instructions=instructions,
                        input=input,
                        # temperature=templature,
                        max_output_tokens=max_tokens,
                        # reasoning={"effort": "minimal", "summary": None},
                    )

                except openai.RateLimitError as e:
                    logger.warning(f"OpenAI rate limit error: {e}")
                    await self.rate_limiter.pause_for_rate_limit()
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

    async def get_parsed_triplets_async(
        self,
        model: str,
        instructions: str,
        input: str,
        templature: float = 0.0,
        max_tokens: int = 1024,
    ):
        async with self.semaphore:
            response = None

            while response is None:
                await self.rate_limiter.acquire()

                try:
                    response = await self.async_client.responses.parse(
                        model=model,
                        instructions=instructions,
                        input=input,
                        # temperature=templature,
                        # max_output_tokens=max_tokens,
                        # reasoning={"effort": "minimal", "summary": None},
                        text_format=KnowledgeBase,
                    )

                except openai.RateLimitError as e:
                    logger.warning(f"OpenAI rate limit error: {e}")
                    await self.rate_limiter.pause_for_rate_limit()
                except openai.APIStatusError as e:
                    logger.warning(f"OpenAI API status error: {e}")
                    await asyncio.sleep(60)
                except openai.APIError as e:
                    logger.warning(f"OpenAI API error: {e}")
                    await asyncio.sleep(3)
                except Exception as e:
                    logger.warning(f"Unexpected error: {e}")
                    await asyncio.sleep(3)

            output = response.output_parsed

            logging.debug(
                f"Model: {model}\nInstructions: {instructions}\nInput: {input}\nResult: {output}"
            )
            return output

    async def get_parsed_definitions_async(
        self,
        model: str,
        instructions: str,
        input: str,
        templature: float = 0.0,
        max_tokens: int = 1024,
    ):
        async with self.semaphore:
            response = None

            while response is None:
                await self.rate_limiter.acquire()

                try:
                    response = await self.async_client.responses.parse(
                        model=model,
                        instructions=instructions,
                        input=input,
                        # temperature=templature,
                        # max_output_tokens=max_tokens,
                        # reasoning={"effort": "minimal", "summary": None},
                        text_format=RelationDefinitions,
                    )

                except openai.RateLimitError as e:
                    logger.warning(f"OpenAI rate limit error: {e}")
                    await self.rate_limiter.pause_for_rate_limit()
                except openai.APIStatusError as e:
                    logger.warning(f"OpenAI API status error: {e}")
                    await asyncio.sleep(60)
                except openai.APIError as e:
                    logger.warning(f"OpenAI API error: {e}")
                    await asyncio.sleep(3)
                except Exception as e:
                    logger.warning(f"Unexpected error: {e}")
                    await asyncio.sleep(3)

            output = response.output_parsed

            logging.debug(
                f"Model: {model}\nInstructions: {instructions}\nInput: {input}\nResult: {output}"
            )
            return output
