# crawler/redis_bloom.py

import redis.asyncio as redis
from redis.exceptions import ResponseError


class RedisBloomFilter:

    def __init__(
            self,
            client: redis.Redis,
            key: str = "crawler:url_bloom",
            capacity: int = 10_000_000,
            error_rate: float = 0.001,
            expansion: int = 2,
    ):
        # Redis URL
        self.client = client

        # BF ID
        self.key = key

        # BF element capacity
        self.capacity = capacity

        # BF error rate
        self.error_rate = error_rate

        # BF expansion ratio
        self.expansion = expansion

        self._initialized = False

    # Initialize
    async def _ensure_created(self) -> None:
        if self._initialized:
            return

        try:
            # BF.RESERVE key error_rate capacity EXPANSION expansion
            await self.client.execute_command(
                "BF.RESERVE",
                self.key,
                self.error_rate,
                self.capacity,
                "EXPANSION",
                self.expansion,
            )
        except ResponseError as e:
            # The key already exists
            if "exists" not in str(e).lower():
                # Unexpected errors
                raise
        finally:
            self._initialized = True

    # Insert
    async def add(self, item: str) -> bool:
        await self._ensure_created()
        # BF.ADD key item -> 1 if new, 0 if probably existing.
        result = await self.client.execute_command("BF.ADD", self.key, item)
        return bool(result)

    # Check
    async def exists(self, item: str) -> bool:
        await self._ensure_created()
        # BF.EXISTS key item -> 1 if maybe yes, 0 if definitely no.
        result = await self.client.execute_command("BF.EXISTS", self.key, item)
        return bool(result)
