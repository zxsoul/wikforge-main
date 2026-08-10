"""Redis client management."""

from redis.asyncio import Redis

from app.core.config import get_settings

settings = get_settings()

redis_client: Redis | None = None


async def get_redis() -> Redis:
    """Get or create the Redis client instance."""
    global redis_client
    if redis_client is None:
        redis_client = Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
        )
    return redis_client


async def close_redis() -> None:
    """Close the Redis connection."""
    global redis_client
    if redis_client is not None:
        await redis_client.close()
        redis_client = None
