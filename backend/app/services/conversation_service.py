"""问答会话历史服务（任务 16.5、16.8、需求 8.5、8.8）。

实现"问答系统应使用 Redis 存储对话历史，每个会话最多保留最近 20 轮对话，
TTL 30 分钟"（需求 8.5）以及"会话超过 30 分钟无新消息后应被标记为过期，
用户下次提问时开启新的对话会话"（需求 8.8）这两条要求。

关键设计：
- 数据结构：使用 Redis List 存储消息 JSON 串，key 为 ``qa:conv:{conversation_id}``。
- 容量上限：20 轮对话即 user+assistant 共 40 条消息。每次追加后通过
  ``LTRIM`` 自动驱逐最旧消息，保证 List 最长 40。
- 过期策略：每次 ``append`` 都用 ``EXPIRE`` 把 TTL 重置为 1800 秒（30 分钟）。
  Redis 自身会在 TTL 到期后删除整个 List，因此"标记为过期"无需额外存
  储——下次 :meth:`get_history` 返回空 list，:meth:`is_active` 返回 ``False``，
  调用方据此把会话视作"新会话"。
- 只读访问（``get_history`` / ``is_active``）不刷新 TTL，避免长时间不交互
  的会话被永远续命。
- 使用 Redis pipeline 把 ``RPUSH + LTRIM + EXPIRE`` 打包成一次往返，
  保证容量与过期一致更新。

接口为异步，与 :mod:`app.core.redis`（``redis.asyncio.Redis``）配合使用。
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ─── 常量 ────────────────────────────────────────────────────────────────

#: 会话 List 在 Redis 中的 key 前缀。
KEY_PREFIX = "qa:conv:"

#: 最多保留的轮数（一轮 = user + assistant 两条消息）。
MAX_TURNS = 20

#: 最多保留的消息条数 = 轮数 × 2。
MAX_MESSAGES = MAX_TURNS * 2

#: 会话 TTL，单位秒。需求 8.5 要求 30 分钟。
TTL_SECONDS = 1800

#: 允许的消息角色集合。
_ALLOWED_ROLES = frozenset({"user", "assistant", "system"})


class ConversationService:
    """基于 Redis 的对话历史存储。

    典型用法::

        >>> svc = ConversationService(redis_client=client)
        >>> await svc.append("conv-1", "user", "你好")
        >>> await svc.append("conv-1", "assistant", "你好，我能帮你什么？")
        >>> history = await svc.get_history("conv-1")
        >>> [m["role"] for m in history]
        ['user', 'assistant']

    线程/协程安全：依赖 ``redis.asyncio`` 客户端的并发模型；同一 conversation_id
    的并发追加在单 key 上由 Redis 串行化，但批次之间 LTRIM 之后才生效，因此
    极端并发场景下可能瞬时超过 40 条，下次 append 即收敛。
    """

    def __init__(self, redis_client: Any | None = None) -> None:
        """初始化服务。

        Args:
            redis_client: ``redis.asyncio.Redis`` 客户端。若为 ``None``，
                第一次需要使用时才通过 :func:`app.core.redis.get_redis` 获取，
                便于在测试中注入 ``fakeredis`` 或 mock 实例。
        """
        self._redis = redis_client

    # ─── 内部工具 ────────────────────────────────────────────────────────

    async def _get_redis(self) -> Any:
        """惰性获取 Redis 客户端。"""
        if self._redis is None:
            # 延迟导入避免在不需要 Redis 的测试中触发连接。
            from app.core.redis import get_redis

            self._redis = await get_redis()
        return self._redis

    @staticmethod
    def _key(conversation_id: str) -> str:
        """构造会话在 Redis 中的 key。

        简单校验防止意外传入空字符串造成所有会话共用一个 key。
        """
        if not conversation_id:
            raise ValueError("conversation_id 不能为空")
        return f"{KEY_PREFIX}{conversation_id}"

    # ─── 公共 API ────────────────────────────────────────────────────────

    async def append(
        self, conversation_id: str, role: str, content: str
    ) -> None:
        """追加一条消息到会话历史。

        - 自动驱逐最旧消息，保证 List 长度 ≤ ``MAX_MESSAGES``（40）。
        - 重置 TTL 为 ``TTL_SECONDS``（1800 秒）。
        - ``role`` 只接受 ``user`` / ``assistant`` / ``system``，其它值会抛
          ``ValueError``，以避免历史里混入非法角色而后续 LLM 拼接出错。

        Args:
            conversation_id: 会话标识，通常由调用方生成的 UUID。
            role: 消息角色，``user`` 或 ``assistant``（``system`` 仅供未来扩展）。
            content: 消息文本内容。空字符串也允许（部分 LLM 流式响应可能瞬时为空）。

        Raises:
            ValueError: ``conversation_id`` 为空或 ``role`` 非法。
        """
        if role not in _ALLOWED_ROLES:
            raise ValueError(
                f"非法的 role: {role!r}，必须是 {sorted(_ALLOWED_ROLES)} 之一"
            )

        redis = await self._get_redis()
        key = self._key(conversation_id)
        payload = json.dumps(
            {"role": role, "content": content}, ensure_ascii=False
        )

        # 用 pipeline 打包 RPUSH + LTRIM + EXPIRE，避免三次往返也避免中间态。
        # transaction=False：这里不需要 MULTI/EXEC 原子性，单次 RTT 即可。
        try:
            pipe = redis.pipeline(transaction=False)
            pipe.rpush(key, payload)
            # LTRIM 保留尾部 MAX_MESSAGES 条（相当于丢弃头部最旧的）。
            pipe.ltrim(key, -MAX_MESSAGES, -1)
            pipe.expire(key, TTL_SECONDS)
            await pipe.execute()
        except Exception:  # noqa: BLE001 - Redis 故障兜底
            logger.exception(
                "ConversationService: 追加消息失败 (conversation_id=%s, role=%s)",
                conversation_id,
                role,
            )
            raise

    async def get_history(self, conversation_id: str) -> list[dict]:
        """获取会话历史，按时间顺序从旧到新返回。

        - 不刷新 TTL，纯读取。
        - 反序列化失败的条目会被跳过并记录 warning，避免单条脏数据破坏整体读取。

        Args:
            conversation_id: 会话标识。

        Returns:
            消息字典列表，每项形如 ``{"role": "user", "content": "..."}``；
            会话不存在或已过期时返回空列表。
        """
        redis = await self._get_redis()
        key = self._key(conversation_id)
        items = await redis.lrange(key, 0, -1)

        history: list[dict] = []
        for raw in items:
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "ConversationService: 解析历史消息失败 (conversation_id=%s, raw=%r)",
                    conversation_id,
                    raw,
                )
                continue
            if isinstance(msg, dict):
                history.append(msg)
        return history

    async def clear(self, conversation_id: str) -> None:
        """清空会话历史（删除整个 List）。

        若会话不存在则静默成功。
        """
        redis = await self._get_redis()
        await redis.delete(self._key(conversation_id))

    async def ttl(self, conversation_id: str) -> int:
        """返回会话剩余 TTL（秒），便于测试与诊断。

        - 会话不存在时返回 ``-2``（与 ``Redis.TTL`` 行为一致）。
        - 会话存在但无 TTL 时返回 ``-1``（理论上不应出现）。
        """
        redis = await self._get_redis()
        return await redis.ttl(self._key(conversation_id))

    async def is_active(self, conversation_id: str) -> bool:
        """判断会话是否处于活跃状态（任务 16.8 / 需求 8.8）。

        "活跃" 的判定标准：会话 List 在 Redis 中存在（``EXISTS`` 返回 1）。

        - Redis 自动过期机制：每次 :meth:`append` 都会把 TTL 重置为
          ``TTL_SECONDS``（1800 秒）。30 分钟无新消息后，整个 List 会被
          Redis 自动删除，``EXISTS`` 返回 0，本方法相应返回 ``False``。
        - 因此调用方收到 ``False`` 等同于"会话过期或从未创建"——按需求 8.8，
          下次提问时应作为新会话起步。
        - 不刷新 TTL，纯只读。

        Args:
            conversation_id: 会话标识。

        Returns:
            ``True`` 表示会话当前活跃（未过期且存在历史）；
            ``False`` 表示会话已过期、被清理或从未创建。
        """
        redis = await self._get_redis()
        exists = await redis.exists(self._key(conversation_id))
        # ``redis.asyncio`` 的 ``EXISTS`` 返回整型；用 int() 兜底防止个别 client
        # 返回 bool，统一转布尔。
        return int(exists) > 0
