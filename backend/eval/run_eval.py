"""检索质量评估脚本（任务 25.7）。

用法：
    python -m backend.eval.run_eval --backend stub
    python -m backend.eval.run_eval --backend live --user-id <uid>

- ``stub`` 模式：使用一个内置的查询→chunk_id 映射，仅用于流水线 smoke
- ``live`` 模式：连接已部署的后端，调用 ``SearchService`` 真实检索

输出聚合的 Recall@K / MRR / NDCG@K 数值。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from backend.eval.metrics import EvalSample, evaluate


QA_PATH = Path(__file__).resolve().parent / "qa_pairs.json"


def load_samples(path: Path = QA_PATH) -> list[EvalSample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalSample(
            query_id=item["id"],
            query=item["query"],
            relevant_chunk_ids=list(item["relevant_chunk_ids"]),
        )
        for item in data["samples"]
    ]


def stub_retrieve(query: str) -> list[str]:
    """占位检索函数：根据查询关键字返回固定的 chunk_id 序列，仅用于 smoke。"""
    mapping = {
        "RAG": ["doc-rag-overview-c0", "doc-rag-architecture-c2"],
        "BM25": ["doc-bm25-c1", "doc-vector-c0"],
        "RRF": ["doc-rrf-c0", "doc-rrf-c1"],
        "Cross-Encoder": ["doc-rerank-c0", "doc-rerank-c1"],
        "权限": ["doc-permission-c0", "doc-permission-c2"],
        "PDF": ["doc-table-c0", "doc-pdf-c2"],
        "状态机": ["doc-pipeline-c0"],
        "改写": ["doc-query-enhance-c0", "doc-query-enhance-c1"],
        "评分": ["doc-quality-c0", "doc-quality-c1"],
        "Profile": ["doc-profile-c0", "doc-profile-c1"],
    }
    for k, v in mapping.items():
        if k in query:
            return v
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Wikforge retrieval eval")
    parser.add_argument(
        "--backend",
        choices=["stub", "live"],
        default="stub",
        help="使用 stub（内置占位）或 live（真实 SearchService）",
    )
    parser.add_argument("--user-id", default="", help="live 模式需要的用户 ID")
    parser.add_argument(
        "--k", nargs="+", type=int, default=[1, 5, 10, 20],
    )
    args = parser.parse_args()

    samples = load_samples()
    if args.backend == "stub":
        retrieve_fn = stub_retrieve
    else:  # live
        # live 模式仅在有完整环境时可用，避免在单测环境强制依赖。
        from app.services.embedding_service import EmbeddingService
        from app.services.search_service import SearchService
        import asyncio

        if not args.user_id:
            print("--user-id 在 live 模式下必填")
            return 1
        service = SearchService(embedding_service=EmbeddingService())

        async def _live_async(query: str) -> list[str]:
            allowed = os.environ.get("WIKFORGE_EVAL_SPACES", "").split(",")
            allowed = [s for s in allowed if s]
            resp = await service.search(
                query=query,
                user_id=args.user_id,
                allowed_space_ids=allowed,
                page=1,
                page_size=max(args.k),
            )
            return [r.chunk_id for r in resp.results]

        def retrieve_fn(query: str) -> list[str]:
            return asyncio.run(_live_async(query))

    result = evaluate(samples, retrieve_fn, k_values=tuple(args.k))
    print(f"Samples: {result.num_samples}")
    print(f"MRR: {result.mrr:.4f}")
    for k in sorted(result.recall_at_k.keys()):
        print(f"Recall@{k}: {result.recall_at_k[k]:.4f}")
    for k in sorted(result.ndcg_at_k.keys()):
        print(f"NDCG@{k}: {result.ndcg_at_k[k]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
