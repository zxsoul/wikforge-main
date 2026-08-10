"""检索质量评估模块（任务 25.6 / 25.7）。

包含：
- ``qa_pairs.json``：标注的问答对 + 相关文档/chunk 列表
- ``metrics.py``：Recall@K、MRR、NDCG 等指标实现
- ``run_eval.py``：基于 ``SearchService`` 跑评测的脚手架
"""
