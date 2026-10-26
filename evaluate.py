from __future__ import annotations

import json
import math
import re
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# 复用 hybrid_ask.py 中的检索与生成组件
from hybrid_ask import (
    hybrid_search,
    qa_chain,
    format_context,
    llm,
    embeddings,
)


# ============================================================
# 0. 配置
# ============================================================

EVAL_FILE = Path("eval_queries.json")
RECALL_K = 5
DIAGNOSTIC_TOP_N = 5          # 诊断时打印前 N 条检索结果
DETERMINISM_RUNS = 3          # 确定性检查重复次数
ENABLE_RAGAS = True           # 是否跑 RAGAS（需要裁判 LLM）
ENABLE_DETERMINISM = True     # 是否跑确定性检查


# ============================================================
# 1. 文本归一化 & 相关性判定
# ============================================================

def normalize_text(s: str) -> str:
    """统一全半角、小写、去所有空白，用于宽松子串匹配。"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"\s+", "", s)
    return s


def is_relevant(doc, item) -> bool:
    """
    判定单个检索结果是否与标注相关。
    优先级：
      1. relevant_chunk_ids 精确匹配 _chroma_id
      2. relevant_snippets 归一化后子串匹配
    """
    chunk_ids = item.get("relevant_chunk_ids")
    if chunk_ids:
        return doc.metadata.get("_chroma_id") in chunk_ids

    doc_norm = normalize_text(doc.page_content)
    for s in item.get("relevant_snippets", []):
        if s and normalize_text(s) in doc_norm:
            return True
    return False


def audit_annotation(retrieved_docs, item, k=5):
    """
    只做标注审计，不修改原有 Recall/MRR。
    用于定位标注 ID 或 snippet 与检索结果不一致的问题。
    """
    top_docs = retrieved_docs[:k]
    relevant_ids = set(item.get("relevant_chunk_ids") or [])
    snippets = item.get("relevant_snippets") or []

    print("\n  [标注审计]")
    print(f"  标注 ID 数量: {len(relevant_ids)}")
    print(f"  标注 snippet 数量: {len(snippets)}")

    # 1. 检查 ID 命中
    retrieved_ids = {
        doc.metadata.get("_chroma_id")
        for doc in top_docs
    }

    if relevant_ids:
        hit_ids = relevant_ids & retrieved_ids
        missing_ids = relevant_ids - retrieved_ids
        print(f"  ID 命中: {len(hit_ids)}/{len(relevant_ids)}")
        if missing_ids:
            print(f"  前{k}未命中的标注 ID: {sorted(missing_ids)}")

    # 2. 每条 snippet 单独检查
    for idx, snippet in enumerate(snippets, start=1):
        snippet_norm = normalize_text(snippet)

        matched = []
        if snippet_norm:
            for rank, doc in enumerate(top_docs, start=1):
                content_norm = normalize_text(doc.page_content)
                if snippet_norm in content_norm:
                    matched.append(
                        (rank, doc.metadata.get("_chroma_id", "?"))
                    )

        if matched:
            print(f"  snippet[{idx}]：文本命中 {matched}")
        else:
            print(f"  snippet[{idx}]：未命中前{k}")
            print(f"    标注文本: {snippet!r}")

    # 3. 输出可供人工核验的完整候选内容
    print("  候选 chunk 内容：")
    for rank, doc in enumerate(top_docs, start=1):
        cid = doc.metadata.get("_chroma_id", "?")
        section = doc.metadata.get("section", "")
        print(f"    rank={rank}, id={cid}, section={section}")
        print(f"      {doc.page_content[:500]!r}")


def count_relevant_annotations(item) -> int:
    """标注中声明的相关文档总数。"""
    ids = item.get("relevant_chunk_ids")
    if ids:
        return len(ids)
    return len(item.get("relevant_snippets", []))


# ============================================================
# 2. 检索指标
# ============================================================

def compute_recall_at_k(retrieved_docs, item, k: int = RECALL_K) -> Optional[float]:
    """
    Recall@k：
      前 k 个结果中命中的相关文档数 / 标注中声明的相关文档总数
    若标注未声明相关文档，返回 None（跳过该样本）。
    """
    total_relevant = count_relevant_annotations(item)
    if total_relevant == 0:
        return None

    top_k = retrieved_docs[:k]
    hit = sum(1 for doc in top_k if is_relevant(doc, item))
    # 命中数不应超过标注总数（同一块重复出现时去重）
    return min(hit, total_relevant) / total_relevant


def compute_rr(retrieved_docs, item) -> float:
    """Reciprocal Rank：第一个相关结果的排名倒数。"""
    for rank, doc in enumerate(retrieved_docs, start=1):
        if is_relevant(doc, item):
            return 1.0 / rank
    return 0.0


# ============================================================
# 3. 统计工具
# ============================================================

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(name: str, xs: List[float]) -> None:
    """均值 + 中位数 + p25 + p75，方差大时比单看均值可靠。"""
    if not xs:
        print(f"{name}: 无有效数据")
        return
    xs_sorted = sorted(xs)
    n = len(xs)
    p25 = xs_sorted[max(0, int(n * 0.25) - 1)]
    p50 = statistics.median(xs_sorted)
    p75 = xs_sorted[min(n - 1, int(n * 0.75))]
    print(
        f"{name}: mean={mean(xs):.4f}  median={p50:.4f}  "
        f"p25={p25:.4f}  p75={p75:.4f}  n={n}"
    )


def print_distribution(name: str, xs: List[float]) -> None:
    """打印取值分布，直观看 0/0.5/1 各占多少。"""
    buckets = defaultdict(int)
    for x in xs:
        buckets[round(x, 2)] += 1
    print(f"\n{name} 分布：")
    for val in sorted(buckets):
        count = buckets[val]
        bar = "█" * count
        print(f"  {val:>5.2f} | {count:>3} {bar}")


# ============================================================
# 4. 确定性检查（可选）
# ============================================================

def check_determinism(question: str, runs: int = DETERMINISM_RUNS) -> bool:
    """同一问题跑多次，检查返回的 _chroma_id 序列是否完全一致。"""
    results = []
    for _ in range(runs):
        docs = hybrid_search(question)
        results.append([d.metadata.get("_chroma_id") for d in docs])

    stable = all(r == results[0] for r in results)
    print(f"  [{'稳定' if stable else '抖动'}] {question}")
    if not stable:
        for i, r in enumerate(results):
            print(f"    run{i}: {r}")
    return stable


# ============================================================
# 5. 主流程：检索 + 生成 + 诊断
# ============================================================

def main() -> None:
    # ---------- 加载标注 ----------
    if not EVAL_FILE.exists():
        raise FileNotFoundError(f"未找到评估文件：{EVAL_FILE}")

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    print(f"加载了 {len(eval_items)} 条评估问题")

    # 检查标注完整性
    missing_gt = sum(
        1 for it in eval_items
        if not it.get("relevant_chunk_ids") and not it.get("relevant_snippets")
    )
    if missing_gt:
        print(
            f"⚠️  {missing_gt} 条样本没有任何相关性标注，"
            f"将被跳过 Recall 计算（只算 MRR，MRR 会恒为 0）。"
        )

    # ---------- 确定性检查（可选） ----------
    if ENABLE_DETERMINISM:
        print("\n" + "=" * 60)
        print("确定性检查")
        print("=" * 60)
        unstable = 0
        for item in eval_items:
            if not check_determinism(item["question"]):
                unstable += 1
        print(f"\n抖动样本数：{unstable}/{len(eval_items)}")

    # ---------- 逐条检索 + 生成 ----------
    print("\n" + "=" * 60)
    print("检索与生成")
    print("=" * 60)

    all_retrieved = []
    ragas_samples = []

    for i, item in enumerate(eval_items, start=1):
        question = item["question"]
        print(f"\n[{i}/{len(eval_items)}] {question}")

        retrieved = hybrid_search(question)
        all_retrieved.append(retrieved)

        r = compute_recall_at_k(retrieved, item, k=RECALL_K)
        rr = compute_rr(retrieved, item)

        r_str = f"{r:.3f}" if r is not None else "跳过（无标注）"
        print(f"  Recall@{RECALL_K}={r_str}  RR={rr:.3f}")

        context = format_context(retrieved)
        answer = qa_chain.invoke({"context": context, "question": question})

        ragas_samples.append({
            "question": question,
            "answer": answer,
            "contexts": [doc.page_content for doc in retrieved],
            "ground_truth": item.get("ground_truth", ""),
        })

        audit_annotation(retrieved, item, k=RECALL_K)

    # ---------- 逐条诊断 ----------
    print("\n" + "=" * 60)
    print("逐条诊断")
    print("=" * 60)

    for i, (item, retrieved) in enumerate(zip(eval_items, all_retrieved), start=1):
        q = item["question"]
        r = compute_recall_at_k(retrieved, item, k=RECALL_K)
        rr = compute_rr(retrieved, item)

        r_str = f"{r:.3f}" if r is not None else "跳过"
        print(f"\n[{i}] {q}")
        print(f"  Recall@{RECALL_K}={r_str}  RR={rr:.3f}")

        for rank, doc in enumerate(retrieved[:DIAGNOSTIC_TOP_N], start=1):
            rel = is_relevant(doc, item)
            cid = doc.metadata.get("_chroma_id", "?")
            preview = doc.page_content[:80].replace("\n", " ")
            flag = "✅" if rel else "  "
            print(f"    {flag} rank{rank} id={cid} | {preview}")

        # 0 分时提示：标注 snippet 是否根本没被召回
        if r == 0 and item.get("relevant_snippets"):
            print(f"  ⚠️  标注 snippet 未命中前{RECALL_K}，检查是否在库中：")
            for s in item["relevant_snippets"]:
                print(f"      - {s[:60]}")

    # ---------- 汇总检索指标 ----------
    recall_scores = []
    rr_scores = []
    for item, retrieved in zip(eval_items, all_retrieved):
        r = compute_recall_at_k(retrieved, item, k=RECALL_K)
        if r is not None:
            recall_scores.append(r)
        rr_scores.append(compute_rr(retrieved, item))

    print("\n" + "=" * 60)
    print("检索指标汇总")
    print("=" * 60)
    summarize(f"Recall@{RECALL_K}", recall_scores)
    summarize("MRR", rr_scores)
    print_distribution(f"Recall@{RECALL_K}", recall_scores)
    print_distribution("MRR", rr_scores)

    # ---------- 按分类统计 ----------
    print("\n" + "=" * 60)
    print("按分类统计")
    print("=" * 60)

    groups = defaultdict(lambda: {"recall": [], "rr": []})
    for item, retrieved in zip(eval_items, all_retrieved):
        cat = item.get("category", "未分类")
        r = compute_recall_at_k(retrieved, item, k=RECALL_K)
        rr = compute_rr(retrieved, item)
        if r is not None:
            groups[cat]["recall"].append(r)
        groups[cat]["rr"].append(rr)

    print(f"{'分类':<20} {'Recall@K':>10} {'MRR':>10} {'样本数':>8}")
    print("-" * 54)
    for cat in sorted(groups):
        vals = groups[cat]
        rc = mean(vals["recall"]) if vals["recall"] else 0.0
        mr = mean(vals["rr"])
        n = len(vals["rr"])
        print(f"{cat:<20} {rc:>10.3f} {mr:>10.3f} {n:>8}")


if __name__ == "__main__":
    main()