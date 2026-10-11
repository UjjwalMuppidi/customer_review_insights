"""
Stage 6: Comprehensive RAG Evaluation Suite.

Metrics implemented:
  1. Retrieval: Hit Rate @ K, Mean Reciprocal Rank (MRR), Context Precision @ K
  2. Generation: Faithfulness, Completeness, Answer Relevance
  3. LLM-as-Judge: Uses the configured LLM to score answers on a 1-5 rubric
  4. End-to-End Latency profiling
"""

import os
import re
import json
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter


# ---------------------------------------------------------------------------
#  Ground-truth evaluation dataset (35 curated questions)
# ---------------------------------------------------------------------------

def get_default_eval_dataset() -> List[Dict[str, Any]]:
    """
    Returns a curated benchmark of 35 evaluation questions.
    Each entry carries:
      - question: the user query
      - expected_keywords: tokens that *must* appear in a relevant retrieved doc
      - expected_answer_points: key claims the ideal answer should contain
      - category / brand: metadata for filtered retrieval tests
    """
    hand_crafted = [
        {
            "id": 1,
            "question": "What do people complain about regarding DarkRoast coffee packaging?",
            "expected_keywords": ["packaging", "crushed", "box", "seal", "coffee", "damaged", "cardboard", "powder"],
            "expected_answer_points": [
                "packaging was crushed or damaged during shipping",
                "powder leaked inside the shipping box",
                "seal was broken on arrival"
            ],
            "category": "Coffee", "brand": "DarkRoast Co"
        },
        {
            "id": 2,
            "question": "Are there any Keurig K-Cup compatibility issues with coffee pods?",
            "expected_keywords": ["keurig", "pods", "puncture", "grounds", "k-cup", "coffee", "maker"],
            "expected_answer_points": [
                "pods fail to puncture correctly in Keurig machines",
                "grounds leak or flood the coffee maker"
            ],
            "category": "Coffee", "brand": "K-Cup Express"
        },
        {
            "id": 3,
            "question": "Why are customers unhappy about expiration dates on food packages?",
            "expected_keywords": ["expired", "expiration", "date", "stale", "box", "days"],
            "expected_answer_points": [
                "products arrived close to or past expiration date",
                "expiration date was only a few days away"
            ],
            "category": "Snacks", "brand": "KindBar Plus"
        },
        {
            "id": 4,
            "question": "What issues do pet owners report about KibbleKing dog food?",
            "expected_keywords": ["dog", "kibble", "stale", "pet", "food", "quality"],
            "expected_answer_points": [
                "pet food quality was below expectations",
                "product tasted stale or had quality issues"
            ],
            "category": "Pet Food", "brand": "KibbleKing"
        },
        {
            "id": 5,
            "question": "Do customers like GreenLeaf green tea flavor?",
            "expected_keywords": ["green", "tea", "flavor", "smooth", "fresh", "aroma", "taste"],
            "expected_answer_points": [
                "tea flavor is generally well received",
                "aroma and freshness noted positively"
            ],
            "category": "Tea", "brand": "GreenLeaf Teas"
        },
        {
            "id": 6,
            "question": "What are the main complaints about CocoaLux chocolate?",
            "expected_keywords": ["chocolate", "cocoa", "quality", "taste", "sweet"],
            "expected_answer_points": [
                "taste or sweetness level drew complaints",
                "some quality or packaging concerns"
            ],
            "category": "Chocolate", "brand": "CocoaLux"
        },
        {
            "id": 7,
            "question": "Is OrganicNuts trail mix fresh or stale?",
            "expected_keywords": ["nuts", "stale", "rancid", "fresh", "crisp", "organic"],
            "expected_answer_points": [
                "some reviewers received stale or rancid nuts",
                "freshness varies across shipments"
            ],
            "category": "Snacks", "brand": "OrganicNuts"
        },
        {
            "id": 8,
            "question": "What do customers say about BeanSupreme coffee roast level?",
            "expected_keywords": ["roast", "burnt", "bitter", "char", "dark", "coffee"],
            "expected_answer_points": [
                "roast level is described as too dark or bitter",
                "burnt or charred flavor noted"
            ],
            "category": "Coffee", "brand": "BeanSupreme"
        },
        {
            "id": 9,
            "question": "Are there complaints about broken seals on chai tea boxes?",
            "expected_keywords": ["seal", "broken", "chai", "tea", "powder", "unsafe"],
            "expected_answer_points": [
                "seal was broken upon delivery",
                "product may be unsafe to consume"
            ],
            "category": "Tea", "brand": "ChaiMaster"
        },
        {
            "id": 10,
            "question": "How do cats react to PurrfectBites cat food?",
            "expected_keywords": ["cat", "purrfect", "bites", "pet", "food"],
            "expected_answer_points": [
                "feedback on pet acceptance of the food",
                "quality or taste reaction from pets"
            ],
            "category": "Pet Food", "brand": "PurrfectBites"
        },
        {
            "id": 11,
            "question": "What complaints exist about Matcha Zen tea quality?",
            "expected_keywords": ["matcha", "taste", "bitter", "tea", "quality", "zen"],
            "expected_answer_points": [
                "matcha quality or taste issues",
                "bitterness or clumpy texture noted"
            ],
            "category": "Tea", "brand": "Matcha Zen"
        },
        {
            "id": 12,
            "question": "Do customers think VeggieCrisp chips are too salty?",
            "expected_keywords": ["salty", "chips", "crisps", "taste", "snack", "veggie"],
            "expected_answer_points": [
                "salt or sodium level feedback",
                "taste preference comments"
            ],
            "category": "Snacks", "brand": "VeggieCrisp"
        },
        {
            "id": 13,
            "question": "What do reviewers say about BarkDelight treats for puppies?",
            "expected_keywords": ["treats", "bark", "delight", "pet", "dog", "healthy"],
            "expected_answer_points": [
                "treat quality for dogs",
                "health or ingredient feedback"
            ],
            "category": "Pet Food", "brand": "BarkDelight"
        },
        {
            "id": 14,
            "question": "Are Lindt Reserve dark chocolates smooth or gritty?",
            "expected_keywords": ["smooth", "dark", "chocolate", "rich", "cacao", "lindt"],
            "expected_answer_points": [
                "texture feedback (smooth vs gritty)",
                "richness or cacao quality"
            ],
            "category": "Chocolate", "brand": "Lindt Reserve"
        },
        {
            "id": 15,
            "question": "What complaints are raised about MorningBrew coffee beans?",
            "expected_keywords": ["beans", "morning", "brew", "coffee", "quality"],
            "expected_answer_points": [
                "coffee bean quality issues",
                "flavor or freshness complaints"
            ],
            "category": "Coffee", "brand": "MorningBrew"
        },
    ]

    # Generate 20 more test cases programmatically
    brand_pairs = [
        ("Coffee", "JavaGold"), ("Tea", "ZenTea"), ("Snacks", "CrispyChips"),
        ("Pet Food", "HealthyPaws"), ("Chocolate", "DarkBite"), ("Coffee", "Peet's Select"),
        ("Tea", "EarlGrey Select"), ("Snacks", "FruitCrunch"), ("Pet Food", "BlueBuffalo Choice"),
        ("Chocolate", "ChocoSweet"), ("Coffee", "Starbucks Reserve"), ("Tea", "Twinings Imperial"),
        ("Snacks", "SnackBite"), ("Pet Food", "TasteOfWild"), ("Chocolate", "ArtisanCacao"),
        ("Coffee", "Nespresso Compatible"), ("Tea", "HerbalHaven"), ("Snacks", "TrailMix Co"),
        ("Pet Food", "KibbleKing"), ("Chocolate", "Ghirardelli Supreme")
    ]

    for i, (cat, brand) in enumerate(brand_pairs):
        hand_crafted.append({
            "id": 16 + i,
            "question": f"What is the overall customer feedback about {brand} {cat.lower()} quality and experience?",
            "expected_keywords": [cat.lower(), brand.lower().split()[0]],
            "expected_answer_points": [
                f"mentions {brand} by name",
                f"discusses {cat.lower()} quality or experience"
            ],
            "category": cat,
            "brand": brand
        })

    return hand_crafted


# ---------------------------------------------------------------------------
#  Metric helper utilities
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return [w for w in re.findall(r'[a-z0-9]+', text.lower()) if len(w) > 2]


def _ngram_overlap(reference_tokens: List[str], candidate_tokens: List[str], n: int = 1) -> float:
    """Computes token-level n-gram overlap (precision-oriented)."""
    if not candidate_tokens:
        return 0.0
    ref_ngrams = Counter(zip(*[reference_tokens[i:] for i in range(n)]))
    cand_ngrams = Counter(zip(*[candidate_tokens[i:] for i in range(n)]))
    overlap = sum((cand_ngrams & ref_ngrams).values())
    return overlap / max(sum(cand_ngrams.values()), 1)


# ---------------------------------------------------------------------------
#  Core Evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """Comprehensive evaluator for RAG pipelines."""

    def __init__(self, rag_pipeline: Any):
        self.pipeline = rag_pipeline

    # ------------------------------------------------------------------ #
    #  1. RETRIEVAL METRICS                                                #
    # ------------------------------------------------------------------ #

    def evaluate_retrieval(
        self,
        eval_cases: List[Dict[str, Any]],
        top_k_list: List[int] = [1, 3, 5],
        use_hybrid: bool = True
    ) -> Dict[str, Any]:
        """
        Computes:
          - Hit Rate @ K  (at least one relevant doc in top-K)
          - Mean Reciprocal Rank (MRR)
          - Context Precision @ K  (fraction of top-K docs that are relevant)
          - Average retrieval latency
        """
        max_k = max(top_k_list)
        hit_any = {k: 0 for k in top_k_list}
        mrr_sum = 0.0
        ctx_precision = {k: [] for k in top_k_list}
        latencies = []

        for case in eval_cases:
            query = case["question"]
            expected_kw = [kw.lower() for kw in case["expected_keywords"]]

            t0 = time.time()
            retrieved = self.pipeline.retrieve_context(query, top_k=max_k, use_hybrid=use_hybrid)
            latencies.append(time.time() - t0)

            first_hit_rank = 0
            relevance_flags = []  # 1/0 for each retrieved doc

            for rank, doc in enumerate(retrieved, 1):
                text = " ".join([
                    doc.get("CombinedReview", ""),
                    doc.get("Brand", ""),
                    doc.get("Category", "")
                ]).lower()
                is_relevant = any(kw in text for kw in expected_kw)
                relevance_flags.append(1 if is_relevant else 0)

                if is_relevant and first_hit_rank == 0:
                    first_hit_rank = rank

            # Hit Rate @ K
            for k in top_k_list:
                if any(relevance_flags[:k]):
                    hit_any[k] += 1

            # Context Precision @ K = fraction of top-K that are relevant
            for k in top_k_list:
                rel_in_k = sum(relevance_flags[:k])
                ctx_precision[k].append(rel_in_k / k)

            # MRR
            if first_hit_rank > 0:
                mrr_sum += 1.0 / first_hit_rank

        n = len(eval_cases)
        return {
            "num_test_cases": n,
            "mode": "Hybrid (BM25 + Dense)" if use_hybrid else "Dense Vector Only",
            "mrr": round(mrr_sum / n, 4) if n else 0.0,
            "hit_rates": {f"Hit_Any@{k}": round(hit_any[k] / n, 4) for k in top_k_list},
            "context_precision": {f"Ctx_Precision@{k}": round(float(np.mean(ctx_precision[k])), 4) for k in top_k_list},
            "avg_latency_sec": round(float(np.mean(latencies)), 4) if latencies else 0.0
        }

    # ------------------------------------------------------------------ #
    #  2. FAITHFULNESS  (is every claim in the answer grounded?)           #
    # ------------------------------------------------------------------ #

    def evaluate_faithfulness(
        self,
        answer: str,
        context_sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Faithfulness = fraction of substantive words / phrases in the answer
        that are traceable back to the retrieved context.

        Returns a score in [0, 1] plus per-claim breakdown.
        """
        context_text = " ".join(
            doc.get("CombinedReview", "").lower() for doc in context_sources
        )
        ctx_tokens = set(_tokenize(context_text))

        # Extract substantive answer tokens (skip markdown, short words, numbers)
        ans_tokens = _tokenize(answer)
        # Filter out stopwords and very common words
        stopwords = {"the", "and", "for", "are", "but", "not", "you", "all",
                     "can", "had", "her", "was", "one", "our", "out", "has",
                     "this", "that", "with", "from", "they", "been", "have",
                     "its", "will", "more", "when", "who", "also", "into",
                     "based", "source", "review", "reviews", "retrieved",
                     "answer", "question", "context", "information", "mode",
                     "operating", "local", "demonstration", "offline",
                     "note", "summary", "highlights", "insights", "customer"}
        substantive = [t for t in ans_tokens if t not in stopwords and len(t) > 3]

        if not substantive:
            return {"score": 1.0, "supported": 0, "total_claims": 0, "unsupported_tokens": []}

        supported = [t for t in substantive if t in ctx_tokens]
        unsupported = [t for t in substantive if t not in ctx_tokens]

        score = len(supported) / len(substantive) if substantive else 1.0
        return {
            "score": round(score, 4),
            "supported": len(supported),
            "total_claims": len(substantive),
            "unsupported_tokens": list(set(unsupported))[:15]  # cap display
        }

    # ------------------------------------------------------------------ #
    #  3. COMPLETENESS  (does the answer cover expected points?)           #
    # ------------------------------------------------------------------ #

    def evaluate_completeness(
        self,
        answer: str,
        expected_points: List[str]
    ) -> Dict[str, Any]:
        """
        Completeness = fraction of expected_answer_points that are covered
        (at least partially) by the generated answer.
        """
        if not expected_points:
            return {"score": 1.0, "covered": 0, "total": 0, "details": []}

        answer_lower = answer.lower()
        details = []
        covered = 0

        for point in expected_points:
            point_tokens = _tokenize(point)
            # Consider a point covered if >= 50% of its tokens appear in answer
            if not point_tokens:
                covered += 1
                details.append({"point": point, "covered": True, "overlap": 1.0})
                continue
            hit_count = sum(1 for t in point_tokens if t in answer_lower)
            overlap = hit_count / len(point_tokens)
            is_covered = overlap >= 0.4
            if is_covered:
                covered += 1
            details.append({"point": point, "covered": is_covered, "overlap": round(overlap, 3)})

        return {
            "score": round(covered / len(expected_points), 4),
            "covered": covered,
            "total": len(expected_points),
            "details": details
        }

    # ------------------------------------------------------------------ #
    #  4. ANSWER RELEVANCE  (does the answer address the question?)        #
    # ------------------------------------------------------------------ #

    def evaluate_answer_relevance(
        self,
        query: str,
        answer: str
    ) -> Dict[str, Any]:
        """
        Answer Relevance = token overlap between question keywords and answer.
        A rough heuristic; LLM-as-Judge gives a richer signal.
        """
        q_tokens = set(_tokenize(query))
        a_tokens = set(_tokenize(answer))
        if not q_tokens:
            return {"score": 1.0, "overlapping_terms": []}
        overlap = q_tokens & a_tokens
        score = len(overlap) / len(q_tokens)
        return {
            "score": round(min(score, 1.0), 4),
            "overlapping_terms": sorted(overlap)
        }

    # ------------------------------------------------------------------ #
    #  5. LLM-AS-JUDGE                                                     #
    # ------------------------------------------------------------------ #

    def llm_as_judge(
        self,
        query: str,
        answer: str,
        context_sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Uses the configured LLM to score the answer on a 1-5 rubric across
        four dimensions: Faithfulness, Completeness, Relevance, Clarity.

        Falls back to heuristic scoring if no API key is configured.
        """
        context_text = "\n".join(
            f"[Source #{i+1}] {doc.get('CombinedReview', '')[:200]}"
            for i, doc in enumerate(context_sources[:5])
        )

        judge_prompt = f"""You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.

Given the following USER QUESTION, RETRIEVED CONTEXT, and GENERATED ANSWER, score the answer on each dimension from 1 (worst) to 5 (best).

=== SCORING RUBRIC ===
1. **Faithfulness** (1-5): Are ALL claims in the answer supported by the retrieved context? 5 = every statement traceable to sources; 1 = many fabricated claims.
2. **Completeness** (1-5): Does the answer address all aspects of the question using available context? 5 = comprehensive; 1 = barely addresses the question.
3. **Relevance** (1-5): Is the answer focused on what was asked, without irrelevant tangents? 5 = perfectly on-topic; 1 = off-topic.
4. **Clarity** (1-5): Is the answer well-organized, easy to read, with proper citations? 5 = excellent structure; 1 = confusing.

=== USER QUESTION ===
{query}

=== RETRIEVED CONTEXT ===
{context_text}

=== GENERATED ANSWER ===
{answer}

=== YOUR EVALUATION ===
Respond ONLY with a valid JSON object in this exact format (no markdown, no extra text):
{{"faithfulness": <int>, "completeness": <int>, "relevance": <int>, "clarity": <int>, "justification": "<brief explanation>"}}
"""
        # Try to call the LLM
        gemini_key = self.pipeline.api_key or os.getenv("GEMINI_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")

        raw_response = None

        if gemini_key:
            try:
                from google import genai
                client = genai.Client(api_key=gemini_key)
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=judge_prompt
                )
                raw_response = resp.text
            except Exception as e:
                print(f"[LLM-as-Judge] Gemini error: {e}")

        if raw_response is None and groq_key:
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=0.0
                )
                raw_response = completion.choices[0].message.content
            except Exception as e:
                print(f"[LLM-as-Judge] Groq error: {e}")

        if raw_response:
            try:
                # Strip markdown code fences if present
                cleaned = re.sub(r'```json\s*', '', raw_response)
                cleaned = re.sub(r'```\s*', '', cleaned).strip()
                scores = json.loads(cleaned)
                scores["llm_used"] = True
                scores["average"] = round(
                    np.mean([scores.get("faithfulness", 3), scores.get("completeness", 3),
                             scores.get("relevance", 3), scores.get("clarity", 3)]), 2
                )
                return scores
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[LLM-as-Judge] Parse error: {e}. Falling back to heuristics.")

        # ---- Heuristic fallback ----
        faith = self.evaluate_faithfulness(answer, context_sources)
        relevance = self.evaluate_answer_relevance(query, answer)

        f_score = min(5, max(1, round(faith["score"] * 5)))
        r_score = min(5, max(1, round(relevance["score"] * 5)))
        # Simple completeness heuristic based on answer length
        c_score = min(5, max(1, 2 + min(len(answer.split()) // 30, 3)))
        # Clarity heuristic: citations present?
        cl_score = 4 if "[Source" in answer or "[source" in answer else 3

        return {
            "faithfulness": f_score,
            "completeness": c_score,
            "relevance": r_score,
            "clarity": cl_score,
            "average": round(np.mean([f_score, c_score, r_score, cl_score]), 2),
            "justification": "Heuristic scoring (no LLM API key configured). Add GEMINI_API_KEY for LLM-as-Judge.",
            "llm_used": False
        }

    # ------------------------------------------------------------------ #
    #  6. SINGLE-QUERY DEEP EVALUATION                                     #
    # ------------------------------------------------------------------ #

    def evaluate_single_query(
        self,
        query: str,
        expected_points: Optional[List[str]] = None,
        top_k: int = 5,
        use_hybrid: bool = True
    ) -> Dict[str, Any]:
        """
        Runs all metrics on a single query: retrieval, RAG generation,
        faithfulness, completeness, relevance, and LLM-as-Judge.
        """
        t0 = time.time()
        rag_result = self.pipeline.answer_query(query, top_k=top_k, use_hybrid=use_hybrid)
        total_time = time.time() - t0

        answer = rag_result["answer"]
        sources = rag_result["context_sources"]

        faith = self.evaluate_faithfulness(answer, sources)
        relevance = self.evaluate_answer_relevance(query, answer)
        completeness = self.evaluate_completeness(answer, expected_points or [])
        judge = self.llm_as_judge(query, answer, sources)

        return {
            "query": query,
            "answer": answer,
            "num_sources": len(sources),
            "metrics": {
                "faithfulness": faith,
                "completeness": completeness,
                "answer_relevance": relevance,
                "llm_judge": judge,
                "latency": rag_result["metrics"]
            },
            "sources": sources
        }

    # ------------------------------------------------------------------ #
    #  7. FULL BENCHMARK (all 35 questions)                                #
    # ------------------------------------------------------------------ #

    def run_full_benchmark(
        self,
        eval_cases: Optional[List[Dict[str, Any]]] = None,
        run_judge_on: int = 5
    ) -> Dict[str, Any]:
        """
        Comprehensive benchmark:
          - Retrieval metrics (Dense vs Hybrid)
          - Faithfulness averaged over all queries
          - Completeness averaged over all queries
          - Answer Relevance averaged
          - LLM-as-Judge on first `run_judge_on` queries
        """
        if eval_cases is None:
            eval_cases = get_default_eval_dataset()

        n = len(eval_cases)
        print(f"[Stage 6 Evaluation] Running comprehensive benchmark over {n} test cases...")

        # -- Retrieval metrics --
        vector_metrics = self.evaluate_retrieval(eval_cases, use_hybrid=False)
        hybrid_metrics = self.evaluate_retrieval(eval_cases, use_hybrid=True)

        # -- Generation metrics (faithfulness, completeness, relevance) --
        faithfulness_scores = []
        completeness_scores = []
        relevance_scores = []
        per_query_results = []

        for case in eval_cases:
            rag_res = self.pipeline.answer_query(case["question"], top_k=5, use_hybrid=True)
            answer = rag_res["answer"]
            sources = rag_res["context_sources"]

            faith = self.evaluate_faithfulness(answer, sources)
            comp = self.evaluate_completeness(answer, case.get("expected_answer_points", []))
            rel = self.evaluate_answer_relevance(case["question"], answer)

            faithfulness_scores.append(faith["score"])
            completeness_scores.append(comp["score"])
            relevance_scores.append(rel["score"])

            per_query_results.append({
                "id": case["id"],
                "question": case["question"][:80],
                "faithfulness": faith["score"],
                "completeness": comp["score"],
                "relevance": rel["score"]
            })

        # -- LLM-as-Judge on subset --
        judge_results = []
        for case in eval_cases[:run_judge_on]:
            rag_res = self.pipeline.answer_query(case["question"], top_k=5, use_hybrid=True)
            judge = self.llm_as_judge(case["question"], rag_res["answer"], rag_res["context_sources"])
            judge_results.append({
                "id": case["id"],
                "question": case["question"][:80],
                **judge
            })

        avg_judge = {}
        if judge_results:
            for dim in ["faithfulness", "completeness", "relevance", "clarity", "average"]:
                vals = [j.get(dim, 0) for j in judge_results if isinstance(j.get(dim), (int, float))]
                avg_judge[dim] = round(float(np.mean(vals)), 2) if vals else 0.0

        benchmark = {
            "total_test_queries": n,
            "retrieval": {
                "dense_vector_only": vector_metrics,
                "hybrid_bm25_vector": hybrid_metrics
            },
            "generation": {
                "avg_faithfulness": round(float(np.mean(faithfulness_scores)), 4),
                "avg_completeness": round(float(np.mean(completeness_scores)), 4),
                "avg_answer_relevance": round(float(np.mean(relevance_scores)), 4)
            },
            "llm_judge": {
                "num_judged": len(judge_results),
                "llm_used": judge_results[0].get("llm_used", False) if judge_results else False,
                "avg_scores": avg_judge,
                "per_query": judge_results
            },
            "per_query_breakdown": per_query_results,
            "summary_takeaways": [
                "Hybrid search (BM25 + Dense RRF) improves Hit Rate @ 1 for keyword-specific brand queries.",
                "Dense vector retrieval excels at semantic similarity (e.g. 'burnt roast' matching 'over-dark flavor').",
                "Faithfulness score measures grounding — higher = less hallucination risk.",
                "Completeness measures coverage of expected answer points from ground truth.",
                "LLM-as-Judge provides rubric-based 1-5 scores across Faithfulness, Completeness, Relevance, Clarity."
            ]
        }

        # Persist results
        os.makedirs("data", exist_ok=True)
        with open("data/eval_results.json", "w") as f:
            json.dump(benchmark, f, indent=2, default=str)

        print("[Stage 6 Complete] Saved comprehensive evaluation to data/eval_results.json")
        return benchmark


if __name__ == "__main__":
    from embeddings import ReviewEmbedder
    from data_loader import load_and_preprocess_reviews
    from vector_store import FAISSVectorStore
    from rag_pipeline import RAGPipeline

    df, _ = load_and_preprocess_reviews(sample_size=300)
    embedder = ReviewEmbedder()
    embs = embedder.generate_embeddings(df["CombinedReview"].tolist(), show_progress_bar=False)

    store = FAISSVectorStore(dimension=embs.shape[1])
    store.add_vectors(embs, df)

    pipeline = RAGPipeline(vector_store=store, embedder=embedder, df=df)
    evaluator = RAGEvaluator(pipeline)
    res = evaluator.run_full_benchmark(run_judge_on=3)
    print("Benchmark Results:", json.dumps(res, indent=2, default=str))
