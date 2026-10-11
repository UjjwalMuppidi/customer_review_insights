"""
CLI Pipeline Runner for Customer Review Insights Assistant.

Runs Stages 1 through 6 sequentially, outputs logs, and validates all components.
"""

import sys
import os
import json
import time

sys.path.append(os.path.abspath("src"))

from data_loader import load_and_preprocess_reviews
from embeddings import get_or_create_embeddings, ReviewEmbedder
from vector_store import FAISSVectorStore
from rag_pipeline import RAGPipeline
from evaluation import RAGEvaluator
from fine_tune import FineTunedReviewClassifier, get_rag_vs_finetuning_comparison


def main():
    print("==================================================================")
    print("     CUSTOMER REVIEW INSIGHTS ASSISTANT - FULL PIPELINE RUNNER     ")
    print("==================================================================")
    
    # Stage 1: Data Exploration, Cleaning, Deduplication
    print("\n[STAGE 1] Data Loading & Preprocessing...")
    df, stats = load_and_preprocess_reviews(sample_size=3000)
    print("Stage 1 Stats Summary:", stats)
    
    # Stage 2: Embeddings Generation
    print("\n[STAGE 2] Embedding Generation & Caching...")
    embeddings, df = get_or_create_embeddings(df, cache_dir="vector_store")
    print(f"Embeddings Matrix Shape: {embeddings.shape}")
    
    # Stage 3: Vector Store Indexing (FAISS)
    print("\n[STAGE 3] Vector Store Indexing (FAISS)...")
    embedder = ReviewEmbedder(model_name="all-MiniLM-L6-v2")
    vstore = FAISSVectorStore(dimension=embeddings.shape[1])
    vstore.add_vectors(embeddings, df)
    vstore.save("vector_store")
    
    # Test Raw Similarity Search (No LLM)
    q_emb = embedder.get_query_embedding("What do customers complain about coffee packaging?")
    raw_res = vstore.search(q_emb, top_k=2)
    print(f"Raw Vector Search (No LLM) Top Match Score: {raw_res[0]['similarity_score']:.4f}")
    
    # Stage 4 & 5: RAG Pipeline & Advanced Hybrid Retrieval
    print("\n[STAGE 4 & 5] RAG Pipeline & Hybrid BM25 Search...")
    pipeline = RAGPipeline(vector_store=vstore, embedder=embedder, df=df)
    sample_query = "What do people dislike about DarkRoast coffee packaging?"
    rag_result = pipeline.answer_query(sample_query, top_k=5, use_hybrid=True)
    
    print("\n--- SAMPLE RAG QUERY ---")
    print("Query:", sample_query)
    print("\nAnswer:\n", rag_result["answer"])
    print("\nMetrics:", rag_result["metrics"])
    
    # Stage 6: Comprehensive RAG Evaluation Benchmark
    print("\n[STAGE 6] Comprehensive Evaluation Benchmark (35 Test Cases)...")
    evaluator = RAGEvaluator(pipeline)
    eval_res = evaluator.run_full_benchmark(run_judge_on=3)
    print("Evaluation Results Summary:", json.dumps({
        "retrieval": eval_res["retrieval"]["hybrid_bm25_vector"],
        "generation": eval_res["generation"],
        "llm_judge_avg": eval_res["llm_judge"]["avg_scores"]
    }, indent=2, default=str))
    
    # Stretch Goal: Fine-Tuned Model Comparison
    print("\n[STRETCH GOAL] Fine-Tuning & RAG Comparison Matrix...")
    clf = FineTunedReviewClassifier()
    clf_stats = clf.train(df)
    test_pred = clf.predict("The plastic seal was broken and coffee leaked out!")
    print("Fine-tuned Prediction Test:", test_pred)
    
    print("\n==================================================================")
    print("     ALL STAGES EXECUTED SUCCESSFULLY! launch web UI with:        ")
    print("     streamlit run app.py                                         ")
    print("==================================================================")


if __name__ == "__main__":
    main()
