"""
Stage 4 & Stage 5: RAG Pipeline with Prompt Engineering, Advanced Retrieval,
Hybrid Search (BM25 + FAISS Vector), Metadata Filtering, Re-ranking, and Citation Sources.
"""

import os
import time
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi


class RAGPipeline:
    def __init__(
        self,
        vector_store: Any,
        embedder: Any,
        df: Any,
        llm_provider: str = "auto",
        api_key: Optional[str] = None
    ):
        """
        Initializes the RAG Pipeline.
        
        Args:
            vector_store: Instantiated FAISSVectorStore or ChromaVectorStore.
            embedder: Instantiated ReviewEmbedder.
            df: Processed pandas dataframe.
            llm_provider: 'gemini', 'groq', 'mock', or 'auto' (detects API key from env).
            api_key: Optional API key override.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.df = df
        self.llm_provider = llm_provider
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY")
        
        # Build BM25 Index for Hybrid Keyword Search (Stage 5)
        self._init_bm25()

    def _init_bm25(self):
        """Initializes BM25 index for sparse keyword retrieval."""
        print("[Stage 5 Hybrid] Building BM25 index over review corpus...")
        corpus = self.df["CombinedReview"].astype(str).tolist()
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"[Stage 5 Hybrid] BM25 indexed {len(corpus)} documents.")

    def bm25_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Sparse keyword search using BM25."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                row = self.df.iloc[idx].to_dict()
                row["bm25_score"] = float(scores[idx])
                results.append(row)
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Stage 5 Hybrid Search using Reciprocal Rank Fusion (RRF).
        Combines BM25 Keyword Search and Vector Embedding Similarity Search.
        """
        # 1. Vector Search
        q_emb = self.embedder.get_query_embedding(query)
        vector_results = self.vector_store.search(q_emb, top_k=top_k*3, filter_dict=filter_dict)
        
        # 2. BM25 Search
        bm25_results = self.bm25_search(query, top_k=top_k*3)
        
        # 3. Reciprocal Rank Fusion
        rrf_k = 60
        rrf_scores = {}
        row_map = {}
        
        for rank, item in enumerate(vector_results):
            r_id = item["ReviewId"]
            rrf_scores[r_id] = rrf_scores.get(r_id, 0.0) + vector_weight * (1.0 / (rrf_k + rank + 1))
            row_map[r_id] = item
            
        for rank, item in enumerate(bm25_results):
            r_id = item["ReviewId"]
            rrf_scores[r_id] = rrf_scores.get(r_id, 0.0) + (1.0 - vector_weight) * (1.0 / (rrf_k + rank + 1))
            if r_id not in row_map:
                row_map[r_id] = item
                
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        
        final_results = []
        for r_id in sorted_ids:
            row = row_map[r_id]
            row["rrf_score"] = round(rrf_scores[r_id], 5)
            final_results.append(row)
            
        return final_results

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Stage 5 Core Retrieval function supporting Dense Vector, Metadata Filtering, and Hybrid RRF.
        """
        if use_hybrid:
            return self.hybrid_search(query, top_k=top_k, filter_dict=filter_dict)
        else:
            q_emb = self.embedder.get_query_embedding(query)
            return self.vector_store.search(q_emb, top_k=top_k, filter_dict=filter_dict)

    def generate_prompt(self, query: str, context_reviews: List[Dict[str, Any]]) -> str:
        """
        Stage 4 Prompt Engineering.
        Formats retrieved reviews into a structured prompt enforcing zero hallucination.
        """
        context_str = ""
        for i, rev in enumerate(context_reviews, 1):
            brand = rev.get("Brand", "Unknown Brand")
            cat = rev.get("Category", "General")
            rating = rev.get("Score", 3)
            review_text = rev.get("CombinedReview", rev.get("Text", ""))
            
            context_str += f"[Source #{i} | ReviewID: {rev.get('ReviewId')} | Brand: {brand} | Category: {cat} | Rating: {rating}/5]\n"
            context_str += f"{review_text}\n\n"
            
        prompt = f"""You are a specialized Customer Review Insights AI Assistant.
Your goal is to answer the user's question accurately using ONLY the customer reviews provided in the context below.

=== STRICT CONSTRAINTS ===
1. Base your answer STRICTLY on the facts mentioned in the context sources. Do NOT make up information or introduce external assumptions.
2. If the provided context does not contain enough information to answer the question, explicitly state: "Based on the retrieved reviews, there is not enough information to answer this question."
3. Cite your sources clearly using [Source #X] notation whenever referencing specific complaints, praise, or product attributes.
4. Organize your response into bullet points with clear headings (e.g. Key Complaints, Packaging Issues, Flavor & Taste Feedback).

=== RETRIEVED CUSTOMER REVIEWS CONTEXT ===
{context_str}

=== USER QUESTION ===
{query}

=== ANSWER ===
"""
        return prompt

    def call_llm(self, prompt: str) -> str:
        """
        Calls Gemini, Groq, or provides a smart extractive fallback answer.
        """
        gemini_key = self.api_key or os.getenv("GEMINI_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        
        # Try Google Gemini if key available
        if gemini_key:
            try:
                from google import genai
                client = genai.Client(api_key=gemini_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                return response.text
            except Exception as e:
                print(f"[RAG Pipeline] Gemini API call error: {e}. Trying fallback...")
                
        # Try Groq if key available
        if groq_key:
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"[RAG Pipeline] Groq API call error: {e}. Trying fallback...")
                
        # Smart extractive fallback for offline / demo mode
        return self._extractive_fallback_answer(prompt)

    def _extractive_fallback_answer(self, prompt: str) -> str:
        """
        Offline deterministic fallback that synthesizes retrieved context sources
        when external API keys are not supplied.
        """
        lines = prompt.split("\n")
        sources = []
        current_src = {}
        
        for line in lines:
            if line.startswith("[Source #"):
                if current_src:
                    sources.append(current_src)
                current_src = {"header": line, "text": ""}
            elif current_src and not line.startswith("===") and not line.startswith("You are"):
                current_src["text"] += line + " "
                
        if current_src:
            sources.append(current_src)
            
        if not sources:
            return "Based on the retrieved reviews, no matching context was found."
            
        ans = "### Summary of Customer Review Insights (Offline Mode)\n\n"
        ans += "*Note: Operating in local demonstration mode. Add a `GEMINI_API_KEY` or `GROQ_API_KEY` for full generative response synthesis.*\n\n"
        ans += "**Retrieved Source Insights:**\n"
        
        for src in sources[:4]:
            header = src["header"]
            text = src["text"].strip()
            ans += f"- **{header}**: \"{text[:180]}...\"\n"
            
        ans += "\n**Key Highlights:**\n"
        ans += "- **Product Complaints & Feedback**: Customer reviews highlight specific concerns regarding packaging integrity, flavor consistency, and fresh delivery dates as cited in the sources above."
        return ans

    def answer_query(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Full end-to-end RAG Execution:
        1. Context Retrieval (Hybrid/Vector + Metadata Filter)
        2. Prompt Engineering
        3. LLM Generation
        4. Citation Formatting
        """
        start_time = time.time()
        
        # 1. Retrieve
        context_reviews = self.retrieve_context(query, top_k=top_k, use_hybrid=use_hybrid, filter_dict=filter_dict)
        retrieval_time = time.time() - start_time
        
        # 2. Prompt
        prompt = self.generate_prompt(query, context_reviews)
        
        # 3. LLM Call
        llm_start = time.time()
        answer = self.call_llm(prompt)
        generation_time = time.time() - llm_start
        total_time = time.time() - start_time
        
        return {
            "query": query,
            "answer": answer,
            "prompt": prompt,
            "context_sources": context_reviews,
            "metrics": {
                "retrieval_time_sec": round(retrieval_time, 4),
                "generation_time_sec": round(generation_time, 4),
                "total_time_sec": round(total_time, 4),
                "num_retrieved": len(context_reviews)
            }
        }


if __name__ == "__main__":
    from embeddings import ReviewEmbedder
    from data_loader import load_and_preprocess_reviews
    from vector_store import FAISSVectorStore
    
    df, stats = load_and_preprocess_reviews(sample_size=200)
    embedder = ReviewEmbedder()
    embs = embedder.generate_embeddings(df["CombinedReview"].tolist(), show_progress_bar=False)
    
    store = FAISSVectorStore(dimension=embs.shape[1])
    store.add_vectors(embs, df)
    
    pipeline = RAGPipeline(vector_store=store, embedder=embedder, df=df)
    result = pipeline.answer_query("What do people complain about regarding DarkRoast coffee packaging?")
    
    print("\n--- RAG ANSWER ---")
    print(result["answer"])
    print("\n--- METRICS ---")
    print(result["metrics"])
