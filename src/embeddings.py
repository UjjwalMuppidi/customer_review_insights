"""
Stage 2: Embeddings Generation and Caching.

Converts textual review data into dense vector representations using SentenceTransformers.
Applies L2 normalization and provides disk caching for fast vector store re-indexing.
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sentence_transformers import SentenceTransformer


class ReviewEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """
        Initializes the sentence transformer embedding model.
        
        Args:
            model_name: Hugging Face model identifier (default: 'all-MiniLM-L6-v2', 384 dimensions).
            device: 'cpu' or 'cuda' (auto-detected if None).
        """
        self.model_name = model_name
        print(f"[Stage 2] Loading SentenceTransformer model '{model_name}'...")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"[Stage 2] Model loaded. Embedding dimension: {self.embedding_dim}")

    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 128,
        normalize: bool = True,
        show_progress_bar: bool = True
    ) -> np.ndarray:
        """
        Encodes a list of review texts into float32 dense numpy vectors.
        """
        start_time = time.time()
        print(f"[Stage 2] Generating embeddings for {len(texts)} reviews in batches of {batch_size}...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        ).astype(np.float32)
        
        elapsed = time.time() - start_time
        print(f"[Stage 2 Complete] Embeddings shape: {embeddings.shape} | Time taken: {elapsed:.2f}s ({len(texts)/elapsed:.1f} reviews/sec)")
        return embeddings

    def get_query_embedding(self, query: str, normalize: bool = True) -> np.ndarray:
        """Generates embedding vector for a user query."""
        emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=normalize).astype(np.float32)
        return emb[0]


def get_or_create_embeddings(
    df: pd.DataFrame,
    cache_dir: str = "vector_store",
    model_name: str = "all-MiniLM-L6-v2",
    force_recompute: bool = False
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Utility function to retrieve embeddings from cache or compute them fresh.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_emb_path = os.path.join(cache_dir, "embeddings.npy")
    cache_meta_path = os.path.join(cache_dir, "metadata.parquet")
    
    if not force_recompute and os.path.exists(cache_emb_path) and os.path.exists(cache_meta_path):
        cached_df = pd.read_parquet(cache_meta_path)
        if len(cached_df) == len(df) and "ReviewId" in cached_df.columns and (cached_df["ReviewId"].values == df["ReviewId"].values).all():
            print(f"[Stage 2 Cache] Found valid cached embeddings at {cache_emb_path}")
            embeddings = np.load(cache_emb_path)
            return embeddings, cached_df

    embedder = ReviewEmbedder(model_name=model_name)
    texts = df["CombinedReview"].tolist()
    embeddings = embedder.generate_embeddings(texts)
    
    # Save cache
    np.save(cache_emb_path, embeddings)
    df.to_parquet(cache_meta_path, index=False)
    print(f"[Stage 2 Cache] Saved embeddings cache to {cache_emb_path}")
    
    return embeddings, df


if __name__ == "__main__":
    embedder = ReviewEmbedder()
    sample_texts = [
        "The DarkRoast coffee tastes burnt and bitter.",
        "Morning coffee brew is super smooth and acidic.",
        "The dog kibble was eaten eagerly by my lab."
    ]
    vectors = embedder.generate_embeddings(sample_texts)
    print("Vectors shape:", vectors.shape)
    # Demonstrate cosine similarity without LLM
    sim_0_1 = np.dot(vectors[0], vectors[1])
    sim_0_2 = np.dot(vectors[0], vectors[2])
    print(f"Similarity ('burnt coffee' vs 'smooth coffee'): {sim_0_1:.4f}")
    print(f"Similarity ('burnt coffee' vs 'dog kibble'): {sim_0_2:.4f}")
