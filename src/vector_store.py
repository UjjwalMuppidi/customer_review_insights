"""
Stage 3: Vector Store (FAISS and ChromaDB) and Vector Similarity Search.

Stores review vector embeddings, provides fast k-Nearest-Neighbor vector search,
and enables direct similarity exploration without an LLM.
"""

import os
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import faiss
import chromadb


class FAISSVectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        # IndexFlatIP uses Inner Product (Cosine similarity when vectors are L2 normalized)
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata_df: Optional[pd.DataFrame] = None

    def add_vectors(self, embeddings: np.ndarray, metadata_df: pd.DataFrame):
        """Adds normalized float32 vector embeddings and associated pandas dataframe metadata."""
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}")
            
        embeddings_norm = embeddings.copy()
        faiss.normalize_L2(embeddings_norm)
        
        self.index.add(embeddings_norm)
        self.metadata_df = metadata_df.copy().reset_index(drop=True)
        print(f"[Stage 3 FAISS] Added {self.index.ntotal} vectors to index.")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs cosine similarity search using query vector.
        Supports post-filtering by metadata (e.g. Brand, Category, Sentiment, Score).
        """
        if self.index.ntotal == 0 or self.metadata_df is None:
            return []
            
        q_vec = query_vector.copy().reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q_vec)
        
        # Search over extra candidates if filter_dict is supplied
        fetch_k = top_k * 10 if filter_dict else top_k
        fetch_k = min(fetch_k, self.index.ntotal)
        
        distances, indices = self.index.search(q_vec, fetch_k)
        
        results = []
        for sim_score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata_df):
                continue
                
            row = self.metadata_df.iloc[idx].to_dict()
            
            # Apply metadata filters if provided
            if filter_dict:
                match = True
                for key, val in filter_dict.items():
                    if key in row and val is not None and val != "All":
                        if isinstance(val, list):
                            if row[key] not in val:
                                match = False
                                break
                        elif row[key] != val:
                            match = False
                            break
                if not match:
                    continue
                    
            row["similarity_score"] = float(sim_score)
            results.append(row)
            if len(results) >= top_k:
                break
                
        return results

    def save(self, index_dir: str):
        """Saves FAISS index binary and metadata parquet file."""
        os.makedirs(index_dir, exist_ok=True)
        idx_path = os.path.join(index_dir, "faiss.index")
        meta_path = os.path.join(index_dir, "faiss_meta.parquet")
        faiss.write_index(self.index, idx_path)
        if self.metadata_df is not None:
            self.metadata_df.to_parquet(meta_path, index=False)
        print(f"[Stage 3 FAISS] Saved index to {idx_path}")

    def load(self, index_dir: str):
        """Loads FAISS index binary and metadata parquet file."""
        idx_path = os.path.join(index_dir, "faiss.index")
        meta_path = os.path.join(index_dir, "faiss_meta.parquet")
        if os.path.exists(idx_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(idx_path)
            self.metadata_df = pd.read_parquet(meta_path)
            print(f"[Stage 3 FAISS] Loaded {self.index.ntotal} vectors from {idx_path}")
            return True
        return False


class ChromaVectorStore:
    def __init__(self, persist_dir: str = "vector_store/chroma_db", collection_name: str = "amazon_reviews"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_vectors(self, embeddings: np.ndarray, metadata_df: pd.DataFrame):
        """Adds embeddings and metadata into ChromaDB collection."""
        documents = metadata_df["CombinedReview"].tolist()
        ids = [str(r_id) for r_id in metadata_df["ReviewId"].tolist()]
        
        metadatas = []
        for _, row in metadata_df.iterrows():
            meta = {
                "Score": int(row.get("Score", 5)),
                "Sentiment": str(row.get("Sentiment", "Positive")),
                "Category": str(row.get("Category", "General")),
                "Brand": str(row.get("Brand", "Unknown")),
                "HelpfulnessRatio": float(row.get("HelpfulnessRatio", 0.0))
            }
            metadatas.append(meta)
            
        embeddings_list = embeddings.tolist()
        
        # Batch add to avoid payload limits
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings_list[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
        print(f"[Stage 3 ChromaDB] Indexed {self.collection.count()} items in collection '{self.collection_name}'")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Queries ChromaDB collection using embedding vector with metadata filtering."""
        where_clause = None
        if filter_dict:
            conditions = []
            for k, v in filter_dict.items():
                if v is not None and v != "All":
                    conditions.append({k: {"$eq": v}})
            if len(conditions) == 1:
                where_clause = conditions[0]
            elif len(conditions) > 1:
                where_clause = {"$and": conditions}
                
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            where=where_clause
        )
        
        output = []
        if results and "ids" in results and results["ids"]:
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 0.0
                doc_text = results["documents"][0][i] if results["documents"] else ""
                
                sim_score = 1.0 - float(dist) # Cosine distance to similarity conversion
                output.append({
                    "ReviewId": int(doc_id),
                    "CombinedReview": doc_text,
                    "similarity_score": round(sim_score, 4),
                    **meta
                })
        return output


if __name__ == "__main__":
    from embeddings import ReviewEmbedder
    from data_loader import load_and_preprocess_reviews
    
    df, stats = load_and_preprocess_reviews(sample_size=100)
    embedder = ReviewEmbedder()
    embs = embedder.generate_embeddings(df["CombinedReview"].tolist())
    
    store = FAISSVectorStore(dimension=embs.shape[1])
    store.add_vectors(embs, df)
    
    q_emb = embedder.get_query_embedding("What do people dislike about coffee packaging?")
    res = store.search(q_emb, top_k=3)
    print("Raw Vector Search Top 3 Results (No LLM):")
    for r in res:
        print(f"Score: {r['similarity_score']:.4f} | Brand: {r.get('Brand')} | Text: {r['CombinedReview'][:100]}...")
