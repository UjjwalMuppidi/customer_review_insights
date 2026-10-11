"""
Analytics Module: Review Clustering & Complaint Keyword Analysis.

Uses K-Means vector clustering over embeddings to automatically group customer feedback
into distinct complaint topics and thematic clusters.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer


class ReviewAnalytics:
    def __init__(self, embedder: Any):
        self.embedder = embedder

    def cluster_reviews(self, df: pd.DataFrame, num_clusters: int = 4) -> Dict[str, Any]:
        """
        Clusters review texts into semantic topic groups using K-Means on sentence vectors.
        Extracts top keywords for each cluster.
        """
        if len(df) < num_clusters:
            return {"error": "Not enough reviews to cluster."}
            
        texts = df["CombinedReview"].tolist()
        vectors = self.embedder.generate_embeddings(texts, show_progress_bar=False)
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)
        
        df_clustered = df.copy()
        df_clustered["Cluster"] = labels
        
        clusters_summary = []
        for c_id in range(num_clusters):
            cluster_df = df_clustered[df_clustered["Cluster"] == c_id]
            cluster_texts = cluster_df["CombinedReview"].tolist()
            
            # Extract top key terms per cluster using CountVectorizer
            vec = CountVectorizer(stop_words="english", max_features=5, ngram_range=(1, 2))
            try:
                dtm = vec.fit_transform(cluster_texts)
                top_terms = list(vec.get_feature_names_out())
            except Exception:
                top_terms = ["general", "feedback"]
                
            clusters_summary.append({
                "cluster_id": c_id + 1,
                "count": len(cluster_df),
                "avg_rating": round(float(cluster_df["Score"].mean()), 2),
                "dominant_sentiment": str(cluster_df["Sentiment"].mode().iloc[0]) if not cluster_df.empty else "Neutral",
                "top_keywords": top_terms,
                "sample_reviews": cluster_texts[:2]
            })
            
        return {
            "num_reviews_analyzed": len(df),
            "num_clusters": num_clusters,
            "clusters": clusters_summary
        }
