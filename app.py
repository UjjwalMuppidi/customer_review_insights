"""
Stage 7: Streamlit Web UI for Customer Review Insights Assistant.

Interactive dashboard supporting multi-tab workflows:
1. RAG Insights Chat Assistant (with Citation Cards & Metadata Filters)
2. Vector Search Explorer (Direct Semantic Retrieval without LLM)
3. Data Exploration & Quality Inspector (Deduplication, Cleaning, Metrics)
4. RAG Evaluation Benchmark Dashboard (Hit Rate @ K, MRR, Faithfulness)
5. Stretch Goal: RAG vs Fine-Tuning Model Comparison
"""

import sys
import os
import time
import pandas as pd
import streamlit as st

# Add src to python path
sys.path.append(os.path.abspath("src"))

from data_loader import load_and_preprocess_reviews
from embeddings import get_or_create_embeddings, ReviewEmbedder
from vector_store import FAISSVectorStore, ChromaVectorStore
from rag_pipeline import RAGPipeline
from evaluation import RAGEvaluator
from fine_tune import FineTunedReviewClassifier, get_rag_vs_finetuning_comparison

# Page Configuration
st.set_page_config(
    page_title="Customer Review Insights Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .source-card {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        border-radius: 4px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_dataset_cached(csv_path: str, sample_size: int):
    return load_and_preprocess_reviews(csv_path=csv_path, sample_size=sample_size)


@st.cache_resource(show_spinner=False)
def initialize_system(sample_size: int, vector_backend: str):
    """Initializes data loader, embedder, vector store, and pipeline."""
    df, stats = load_dataset_cached(csv_path="data/Reviews.csv", sample_size=sample_size)
    embeddings, df = get_or_create_embeddings(df, cache_dir="vector_store")
    
    embedder = ReviewEmbedder(model_name="all-MiniLM-L6-v2")
    
    if vector_backend == "FAISS":
        vstore = FAISSVectorStore(dimension=embeddings.shape[1])
        vstore.add_vectors(embeddings, df)
    else:
        vstore = ChromaVectorStore(persist_dir="vector_store/chroma_db")
        vstore.add_vectors(embeddings, df)
        
    pipeline = RAGPipeline(vector_store=vstore, embedder=embedder, df=df)
    return df, stats, embedder, vstore, pipeline


# --- SIDEBAR CONTROLS ---
st.sidebar.image("https://img.icons8.com/color/96/amazon.png", width=64)
st.sidebar.title("Configuration")

sample_size = st.sidebar.slider("Dataset Sample Size", min_value=500, max_value=25000, value=3000, step=500)
vector_backend = st.sidebar.selectbox("Vector Store Backend", ["FAISS", "ChromaDB"])

st.sidebar.subheader("LLM Options")
api_key_input = st.sidebar.text_input("Gemini API Key (Optional)", type="password", help="Leave empty to use local fallback mode.")
if api_key_input:
    os.environ["GEMINI_API_KEY"] = api_key_input

st.sidebar.subheader("Stage Quick Jump")
st.sidebar.info("💡 Pipeline stages: Data Prep → Embeddings → Vector Index → RAG → Hybrid Search → Evaluation → Fine-Tune")

# Initialize Pipeline
with st.spinner("Loading vector store and models..."):
    df, stats, embedder, vstore, pipeline = initialize_system(sample_size, vector_backend)
    if api_key_input:
        pipeline.api_key = api_key_input

# --- MAIN APP LAYOUT ---
st.markdown('<div class="main-title">🛍️ Customer Review Insights Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by RAG, Hybrid Retrieval (BM25 + FAISS), and Evaluation</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 RAG Chat Assistant",
    "🔍 Vector Search Explorer",
    "📊 Data Quality & Preprocessing",
    "📈 Evaluation Benchmark",
    "🎯 Stretch Goal: RAG vs Fine-Tuning"
])


# --- TAB 1: RAG CHAT ASSISTANT ---
with tab1:
    st.subheader("Ask Questions About Customer Reviews")
    st.write("Examples: *'What do people dislike about DarkRoast coffee?'*, *'Summarize complaints about packaging'*, *'Are tea bags fresh?'*")
    
    col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)
    with col_filter1:
        cat_options = ["All"] + sorted(list(df["Category"].dropna().unique()))
        selected_cat = st.selectbox("Category Filter", cat_options)
    with col_filter2:
        brand_options = ["All"] + sorted(list(df["Brand"].dropna().unique()))
        selected_brand = st.selectbox("Brand Filter", brand_options)
    with col_filter3:
        selected_rating = st.selectbox("Min Star Rating", ["All", "1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"])
    with col_filter4:
        use_hybrid = st.checkbox("Enable Hybrid BM25 Search", value=True)

    # Build filter dict
    filter_dict = {}
    if selected_cat != "All":
        filter_dict["Category"] = selected_cat
    if selected_brand != "All":
        filter_dict["Brand"] = selected_brand
    if selected_rating != "All":
        filter_dict["Score"] = int(selected_rating[0])
        
    user_query = st.text_input("Enter your question:", value="What do people complain about regarding coffee packaging?")
    
    if st.button("Generate Insights", type="primary"):
        with st.spinner("Retrieving reviews & generating answer..."):
            res = pipeline.answer_query(
                query=user_query,
                top_k=5,
                use_hybrid=use_hybrid,
                filter_dict=filter_dict if filter_dict else None
            )
            
            st.markdown("### 🤖 Answer")
            st.markdown(res["answer"])
            
            st.markdown("### ⏱️ Performance Metrics")
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Retrieval Latency", f"{res['metrics']['retrieval_time_sec']}s")
            m_col2.metric("Generation Latency", f"{res['metrics']['generation_time_sec']}s")
            m_col3.metric("Sources Retrieved", f"{res['metrics']['num_retrieved']}")
            
            st.markdown("### 📚 Source Citations & Context")
            for idx, src in enumerate(res["context_sources"], 1):
                score_val = src.get("similarity_score", src.get("rrf_score", "N/A"))
                with st.expander(f"Source #{idx} | {src.get('Brand')} ({src.get('Category')}) - Rating: {src.get('Score')}/5 | Score: {score_val}"):
                    st.write(f"**Review Text**: {src.get('CombinedReview')}")
                    st.caption(f"Review ID: {src.get('ReviewId')} | Product ID: {src.get('ProductId')}")


# --- TAB 2: VECTOR SEARCH EXPLORER (NO LLM) ---
with tab2:
    st.subheader("Raw Semantic Vector Search (Stage 3)")
    st.write("Explore similarity search results directly from FAISS/Chroma without sending data to an LLM.")
    
    query_vec = st.text_input("Search query for raw vector retrieval:", value="burnt coffee taste and char")
    top_k_vec = st.slider("Top K Results", 1, 20, 5)
    
    if st.button("Run Raw Vector Search"):
        t0 = time.time()
        q_emb = embedder.get_query_embedding(query_vec)
        raw_results = vstore.search(q_emb, top_k=top_k_vec)
        search_time = time.time() - t0
        
        st.success(f"Retrieved {len(raw_results)} results in {search_time*1000:.2f} ms")
        
        for r in raw_results:
            st.markdown(f"""
            <div class="source-card">
                <b>Brand:</b> {r.get('Brand')} | <b>Category:</b> {r.get('Category')} | <b>Rating:</b> {r.get('Score')}/5 | <b>Similarity Score:</b> {r.get('similarity_score', 0):.4f}<br/>
                <b>Text:</b> {r.get('CombinedReview')}
            </div>
            """, unsafe_allow_html=True)


# --- TAB 3: DATA QUALITY & PREPROCESSING ---
with tab3:
    st.subheader("Stage 1 Data Cleaning & Exploration")
    st.write("Data quality directly determines RAG quality. Review preprocessing statistics below:")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw Inputs", stats["initial_raw_count"])
    c2.metric("Duplicates Removed", int(stats["duplicates_removed"]))
    c3.metric("Short Reviews Filtered", int(stats["short_reviews_removed"]))
    c4.metric("Final Processed", stats["final_processed_count"])
    
    st.markdown("### Sample Cleaned Dataset Preview")
    st.dataframe(df[["ReviewId", "Brand", "Category", "Score", "Sentiment", "CombinedReview"]].head(20), use_container_width=True)


# --- TAB 4: EVALUATION BENCHMARK DASHBOARD ---
with tab4:
    st.subheader("Stage 6 RAG Evaluation Suite")
    st.write("Quantitative assessment measuring Hit Rate @ K, MRR, and Faithfulness across a 35-question test suite.")
    
    if st.button("Run Full 35-Question Benchmark"):
        with st.spinner("Running retrieval evaluation suite..."):
            evaluator = RAGEvaluator(pipeline)
            eval_res = evaluator.run_full_benchmark()
            
            st.markdown("### 📊 Retrieval Hit Rate Comparison")
            
            h_df = pd.DataFrame([
                {"Metric": "Hit Rate @ 1", "Dense Vector Only": eval_res["dense_vector_only"]["hit_rates"]["Hit_Any@1"], "Hybrid (BM25+Vector)": eval_res["hybrid_bm25_vector"]["hit_rates"]["Hit_Any@1"]},
                {"Metric": "Hit Rate @ 3", "Dense Vector Only": eval_res["dense_vector_only"]["hit_rates"]["Hit_Any@3"], "Hybrid (BM25+Vector)": eval_res["hybrid_bm25_vector"]["hit_rates"]["Hit_Any@3"]},
                {"Metric": "Hit Rate @ 5", "Dense Vector Only": eval_res["dense_vector_only"]["hit_rates"]["Hit_Any@5"], "Hybrid (BM25+Vector)": eval_res["hybrid_bm25_vector"]["hit_rates"]["Hit_Any@5"]},
                {"Metric": "Mean Reciprocal Rank (MRR)", "Dense Vector Only": eval_res["dense_vector_only"]["mrr"], "Hybrid (BM25+Vector)": eval_res["hybrid_bm25_vector"]["mrr"]}
            ])
            st.table(h_df)
            
            st.markdown("### 🔒 Faithfulness & Groundedness")
            st.write(f"**Sample Groundedness Score**: {eval_res['sample_faithfulness_score']*100:.1f}% grounded in context sources.")
            
            st.markdown("### 💡 Key Takeaways")
            for t in eval_res["summary_takeaways"]:
                st.write(f"- {t}")


# --- TAB 5: STRETCH GOAL - RAG VS FINE-TUNING ---
with tab5:
    st.subheader("Stretch Goal: RAG vs Fine-Tuning Trade-off Analysis")
    st.write("Understand when to use RAG vs Fine-Tuning for Customer Review Insights.")
    
    st.markdown("### 1. Interactive Fine-Tuned Classifier Demo")
    if st.button("Train Classifier on Current Dataset"):
        clf = FineTunedReviewClassifier()
        clf_stats = clf.train(df)
        st.success(f"Classifier trained in {clf_stats['training_time_sec']}s with Accuracy: {clf_stats['accuracy']*100:.2f}%")
        
        sample_input = st.text_input("Test review for fine-tuned sentiment classifier:", value="The packaging was completely crushed and the coffee pods leaked grounds!")
        if sample_input:
            pred = clf.predict(sample_input)
            st.write(f"**Predicted Sentiment**: `{pred['predicted_sentiment']}`")
            st.write(f"**Inference Latency**: `{pred['inference_time_ms']} ms` (vs ~1000 ms for RAG)")
            st.write(f"**Extracted Complaint Themes**: `{', '.join(pred['detected_complaint_themes'])}`")
            
    st.markdown("---")
    st.markdown("### 2. RAG vs Fine-Tuning Architectural Comparison")
    comp_data = get_rag_vs_finetuning_comparison()
    comp_df = pd.DataFrame(comp_data["comparison_matrix"])
    st.table(comp_df)
    
    st.markdown("### 🎯 Decision Framework")
    for item in comp_data["recommendation_framework"]:
        st.write(f"- {item}")
