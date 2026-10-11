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
from dotenv import load_dotenv
load_dotenv()
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
from analytics import ReviewAnalytics

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 RAG Chat Assistant",
    "🔍 Vector Search Explorer",
    "📊 Data Quality & Preprocessing",
    "📈 Evaluation Benchmark",
    "🎯 Stretch Goal: RAG vs Fine-Tuning",
    "🧠 Semantic Topic Clustering"
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
    st.subheader("Stage 6: Comprehensive RAG Evaluation Suite")
    st.write("Quantitative assessment across **6 metric dimensions**: Retrieval, Faithfulness, Completeness, Answer Relevance, Context Precision, and LLM-as-Judge.")

    eval_col1, eval_col2 = st.columns(2)
    with eval_col1:
        judge_count = st.slider("LLM-as-Judge queries (0 = skip)", 0, 15, 5)
    with eval_col2:
        st.info("💡 Set `GEMINI_API_KEY` in sidebar for real LLM-as-Judge scoring. Without it, heuristic fallback is used.")

    if st.button("🚀 Run Full 35-Question Benchmark", type="primary"):
        with st.spinner("Running comprehensive evaluation suite (retrieval + generation + judge)..."):
            evaluator = RAGEvaluator(pipeline)
            eval_res = evaluator.run_full_benchmark(run_judge_on=judge_count)

            # ---- Section 1: Retrieval Metrics ----
            st.markdown("---")
            st.markdown("### 📡 1. Retrieval Metrics")
            dense = eval_res["retrieval"]["dense_vector_only"]
            hybrid = eval_res["retrieval"]["hybrid_bm25_vector"]

            ret_df = pd.DataFrame([
                {"Metric": "Hit Rate @ 1", "Dense Vector": dense["hit_rates"]["Hit_Any@1"], "Hybrid (BM25+Vector)": hybrid["hit_rates"]["Hit_Any@1"]},
                {"Metric": "Hit Rate @ 3", "Dense Vector": dense["hit_rates"]["Hit_Any@3"], "Hybrid (BM25+Vector)": hybrid["hit_rates"]["Hit_Any@3"]},
                {"Metric": "Hit Rate @ 5", "Dense Vector": dense["hit_rates"]["Hit_Any@5"], "Hybrid (BM25+Vector)": hybrid["hit_rates"]["Hit_Any@5"]},
                {"Metric": "MRR", "Dense Vector": dense["mrr"], "Hybrid (BM25+Vector)": hybrid["mrr"]},
                {"Metric": "Context Precision @ 5", "Dense Vector": dense["context_precision"]["Ctx_Precision@5"], "Hybrid (BM25+Vector)": hybrid["context_precision"]["Ctx_Precision@5"]},
                {"Metric": "Avg Latency (sec)", "Dense Vector": dense["avg_latency_sec"], "Hybrid (BM25+Vector)": hybrid["avg_latency_sec"]},
            ])
            st.table(ret_df)

            # ---- Section 2: Generation Quality Metrics ----
            st.markdown("---")
            st.markdown("### 🔬 2. Generation Quality Metrics (Averaged over 35 queries)")
            gen = eval_res["generation"]

            g1, g2, g3 = st.columns(3)
            g1.metric("🔒 Faithfulness", f"{gen['avg_faithfulness']*100:.1f}%",
                       help="Fraction of answer content words grounded in retrieved context. Higher = less hallucination.")
            g2.metric("✅ Completeness", f"{gen['avg_completeness']*100:.1f}%",
                       help="Fraction of expected answer points covered by the generated answer.")
            g3.metric("🎯 Answer Relevance", f"{gen['avg_answer_relevance']*100:.1f}%",
                       help="Token overlap between question keywords and answer content.")

            # ---- Section 3: LLM-as-Judge ----
            st.markdown("---")
            judge_data = eval_res["llm_judge"]
            llm_tag = "✅ LLM Scoring" if judge_data["llm_used"] else "⚠️ Heuristic Fallback"
            st.markdown(f"### 🧑‍⚖️ 3. LLM-as-Judge Rubric Scores ({llm_tag})")

            if judge_data["num_judged"] > 0:
                avg = judge_data["avg_scores"]
                j1, j2, j3, j4, j5 = st.columns(5)
                j1.metric("Faithfulness", f"{avg.get('faithfulness', 0)}/5")
                j2.metric("Completeness", f"{avg.get('completeness', 0)}/5")
                j3.metric("Relevance", f"{avg.get('relevance', 0)}/5")
                j4.metric("Clarity", f"{avg.get('clarity', 0)}/5")
                j5.metric("🏆 Overall", f"{avg.get('average', 0)}/5")

                with st.expander("Per-Query Judge Breakdown"):
                    judge_display = []
                    for j in judge_data["per_query"]:
                        judge_display.append({
                            "ID": j.get("id"),
                            "Question": j.get("question", "")[:60],
                            "Faith.": j.get("faithfulness"),
                            "Compl.": j.get("completeness"),
                            "Relev.": j.get("relevance"),
                            "Clarity": j.get("clarity"),
                            "Avg": j.get("average"),
                        })
                    st.dataframe(pd.DataFrame(judge_display), use_container_width=True)
                    # Show justification for first entry
                    if judge_data["per_query"]:
                        st.caption(f"**Sample justification**: {judge_data['per_query'][0].get('justification', 'N/A')}")
            else:
                st.write("LLM-as-Judge was skipped (set slider > 0 to enable).")

            # ---- Section 4: Per-Query Breakdown Table ----
            st.markdown("---")
            st.markdown("### 📋 4. Per-Query Metric Breakdown")
            pq_df = pd.DataFrame(eval_res["per_query_breakdown"])
            pq_df.columns = ["ID", "Question", "Faithfulness", "Completeness", "Relevance"]
            st.dataframe(pq_df, use_container_width=True, height=400)

            # ---- Section 5: Takeaways ----
            st.markdown("---")
            st.markdown("### 💡 5. Key Takeaways")
            for t in eval_res["summary_takeaways"]:
                st.write(f"- {t}")

    # ---- Single Query Deep Eval ----
    st.markdown("---")
    st.markdown("### 🔍 Deep-Dive: Evaluate a Single Query")
    single_q = st.text_input("Enter a query to evaluate in depth:", value="What do people complain about regarding coffee packaging?", key="eval_single_q")
    if st.button("Run Single-Query Evaluation"):
        with st.spinner("Evaluating..."):
            evaluator = RAGEvaluator(pipeline)
            sq_res = evaluator.evaluate_single_query(single_q)

            st.markdown(f"**Answer:** {sq_res['answer'][:500]}...")

            sq_m = sq_res["metrics"]
            s1, s2, s3 = st.columns(3)
            s1.metric("Faithfulness", f"{sq_m['faithfulness']['score']*100:.1f}%")
            s2.metric("Answer Relevance", f"{sq_m['answer_relevance']['score']*100:.1f}%")
            s3.metric("Sources Retrieved", sq_res["num_sources"])

            if sq_m["faithfulness"]["unsupported_tokens"]:
                st.warning(f"Unsupported tokens in answer: `{', '.join(sq_m['faithfulness']['unsupported_tokens'][:10])}`")

            judge = sq_m["llm_judge"]
            st.markdown("**LLM-as-Judge Scores:**")
            jc1, jc2, jc3, jc4 = st.columns(4)
            jc1.metric("Faith.", f"{judge.get('faithfulness', 'N/A')}/5")
            jc2.metric("Compl.", f"{judge.get('completeness', 'N/A')}/5")
            jc3.metric("Relev.", f"{judge.get('relevance', 'N/A')}/5")
            jc4.metric("Clarity", f"{judge.get('clarity', 'N/A')}/5")
            st.caption(f"Justification: {judge.get('justification', 'N/A')}")


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


# --- TAB 6: SEMANTIC TOPIC CLUSTERING ---
with tab6:
    st.subheader("Semantic Topic & Complaint Theme Clustering")
    st.write("Automatically group customer reviews into K-Means vector clusters to discover common complaint themes.")
    
    num_k = st.slider("Select Number of Clusters", 2, 6, 4)
    if st.button("Run Vector Topic Clustering"):
        with st.spinner("Clustering review embeddings..."):
            analytics = ReviewAnalytics(embedder=embedder)
            cluster_res = analytics.cluster_reviews(df, num_clusters=num_k)
            
            st.markdown(f"**Analyzed {cluster_res['num_reviews_analyzed']} reviews across {cluster_res['num_clusters']} clusters:**")
            
            for c in cluster_res["clusters"]:
                with st.expander(f"Cluster #{c['cluster_id']} | Count: {c['count']} reviews | Avg Rating: {c['avg_rating']}/5"):
                    st.write(f"**Dominant Sentiment**: `{c['dominant_sentiment']}`")
                    st.write(f"**Top Key Terms**: `{', '.join(c['top_keywords'])}`")
                    st.write("**Sample Reviews in Cluster:**")
                    for text in c["sample_reviews"]:
                        st.write(f"- \"{text[:150]}...\"")
