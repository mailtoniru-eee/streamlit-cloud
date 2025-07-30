import streamlit as st
import altair as alt
from supabase import create_client, Client
import pandas as pd
from PIL import Image
import json

if st.button("🔄 Refresh Data"):
    st.cache_data.clear()  # Clears only the data cache
    st.rerun()             # Forces the app to reload fresh data

# Load logo image
image = Image.open("logo/iith.jpg")

# Read credentials from environment variables
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase credentials are not set. Please set SUPABASE_URL and SUPABASE_KEY as environment variables.")
    st.stop()

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase: Client = init_supabase()

@st.cache_data(ttl=300)  # Refresh cache every 300 seconds
def get_data():
    response = supabase.table("rag_metrics").select("*").range(0, 9999).execute()
    return pd.DataFrame(response.data)

df = get_data()

# Normalize nested JSON column
input_vars_df = pd.json_normalize(df["input_variable"])

def extract_first(dataset):
    if isinstance(dataset, list) and len(dataset) > 0:
        return dataset[0]
    return dataset

input_vars_df["input_dataset"] = input_vars_df["input_dataset"].apply(extract_first)

df = df.drop(columns=["input_variable"]).reset_index(drop=True)
input_vars_df = input_vars_df.reset_index(drop=True)
df = pd.concat([df, input_vars_df], axis=1)

# --------------------- UI Header ------------------------
st.subheader("Group 23 - RAG Application - RAGBench Dataset")

st.markdown(
    """
    <style>
        /* Make sidebar thinner */
        [data-testid="stSidebar"] {
            min-width: 200px;
            width: 200px;
        }

        /* Expand main content area */
        .main {
            margin-left: 220px; /* Slight offset for sidebar padding */
        }

        /* Optional: reduce padding in the main container */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

#col1, col2, col3 = st.columns([1, 2, 1])
#with col2:
#    st.image(image, width=300)

# ---------------- Sidebar: Shared Filters ----------------

# Sidebar: Selection Level (Dataset vs Domain)
selection_level = st.sidebar.radio("View Level", ["Dataset", "Domain"], horizontal=True)

if selection_level == "Dataset":
    selected_value = st.sidebar.selectbox("Dataset", sorted(df["input_dataset"].dropna().unique()), key="dataset_sidebar")
else:
    dataset_domain_map = {
    "pubmedqa": "Bio-medical Research",
    "covidqa": "Bio-medical Research",
    "hotpotqa": "General Knowledge",
    "msmarco": "General Knowledge",
    "hagrid": "General Knowledge",
    "expertqa": "General Knowledge",
    "delucionqa": "Customer Support",
    "emanual": "Customer Support",
    "techqa": "Customer Support",
    "finqa": "Finance",
    "tatqa": "Finance"}
    df["domain"] = df["input_dataset"].map(dataset_domain_map)
    selected_value = st.sidebar.selectbox("Domain", sorted(df["domain"].dropna().unique()), key="domain_sidebar")

st.sidebar.write(f"🔎 Number of rows fetched: {len(df)}")

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["Dashboard - Individual", "Dashboard - Comparison", "Best Config"])

# ---------------- TAB 1 ----------------
with tab1:
    if selection_level == "Dataset":
        df1 = df[df["input_dataset"] == selected_value]
        selected_dataset = selected_value
    else:
        # Domain-level: Add one more selectbox in second-level slicers
        available_datasets = sorted(df[df["domain"] == selected_value]["input_dataset"].dropna().unique())
        selected_dataset = st.selectbox("Select Dataset in Domain", available_datasets, key="domain_dataset_inside_tab1")
        df1 = df[df["input_dataset"] == selected_dataset]
        
    # Create two columns
    col1, col2 = st.columns(2)

    # First column slicers
    with col1:
        # vector_db = st.selectbox("Vector DB", sorted(df1["vector_db"].dropna().unique()))
        vector_db_options = sorted(df1["vector_db"].dropna().unique())
        vector_db = st.selectbox("Vector DB", vector_db_options, index=0 if vector_db_options else None)
        chunking_type_options = sorted(df1["chunking_type"].dropna().unique())
        # chunking_type = st.selectbox("Chunking Type", sorted(df1["chunking_type"].dropna().unique()))
        chunking_type = st.selectbox("Chunking Type", chunking_type_options, index=0 if chunking_type_options else None)
        repacking_strategy_options = sorted(df1["repacking_strategy"].dropna().unique())
        repacking_strategy = st.selectbox("Repacking Strategy", repacking_strategy_options, index=0 if repacking_strategy_options else None)
        # repacking_strategy = st.selectbox("Repacking Strategy", sorted(df1["repacking_strategy"].dropna().unique()))
        # generator_model = st.selectbox("Generator Model", sorted(df1["generator_model"].dropna().unique()))
        generator_model_options = sorted(df1["generator_model"].dropna().unique())
        generator_model = st.selectbox("Generator Model", generator_model_options, index=0 if generator_model_options else None)
        
    with col2:
        embedding_model_options = sorted(df1["embedding_model"].dropna().unique())
        embedding_model = st.selectbox("Embedding Model", embedding_model_options, index=0 if embedding_model_options else None)
        # embedding_model = st.selectbox("Embedding Model", sorted(df1["embedding_model"].dropna().unique()))
        # reranking_model = st.selectbox("Reranking Model", sorted(df1["reranking_model"].dropna().unique()))
        reranking_model_options = sorted(df1["reranking_model"].dropna().unique())
        reranking_model = st.selectbox("Reranking Model", reranking_model_options, index=0 if reranking_model_options else None)
        # summarization_model = st.selectbox("Summarization Model", sorted(df1["summarization_model"].dropna().unique()))
        summarization_model_options = sorted(df1["summarization_model"].dropna().unique())
        summarization_model = st.selectbox("Summarization Model", summarization_model_options, index=0 if summarization_model_options else None)
        # template = st.selectbox("Template Used", sorted(df1["template"].dropna().unique()))
        template_options = sorted(df1["template"].dropna().unique())
        template = st.selectbox("Template Used", template_options, index=0 if template_options else None)

    # Apply filters
    filtered_df = df1[
        (df1["vector_db"] == vector_db) &
        (df1["embedding_model"] == embedding_model) &
        (df1["chunking_type"] == chunking_type) &
        (df1["reranking_model"] == reranking_model) &
        (df1["repacking_strategy"] == repacking_strategy) &
        (df1["summarization_model"] == summarization_model) &
        (df1["generator_model"] == generator_model) &
        (df1["template"] == template)
    ]

    metrics = [
        "context_relevance", "context_utilization", "adherence", "completeness",
        "hallucination_auroc", "relevance_rmse", "utilization_rmse"
    ]

    metric_averages = {metric: filtered_df[metric].mean() for metric in metrics}

    avg_df = pd.DataFrame({
        "Metric": list(metric_averages.keys()),
        "Average": list(metric_averages.values())
    })

    st.subheader("Metric Averages for Selected Configuration")

    chart = (
        alt.Chart(avg_df)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Average:Q"),
            color=alt.Color("Metric:N", legend=None),
            tooltip=["Metric", "Average"]
        )
        .properties(height=400, width=800)
    )

    st.altair_chart(chart, use_container_width=False)

# ---------------- TAB 2 ----------------
with tab2:
    
    if selection_level == "Dataset":
        df2 = df[df["input_dataset"] == selected_value]
    else:
        df2 = df[df["domain"] == selected_value]

    st.header("Metric Comparison Across Configurations")
    
    col1, col2 = st.columns(2)
    
    # First column slicers
    with col1:
        vector_dbs = st.multiselect("Vector DB", df2["vector_db"].dropna().unique())
        chunking_types = st.multiselect("Chunking Type", df2["chunking_type"].dropna().unique())
        repacking_strategies = st.multiselect("Repacking Strategy", df2["repacking_strategy"].dropna().unique())
        generator_models = st.multiselect("Generator Model", df2["generator_model"].dropna().unique())
    with col2:
        embedding_models = st.multiselect("Embedding Model", df2["embedding_model"].dropna().unique())
        reranking_models = st.multiselect("Reranking Model", df2["reranking_model"].dropna().unique())
        summarization_models = st.multiselect("Summarization Model", df2["summarization_model"].dropna().unique())
        templates = st.multiselect("Templates Used", df2["template"].dropna().unique())
    # Apply filters
    filtered_df = df2.copy()

    if vector_dbs:
        filtered_df = filtered_df[filtered_df["vector_db"].isin(vector_dbs)]
    if embedding_models:
        filtered_df = filtered_df[filtered_df["embedding_model"].isin(embedding_models)]
    if chunking_types:
        filtered_df = filtered_df[filtered_df["chunking_type"].isin(chunking_types)]
    if reranking_models:
        filtered_df = filtered_df[filtered_df["reranking_model"].isin(reranking_models)]
    if repacking_strategies:
        filtered_df = filtered_df[filtered_df["repacking_strategy"].isin(repacking_strategies)]
    if summarization_models:
        filtered_df = filtered_df[filtered_df["summarization_model"].isin(summarization_models)]
    if generator_models:
        filtered_df = filtered_df[filtered_df["generator_model"].isin(generator_models)]
    if templates:
        filtered_df = filtered_df[filtered_df["template"].isin(templates)]        

    group_fields = [
        "input_dataset", "vector_db", "embedding_model", "chunking_type", "reranking_model",
        "repacking_strategy", "summarization_model", "generator_model", "template"
    ]

    metrics = [
        "context_relevance", "context_utilization", "adherence", "completeness",
        "hallucination_auroc", "relevance_rmse", "utilization_rmse"
    ]

    grouped_df = (
        filtered_df.groupby(group_fields)[metrics]
        .mean()
        .reset_index()
    )

    grouped_df["config_label"] = grouped_df[group_fields].agg(" | ".join, axis=1)

    chart_df = grouped_df.melt(
        id_vars=["config_label"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Average"
    )

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", title="Metric", sort=metrics, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Average:Q", title="Average Score"),
            color=alt.Color("config_label:N", title="Configuration",
                            legend=alt.Legend(orient="bottom", columns=2, labelFontSize=12, titleFontSize=14)),
            xOffset="config_label:N",
            tooltip=["config_label", "Metric", "Average"]
        )
        .properties(
            width=1000,
            height=500,
            title="Grouped Bar Chart: Metric Comparison by Configuration"
        )
    )

    st.altair_chart(chart, use_container_width=False)


with tab3:
    if selection_level == "Dataset":
        st.header(f"Best Configuration for Dataset: **{selected_value}**")
    else:
        st.header(f"Best Configuration for Domain: **{selected_value}**")

    if selection_level == "Dataset":
        df3 = df[df["input_dataset"] == selected_value]
    else:
        df3 = df[df["domain"] == selected_value]

    group_fields = [
        "vector_db", "embedding_model", "chunking_type", "reranking_model",
        "repacking_strategy", "summarization_model", "generator_model", "template"
    ]

    metrics = [
        "context_relevance", "context_utilization", "adherence", "completeness",
        "hallucination_auroc", "relevance_rmse", "utilization_rmse"
    ]

    if df3.empty:
        st.warning("No data available for the selected dataset.")
    else:
        # Group and compute mean of metrics
        grouped = df3.groupby(group_fields)[metrics].mean().reset_index()

        # Normalize each metric
        norm_cols = []
        for metric in metrics:
            min_val = grouped[metric].min()
            max_val = grouped[metric].max()
            if min_val == max_val:
                norm = 1.0
            else:
                norm = (grouped[metric] - min_val) / (max_val - min_val)

            if metric in ["relevance_rmse", "utilization_rmse"]:
                norm = 1 - norm  # Invert because lower is better

            norm_col = f"{metric}_norm"
            grouped[norm_col] = norm
            norm_cols.append(norm_col)

        # Calculate total normalized score
        grouped["total_score"] = grouped[norm_cols].sum(axis=1)

        # Sort and select top 3 configurations
        top_n = 5
        best_configs = grouped.sort_values(by="total_score", ascending=False).head(top_n)

        st.markdown(f"### Top {top_n} Configurations Based on Normalized Score")
        st.dataframe(
                    best_configs[group_fields + metrics + ["total_score"]]
                    .style.format(precision=3),
                    width=1400  # You can increase or decrease this based on screen size
                )

        # Optional: Chart for visual comparison
        # st.markdown("### 📊 Score Comparison")
        # chart = (
        #     alt.Chart(best_configs)
        #     .mark_bar()
        #     .encode(
        #         x=alt.X("total_score:Q", title="Total Normalized Score"),
        #         y=alt.Y("summarization_model:N", title="Summarization Model"),
        #         color=alt.Color("vector_db:N", title="Vector DB"),
        #         tooltip=group_fields + metrics + ["total_score"]
        #     )
        #     .properties(height=350)
        # )
        # st.altair_chart(chart, use_container_width=True)
        # with st.expander("📋 Score Comparison Table"):
        #     score_cols = ["total_score"] + metrics
        #     score_display = best_configs[["summarization_model", "vector_db"] + score_cols].copy()
        #     score_display = score_display.rename(columns=lambda c: c.replace("_", " ").title())
        #     st.dataframe(score_display.style.format(precision=3), use_container_width=True)
        st.markdown("### 🏆 Top 5 Configurations Leaderboard")

        podium = best_configs.reset_index(drop=True)
        
        # Top 3 - Podium Style
        col1, col2, col3 = st.columns([1, 1.2, 1])  # 2nd, 1st, 3rd

        with col1:
            st.markdown("### 🥇 1st")
            st.metric("Score", f"{podium.loc[0, 'total_score']:.3f}")
            st.markdown(f"**Vector DB:** {podium.loc[0, 'vector_db']}")
            st.markdown(f"**Summarization:** {podium.loc[0, 'summarization_model']}")
            st.markdown(f"**Generator:** {podium.loc[0, 'generator_model']}")
            st.markdown(f"**Reranking Model:** {podium.loc[0, 'reranking_model']}")
        
        with col2:
            st.markdown("### 🥈 2nd")
            st.metric("Score", f"{podium.loc[1, 'total_score']:.3f}")
            st.markdown(f"**Vector DB:** {podium.loc[1, 'vector_db']}")
            st.markdown(f"**Summarization:** {podium.loc[1, 'summarization_model']}")
            st.markdown(f"**Generator:** {podium.loc[1, 'generator_model']}")
            st.markdown(f"**Reranking Model:** {podium.loc[1, 'reranking_model']}")
        
        with col3:
            st.markdown("### 🥉 3rd")
            st.metric("Score", f"{podium.loc[2, 'total_score']:.3f}")
            st.markdown(f"**Vector DB:** {podium.loc[2, 'vector_db']}")
            st.markdown(f"**Summarization:** {podium.loc[2, 'summarization_model']}")
            st.markdown(f"**Generator:** {podium.loc[2, 'generator_model']}")
            st.markdown(f"**Reranking Model:** {podium.loc[2, 'reranking_model']}")
        
        # 4th and 5th below
        st.markdown("### 🎖️ Honorable Mentions")
        
        for idx in range(3, min(5, len(podium))):  # only access rows that exist
            config = podium.iloc[idx]
            with st.expander(f"#{idx+1} - Score: {config['total_score']:.3f}"):
                st.markdown(f"**Vector DB:** {config['vector_db']}")
                st.markdown(f"**Summarization Model:** {config['summarization_model']}")
                st.markdown(f"**Generator Model:** {config['generator_model']}")
                st.markdown(f"**Reranking Model:** {config['reranking_model']}")
