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
    response = supabase.table("rag_metrics").select("*").execute()
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

st.sidebar.markdown("#### Available Datasets:")
st.sidebar.write(sorted(df["input_dataset"].unique()))

st.sidebar.write(f"🔎 Number of rows fetched: {len(df)}")

# --------------------- UI Header ------------------------
st.subheader("Group 23 - RAG Application - RAGBench Dataset")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(image, width=300)

# ---------------- Sidebar: Shared Filters ----------------
input_dataset = st.sidebar.selectbox("Dataset", sorted(df["input_dataset"].dropna().unique()))

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard - Individual", "⚖️ Dashboard - Comparison", "🏆 Best Config"])

# ---------------- TAB 1 ----------------
with tab1:
    df1 = df[df["input_dataset"] == input_dataset]

    # Second-level slicers
    vector_db = st.selectbox("Vector DB", sorted(df1["vector_db"].dropna().unique()))
    embedding_model = st.selectbox("Embedding Model", sorted(df1["embedding_model"].dropna().unique()))
    reranking_model = st.selectbox("Reranking Model", sorted(df1["reranking_model"].dropna().unique()))
    repacking_strategy = st.selectbox("Repacking Strategy", sorted(df1["repacking_strategy"].dropna().unique()))
    summarization_model = st.selectbox("Summarization Model", sorted(df1["summarization_model"].dropna().unique()))
    generator_model = st.selectbox("Generator Model", sorted(df1["generator_model"].dropna().unique()))

    # Apply filters
    filtered_df = df1[
        (df1["vector_db"] == vector_db) &
        (df1["embedding_model"] == embedding_model) &
        (df1["reranking_model"] == reranking_model) &
        (df1["repacking_strategy"] == repacking_strategy) &
        (df1["summarization_model"] == summarization_model) &
        (df1["generator_model"] == generator_model)
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

    st.subheader("📊 Metric Averages for Selected Configuration")

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
    df2 = df[df["input_dataset"] == input_dataset]

    st.header("📊 Metric Comparison Across Configurations")

    vector_dbs = st.multiselect("Vector DB", df2["vector_db"].dropna().unique())
    embedding_models = st.multiselect("Embedding Model", df2["embedding_model"].dropna().unique())
    reranking_models = st.multiselect("Reranking Model", df2["reranking_model"].dropna().unique())
    repacking_strategies = st.multiselect("Repacking Strategy", df2["repacking_strategy"].dropna().unique())
    summarization_models = st.multiselect("Summarization Model", df2["summarization_model"].dropna().unique())
    generator_models = st.multiselect("Generator Model", df2["generator_model"].dropna().unique())

    # Apply filters
    filtered_df = df2.copy()

    if vector_dbs:
        filtered_df = filtered_df[filtered_df["vector_db"].isin(vector_dbs)]
    if embedding_models:
        filtered_df = filtered_df[filtered_df["embedding_model"].isin(vector_dbs)]        
    if reranking_models:
        filtered_df = filtered_df[filtered_df["reranking_model"].isin(reranking_models)]
    if repacking_strategies:
        filtered_df = filtered_df[filtered_df["repacking_strategy"].isin(repacking_strategies)]
    if summarization_models:
        filtered_df = filtered_df[filtered_df["summarization_model"].isin(summarization_models)]
    if generator_models:
        filtered_df = filtered_df[filtered_df["generator_model"].isin(generator_models)]        

    group_fields = [
        "input_dataset", "vector_db", "embedding_model", "reranking_model",
        "repacking_strategy", "summarization_model", "generator_model"
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

# with tab3:
#     st.header("🏆 Best Configuration for Selected Dataset")

#     df3 = df[df["input_dataset"] == input_dataset]

#     group_fields = [
#         "vector_db", "reranking_model", "repacking_strategy", "summarization_model"
#     ]

#     metrics = [
#         "context_relevance", "context_utilization", "adherence", "completeness",
#         "hallucination_auroc", "relevance_rmse", "utilization_rmse"
#     ]

#     if df3.empty:
#         st.warning("No data available for the selected dataset.")
#     else:
#         # Group by config and compute mean of each metric
#         grouped = df3.groupby(group_fields)[metrics].mean().reset_index()

#         # Calculate a combined score (you can adjust weights if needed)
#         grouped["total_score"] = grouped[metrics].sum(axis=1)

#         # Get row(s) with max total score
#         best_configs = grouped.sort_values(by="total_score", ascending=False).head(3)

#         st.markdown("### 🥇 Top Configurations")
#         st.dataframe(best_configs.style.format(precision=3), use_container_width=True)

#         # Optional: show bar chart of top 3 total scores
#         st.markdown("### 📊 Score Comparison")
#         chart = (
#             alt.Chart(best_configs)
#             .mark_bar()
#             .encode(
#                 x=alt.X("total_score:Q", title="Total Score"),
#                 y=alt.Y("vector_db:N", title="Vector DB"),
#                 color=alt.Color("reranking_model:N", title="Reranker"),
#                 tooltip=group_fields + ["total_score"]
#             )
#             .properties(height=300)
#         )
#         st.altair_chart(chart, use_container_width=True)

with tab3:
    st.header("🏆 Best Configuration for Selected Dataset")

    df3 = df[df["input_dataset"] == input_dataset]

    group_fields = [
        "vector_db", "embedding_model", "reranking_model",
        "repacking_strategy", "summarization_model", "generator_model"
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
        top_n = 3
        best_configs = grouped.sort_values(by="total_score", ascending=False).head(top_n)

        st.markdown(f"### 🥇 Top {top_n} Configurations Based on Normalized Score")
        st.dataframe(
            best_configs[group_fields + metrics + ["total_score"]]
            .style.format(precision=3),
            use_container_width=True
        )

        # Optional: Chart for visual comparison
        st.markdown("### 📊 Score Comparison")
        chart = (
            alt.Chart(best_configs)
            .mark_bar()
            .encode(
                x=alt.X("total_score:Q", title="Total Normalized Score"),
                y=alt.Y("summarization_model:N", title="Summarization Model"),
                color=alt.Color("vector_db:N", title="Vector DB"),
                tooltip=group_fields + metrics + ["total_score"]
            )
            .properties(height=350)
        )
        st.altair_chart(chart, use_container_width=True)
