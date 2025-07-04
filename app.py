import streamlit as st
import altair as alt
from supabase import create_client, Client
import pandas as pd
import os
import requests
from PIL import Image
from io import BytesIO
import json

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

@st.cache_data
def get_data():
    response = supabase.table("rag_metrics").select("*").execute()
    return pd.DataFrame(response.data)

df = get_data()

input_vars_df = pd.json_normalize(df["input_variable"])

def extract_first(dataset):
    if isinstance(dataset, list) and len(dataset) > 0:
        return dataset[0]
    return dataset  # fallback

input_vars_df["input_dataset"] = input_vars_df["input_dataset"].apply(extract_first)

df = df.drop(columns=["input_variable"]).reset_index(drop=True)
input_vars_df = input_vars_df.reset_index(drop=True)
df = pd.concat([df, input_vars_df], axis=1)

st.subheader("Group 23 - RAG Application - RAGBench Dataset")
col1, col2, col3 = st.columns([1, 2, 1])  # Center column is wider
with col2:
    st.image(image, width=300)
    st.markdown("</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🖼️ Image Viewer", "Testing"])

with tab1:
    df1 = df.copy()
    # Slicers
    input_dataset = st.sidebar.selectbox("Dataset", sorted(df1["input_dataset"].dropna().unique()))
    
    # Apply top-level filter
    filtered_df = df1[df1["input_dataset"] == input_dataset]
    
    # Second-level slicers
    vector_db = st.selectbox("Vector DB", sorted(filtered_df["vector_db"].dropna().unique()))
    reranking_model = st.selectbox("Reranking Model", sorted(filtered_df["reranking_model"].dropna().unique()))
    repacking_strategy = st.selectbox("Repacking Strategy", sorted(filtered_df["repacking_strategy"].dropna().unique()))
    summarization_model = st.selectbox("Summarization Model", sorted(filtered_df["summarization_model"].dropna().unique()))
    
    # Apply all filters
    filtered_df = filtered_df[
        (filtered_df["vector_db"] == vector_db) &
        (filtered_df["reranking_model"] == reranking_model) &
        (filtered_df["repacking_strategy"] == repacking_strategy) &
        (filtered_df["summarization_model"] == summarization_model)
    ]
    
    metrics = [
        "context_relevance",
        "context_utilization",
        "adherence",
        "completeness",
        "hallucination_auroc",
        "relevance_rmse",
        "utilization_rmse"
        ]
    
    # 2. Compute average of each metric over the filtered rows
    metric_averages = {metric: filtered_df[metric].mean() for metric in metrics}
    
    # 3. Convert to DataFrame for Altair
    avg_df = pd.DataFrame({
        "Metric": list(metric_averages.keys()),
        "Average": list(metric_averages.values())
    })

    # 4. Plot one bar chart
    st.subheader("📊 Metric Averages for Selected Configuration")

    chart = (
        alt.Chart(avg_df)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", sort=None),
            y=alt.Y("Average:Q"),
            color=alt.Color("Metric:N", legend=None),
            tooltip=["Metric", "Average"]
                )
                .properties(height=400)
            )

    st.altair_chart(chart, use_container_width=True)
    
with tab2:
    df2 = df.copy()
    
    # Slicers
    input_datasets = st.sidebar.selectbox("Dataset Selection", sorted(df1["input_dataset"].dropna().unique()))    
    
    st.header("📊 Metric Averages (One Chart per Metric)")
    vector_dbs = st.multiselect("Vector DB", df2["vector_db"].dropna().unique(), default=None)
    reranking_models = st.multiselect("Reranking Model", df2["reranking_model"].dropna().unique(), default=None)
    repacking_strategys = st.multiselect("Repacking Strategy", df2["repacking_strategy"].dropna().unique(), default=None)
    summarization_models = st.multiselect("Summarization Model", df2["summarization_model"].dropna().unique(), default=None)

    filtered_df = df2.copy()

    if vector_dbs:
        filtered_df = filtered_df[filtered_df["vector_db"].isin(vector_dbs)]
    
    if reranking_models:
        filtered_df = filtered_df[filtered_df["reranking_model"].isin(reranking_models)]
    
    if repacking_strategys:
        filtered_df = filtered_df[filtered_df["repacking_strategy"].isin(repacking_strategies)]
    
    if summarization_models:
        filtered_df = filtered_df[filtered_df["summarization_model"].isin(summarization_models)]

    group_fields = [
        "input_dataset", "vector_db", "reranking_model",
        "repacking_strategy", "summarization_model"
    ]
    
    metrics = [
        "context_relevance", "context_utilization", "adherence", "completeness",
        "hallucination_auroc", "relevance_rmse", "utilization_rmse"
    ]
    
    # Group and average
    grouped_df = (
        filtered_df.groupby(group_fields)[metrics]
        .mean()
        .reset_index()
    )
    
    # Create a label for each config to show in chart
    grouped_df["config_label"] = grouped_df[group_fields].agg(" | ".join, axis=1)

    # Prepare for grouped bar chart
    chart_df = grouped_df.melt(
        id_vars=["config_label"], 
        value_vars=metrics, 
        var_name="Metric", 
        value_name="Average"
    )
    
    st.subheader("📊 Metric Comparison Across Configurations")

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", title="Metric", sort=metrics),
            y=alt.Y("Average:Q", title="Average Score"),
            color=alt.Color("config_label:N", title="Configuration", legend=alt.Legend(orient="bottom", columns=2, labelFontSize=12, titleFontSize=14)),
            #column=alt.Column("Metric:N", title="Metric", sort=metrics),  # This does the grouping
            xOffset="config_label:N",  # <- 👈 KEY for grouping side-by-side
            tooltip=["config_label", "Metric", "Average"]
        )
        .properties(
            width=2000,
            height=500
            )
        )
    
    st.altair_chart(chart, use_container_width=True)
    
    # chart = (
    #     alt.Chart(chart_df)
    #     .mark_bar()
    #     .encode(
    #         x=alt.X("Metric:N", title="Metric", sort=metrics),
    #         y=alt.Y("Average:Q"),
    #         color="config_label:N",
    #         column=alt.Column("config_label:N", title="Configuration"),
    #         tooltip=["config_label", "Metric", "Average"]
    #     )
    #     .resolve_scale(y="shared")
    #     .properties(height=300)
    # )

    # chart = (
    #     alt.Chart(chart_df)
    #     .mark_bar()
    #     .encode(
    #         x=alt.X("Metric:N", title="Metric", sort=metrics),
    #         y=alt.Y("Average:Q", title="Average Score"),
    #         color=alt.Color("config_label:N", title="Configuration"),
    #         tooltip=["config_label", "Metric", "Average"]
    #     )
    #     .properties(
    #         width=600,
    #         height=400,
    #         title="Grouped Bar Chart: Metric Comparison by Configuration"
    #         )
    # )
    
    # st.altair_chart(chart, use_container_width=True)

    
    
# with tab3:
#     df3 = df.copy()
    
#     st.header("📊 Metric Grouped (One Chart per Metric)")
#     vector_dbs = st.multiselect("Vector DB", df3["vector_db"].dropna().unique(), default=None)
#     reranking_models = st.multiselect("Reranking Model", df3["reranking_model"].dropna().unique(), default=None)
#     repacking_strategys = st.multiselect("Repacking Strategy", df3["repacking_strategy"].dropna().unique(), default=None)
#     summarization_models = st.multiselect("Summarization Model", df3["summarization_model"].dropna().unique(), default=None)

#     filtered_df = df3.copy()

#     if vector_dbs:
#         filtered_df = filtered_df[filtered_df["vector_db"].isin(vector_dbs)]
    
#     if reranking_models:
#         filtered_df = filtered_df[filtered_df["reranking_model"].isin(reranking_models)]
    
#     if repacking_strategys:
#         filtered_df = filtered_df[filtered_df["repacking_strategy"].isin(repacking_strategies)]
    
#     if summarization_models:
#         filtered_df = filtered_df[filtered_df["summarization_model"].isin(summarization_models)]

#     group_fields = [
#         "input_dataset", "vector_db", "reranking_model",
#         "repacking_strategy", "summarization_model"
#     ]
    
#     metrics = [
#         "context_relevance", "context_utilization", "adherence", "completeness",
#         "hallucination_auroc", "relevance_rmse", "utilization_rmse"
#     ]
    
#     # Group and average
#     clustered_df = (
#         filtered_df.groupby(group_fields)[metrics]
#         .mean()
#         .reset_index()
#     )
    
#     # Create a label for each config to show in chart
#     clustered_df["config_label"] = clustered_df[group_fields].agg(" | ".join, axis=1)

#     # Prepare for grouped bar chart
#     chart_df = grouped_df.melt(
#         id_vars=["config_label"], 
#         value_vars=metrics, 
#         var_name="Metric", 
#         value_name="Average"
#     )
    
#     st.subheader("📊 Grouped Metric Comparison Across Configurations")
    
#     chart = (
#             alt.Chart(chart_df)
#             .mark_bar()
#             .encode(
#                 x=alt.X("Metric:N", title="Metric", sort=metrics),
#                 y=alt.Y("Average:Q", title="Average Score"),
#                 color=alt.Color("config_label:N", title="Configuration"),
#                 tooltip=["config_label", "Metric", "Average"]
#             )
#             .properties(
#                 width=600,
#                 height=400,
#                 title="Grouped Bar Chart: Metric Comparison by Configuration"
#             )
#         )
    
#     st.altair_chart(chart, use_container_width=True)
