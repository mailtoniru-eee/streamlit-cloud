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

tab1, tab2 = st.tabs(["📊 Dashboard", "🖼️ Image Viewer"])

with tab1:
    # Slicers
    st.sidebar.header("🔍 Single Select Filters")
    
    input_dataset = st.sidebar.selectbox("Dataset", sorted(df["input_dataset"].dropna().unique()))
    
    # Apply top-level filter
    filtered_df = df[df["input_dataset"] == input_dataset]
    
    # Second-level slicers
    vector_db = st.sidebar.selectbox("Vector DB", sorted(filtered_df["vector_db"].dropna().unique()))
    reranking_model = st.sidebar.selectbox("Reranking Model", sorted(filtered_df["reranking_model"].dropna().unique()))
    repacking_strategy = st.sidebar.selectbox("Repacking Strategy", sorted(filtered_df["repacking_strategy"].dropna().unique()))
    summarization_model = st.sidebar.selectbox("Summarization Model", sorted(filtered_df["summarization_model"].dropna().unique()))
    
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
    st.header("📊 Metric Averages (One Chart per Metric)")
    st.sidebar.header("🔍 Multi Select Filters")
    vector_db = st.sidebar.multiselect("Vector DB", sorted(filtered_df["vector_db"].dropna().unique()))
    reranking_model = st.sidebar.multiselect("Reranking Model", sorted(filtered_df["reranking_model"].dropna().unique()))
    repacking_strategy = st.sidebar.multiselect("Repacking Strategy", sorted(filtered_df["repacking_strategy"].dropna().unique()))
    summarization_model = st.sidebar.multiselect("Summarization Model", sorted(filtered_df["summarization_model"].dropna().unique()))
    
