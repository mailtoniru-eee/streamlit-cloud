import streamlit as st
import altair as alt
from supabase import create_client, Client
import pandas as pd
import os
import requests
from PIL import Image
from io import BytesIO

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

st.subheader("Group 23 - RAG Application - RAGBench Dataset")
col1, col2, col3 = st.columns([1, 2, 1])  # Center column is wider
with col2:
    st.image(image, width=300)
st.markdown("</div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Dashboard", "🖼️ Image Viewer"])

with tab1:
    st.header("Analytics Dashboard")

    st.write("input_variable column type example:", type(df["input_variable"].iloc[0]))
    st.write("Sample input_variable value:", df["input_variable"].iloc[0])
    
    # Flatten the 'input_variable' JSON column
    input_vars_df = pd.json_normalize(df["input_variable"])
    df = df.drop(columns=["input_variable"]).join(input_vars_df)
    
    # Slicers
    st.sidebar.header("🔍 Filters")
    
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
    
    # Metrics to plot
    metrics = [
        "context_relevance",
        "context_utilization",
        "adherence",
        "completeness",
        "hallucination_auroc",
        "relevance_rmse",
        "utilization_rmse"
    ]
    
    # Compute average by aggregate_id
    agg_df = (
        filtered_df.groupby("aggregate_id")[metrics]
        .mean()
        .reset_index()
        .melt(id_vars=["aggregate_id"], var_name="Metric", value_name="Average")
    )
    
    # Bar chart
    st.subheader("📊 Average Metrics by Aggregate ID")
    
    chart = (
        alt.Chart(agg_df)
        .mark_bar()
        .encode(
            x=alt.X("aggregate_id:N", title="Aggregate ID"),
            y=alt.Y("Average:Q"),
            color="Metric:N",
            tooltip=["Metric", "Average", "aggregate_id"]
        )
        .properties(width=800, height=400)
    )
    
    st.altair_chart(chart, use_container_width=True)

with tab2:
    st.header("Image Viewer")
