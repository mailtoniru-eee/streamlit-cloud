import streamlit as st
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

st.subheader("Group 23 - RAG Application - RAGBench Dataset")
col1, col2, col3 = st.columns([1, 2, 1])  # Center column is wider
with col2:
    st.image(image, caption="Group 23 - RAG Application - RAGBench Dataset", width=300)
st.markdown("</div>", unsafe_allow_html=True)


df = get_data()

if not df.empty:
    unique_values = df["aggregate_id"].unique()
    selected_value = st.selectbox("Select value", unique_values)
    filtered_df = df[df["aggregate_id"] == selected_value]
    st.dataframe(filtered_df)
else:
    st.write("No data found.")
