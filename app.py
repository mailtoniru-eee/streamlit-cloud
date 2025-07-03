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
    st.image(image, width=300)
st.markdown("</div>", unsafe_allow_html=True)


df = get_data()

tab1, tab2 = st.tabs(["📊 Dashboard", "🖼️ Image Viewer"])

with tab1:
    st.header("Analytics Dashboard")
    st.line_chart([1, 5, 2, 6])

with tab2:
    st.header("Image Viewer")
