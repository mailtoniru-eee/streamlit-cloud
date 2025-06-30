import streamlit as st
from supabase import create_client, Client
import pandas as pd
import os

# Read credentials from environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase credentials are not set. Please set SUPABASE_URL and SUPABASE_KEY as environment variables.")
    st.stop()

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase: Client = init_supabase()

@st.cache_data
def get_data():
    response = supabase.table("your_table").select("*").execute()
    return pd.DataFrame(response.data)

st.title("Interactive Dashboard Example with Supabase")

df = get_data()

if not df.empty:
    unique_values = df["some_column"].unique()
    selected_value = st.selectbox("Select value", unique_values)
    filtered_df = df[df["some_column"] == selected_value]
    st.dataframe(filtered_df)
else:
    st.write("No data found.")