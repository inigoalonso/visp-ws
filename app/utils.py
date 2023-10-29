"""
Utility functions for:
    1. reading data
    2. setting background
    3. writing head, body, and footer
    4. reporting problems
"""

import json
import base64
import pandas as pd
import streamlit as st


@st.cache_data()
def read_data(path):
    return pd.read_csv(path)

@st.cache_data()
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = """
        <style>
        .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        }
        </style>
    """ % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)