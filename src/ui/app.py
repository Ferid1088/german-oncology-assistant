import os
import streamlit as st
from dotenv import load_dotenv
from src.ui.components.chat_page import render_chat_page
from src.ui.components.filters import render_filters

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "dev-secret-key")

st.set_page_config(
    page_title="Onkologie Leitlinien-Assistent",
    page_icon="🏥",
    layout="wide",
)

filters = render_filters()
render_chat_page(api_url=API_URL, api_key=API_KEY, filters=filters)
