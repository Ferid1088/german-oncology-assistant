import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from dotenv import load_dotenv
from src.ui.components.chat_page import render_chat_page

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "dev-secret-key")

st.set_page_config(
    page_title="Ola · Onkologie-Assistent",
    page_icon="🏥",
    layout="wide",
)

render_chat_page(api_url=API_URL, api_key=API_KEY)
