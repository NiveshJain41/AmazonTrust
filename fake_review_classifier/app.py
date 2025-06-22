# This is app.py (FIXED)
import streamlit as st
import os
from core_logic import MLIntegratedAppCore, Config  # Import core logic and config
from app_ui import AppUI  # Import UI handler

# Load custom CSS for styling
def load_custom_css(css_file="style.css"):
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    # Load styling
    load_custom_css()

    # Instantiate the core backend logic
    core_app = MLIntegratedAppCore()

    # Instantiate and run the frontend UI
    ui_app = AppUI(core_app)
    ui_app.run()

if __name__ == "__main__":
    main()