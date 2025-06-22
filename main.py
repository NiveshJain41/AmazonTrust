import streamlit as st
import subprocess
import os
import sys

# Configure page
st.set_page_config(
    page_title="Amazon Trust - Review & Bot Detection",
    page_icon="📦",
    layout="wide"
)

# Load custom CSS
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Amazon-style header
st.markdown("""
<div class="amazon-header">
    <div class="header-content">
        <div class="logo-section">
            <h1 class="amazon-logo">amazon trust</h1>
            <span class="logo-subtitle">Review & Bot Detection Suite</span>
        </div>
        <div class="header-nav">
            <span class="nav-item">Tools</span>
            <span class="nav-item">Analytics</span>
            <span class="nav-item">Help</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main content area
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="hero-section">
    <h2 class="hero-title">Protect Your Amazon Experience</h2>
    <p class="hero-subtitle">Advanced AI tools to identify fake reviews and bot accounts</p>
</div>
""", unsafe_allow_html=True)

# Full paths to the app scripts
app1_path = os.path.join("fake_review_classifier", "app.py")
app2_path = os.path.join("buying_bots_classifier", "app.py")

# Function to launch a Streamlit app
def launch_streamlit_app(app_path):
    python_exec = sys.executable
    command = f"{python_exec} -m streamlit run {app_path}"
    subprocess.Popen(command, shell=True)

# App selection cards
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="app-card">
        <div class="card-icon">🔍</div>
        <h3 class="card-title">Fake Review Classifier</h3>
        <p class="card-description">
            Identify suspicious reviews using advanced machine learning algorithms. 
            Analyze review patterns, language, and metadata to detect fake content.
        </p>
        <div class="card-features">
            <span class="feature-tag">✓ AI / Script / Hijacked</span>
            <span class="feature-tag">✓ Pattern Analysis</span>
            <span class="feature-tag">✓ Robust</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Launch Fake Review Classifier", key="fake_review", help="Detect fake reviews with AI"):
        launch_streamlit_app(app1_path)
        st.success("🚀 Fake Review Classifier is launching...")

with col2:
    st.markdown("""
    <div class="app-card">
        <div class="card-icon">🤖</div>
        <h3 class="card-title">Buying Bots Classifier</h3>
        <p class="card-description">
            DIdentify suspicious shopping behaviors using advanced machine learning algorithms. Analyze user interaction patterns, purchase metrics, and session data to detect bot activity.
        </p>
        <div class="card-features">
            <span class="feature-tag">✓Ensemble Model</span>
            <span class="feature-tag">✓ Protect Platform</span>
            <span class="feature-tag">✓ Behavior Analysis</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Launch Buying Bots Classifier", key="bot_classifier", help="Identify bot accounts"):
        launch_streamlit_app(app2_path)
        st.success("🚀 Buying Bots Classifier is launching...")

# Additional info section
st.markdown("""
<div class="info-section">
    <div class="info-grid">
        <div class="info-item">
            <h4>🛡️ Customer Trust</h4>
            <p>Our tools help maintain the integrity of the Amazon marketplace by identifying deceptive practices.</p>
        </div>
        <div class="info-item">
            <h4>⚡Platform Saftey </h4>
            <p>Protect platform from bots and scripts</p>
        </div>
        <div class="info-item">
            <h4>📊 Latest Tect</h4>
            <p>Uses latest practices of Machine learning algorithm</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>Amazon Trust Detection Suite | Built with Streamlit | © 2025</p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)