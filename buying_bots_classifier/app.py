import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime
import random
import string
import joblib # Import joblib for model persistence
from ml_classifier import AdvancedBotClassifier

# --- Path Configuration ---
# Get the current script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define data and model directories
# CORRECTED: Changed "models" to "model" to match ml_classifier.py
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_DIR = os.path.join(SCRIPT_DIR, "model") 

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Page Setup ---
st.set_page_config(page_title="Amazon Bot Detector", layout="wide")

# --- Load External Dark Theme CSS ---
css_path = os.path.join(SCRIPT_DIR, "styles.css")
try:
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("styles.css not found. Using default styling.")

st.markdown('<div class="title-style">🛒 Amazon Bot Detection Simulator</div>', unsafe_allow_html=True)
st.write("Train on historical data, then simulate new purchases to detect bot vs human behavior.")

# --- File Structure with Proper Paths ---
# Update file paths to use DATA_DIR
training_file_map = {
    "Laptop": os.path.join(DATA_DIR, "training_data.xlsx"),
    "Jeans": os.path.join(DATA_DIR, "training_data.xlsx"),
    "Decoration Lamp": os.path.join(DATA_DIR, "training_data.xlsx")
}
product_file_map = {
    "Laptop": os.path.join(DATA_DIR, "laptop.xlsx"),
    "Jeans": os.path.join(DATA_DIR, "jeans.xlsx"),
    "Decoration Lamp": os.path.join(DATA_DIR, "decoration_lamp.xlsx")
}

# --- Helper: File Creation ---
def create_file_if_not_exists(file_path, is_training=False):
    """
    Creates an empty Excel file with predefined columns if it doesn't exist.
    Adds a 'Label' column if it's a training file.
    """
    if not os.path.exists(file_path):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        columns = [
            "DateTime", "BuyingTime_Seconds", "PageViewDuration", "CartTime_Seconds",
            "IP_Address", "UserID", "CouponUsed", "DiscountApplied", "PaymentMethod",
            "ProductViewCount", "ProductSearchCount", "AddToCart_RemoveCount",
            "ReviewsRead", "DeviceType", "MouseClicks", "KeyboardStrokes", "ProductID"
        ]
        if is_training:
            columns.append("Label")
        pd.DataFrame(columns=columns).to_excel(file_path, index=False, engine='openpyxl')
        st.info(f"Created empty file: {file_path}")

# Create necessary files: product files for simulation, and ensuring the single training file exists
for product in product_file_map:
    create_file_if_not_exists(product_file_map[product], is_training=False)

# Create the single training data file
training_data_path = os.path.join(DATA_DIR, "training_data.xlsx")
create_file_if_not_exists(training_data_path, is_training=True)

# --- Session State Initialization ---
if "bot_ids" not in st.session_state:
    st.session_state.bot_ids = []
if "bot_buy_count" not in st.session_state:
    st.session_state.bot_buy_count = {}
if "genuine_user_id" not in st.session_state:
    st.session_state.genuine_user_id = "HUMAN-" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
if "genuine_ip" not in st.session_state:
    st.session_state.genuine_ip = f"192.168.1.{random.randint(2, 254)}"
# New session state variables for model persistence
if "bot_classifier" not in st.session_state:
    st.session_state.bot_classifier = None
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "training_metrics_display" not in st.session_state: # To store metrics for display
    st.session_state.training_metrics_display = None
if "feature_importance_display" not in st.session_state: # To store feature importance for display
    st.session_state.feature_importance_display = None
if "training_report_display" not in st.session_state: # To store classification report
    st.session_state.training_report_display = None

BOT_IP_POOL = [
    "203.0.113.5", "198.51.100.10", "192.0.2.15",
    "203.0.113.20", "198.51.100.25", "192.0.2.30"
]

# --- Model Loading on App Start ---
# Try to load a default model (e.g., 'ensemble') when the app starts
if not st.session_state.model_trained and st.session_state.bot_classifier is None:
    # CORRECTED: Pass base_path=SCRIPT_DIR instead of data_dir, model_dir
    temp_classifier = AdvancedBotClassifier(product_file_map, base_path=SCRIPT_DIR)
    # Attempt to load the ensemble model as a default
    if temp_classifier.load_model('ensemble'):
        st.session_state.bot_classifier = temp_classifier
        st.session_state.model_trained = True
        st.session_state.training_metrics_display = temp_classifier.get_model_performance()
        st.success("✅ Previously trained 'ensemble' model loaded successfully!")
    else:
        st.info("No pre-trained model found. Please train a model.")

# --- Product & Model Selection ---
st.markdown("## 🎯 Model Training & Product Selection")
col1, col2 = st.columns(2)
product_names = list(product_file_map.keys())
model_options = ["random_forest", "xgboost", "lightgbm", "ensemble", "logistic_regression", "svc"]

with col1:
    selected_product = st.selectbox("📦 Select a Product", product_names)
    selected_file = product_file_map[selected_product]

with col2:
    selected_model = st.selectbox("🧠 Select ML Model", model_options)

# --- Model Training Section ---
st.markdown("### 🔄 Train Models")
train_col1, train_col2, train_col3 = st.columns(3)

with train_col1:
    if st.button("🔁 Train Selected Model"):
        with st.spinner(f"Training {selected_model} model using 'training_data.xlsx'..."):
            # Initialize the classifier with proper paths
            # CORRECTED: Pass base_path=SCRIPT_DIR instead of data_dir, model_dir
            classifier = AdvancedBotClassifier(product_file_map, base_path=SCRIPT_DIR)
            # Force the algorithm to be the selected one for this training run
            training_result = classifier.train_model(algorithm=selected_model)

            if training_result["trained"]:
                st.session_state.bot_classifier = classifier # Store the trained classifier
                st.session_state.model_trained = True
                st.session_state.training_metrics_display = training_result.get("metrics")
                st.session_state.feature_importance_display = training_result.get("feature_importance")
                st.session_state.training_report_display = training_result.get("report")
                st.success(f"✅ {training_result['message']}")
            else:
                st.session_state.model_trained = False
                st.warning(f"⚠️ {training_result['message']}")
            # Ensure the currently loaded classifier matches the last trained one for single model training
            if st.session_state.bot_classifier and st.session_state.model_trained:
                st.session_state.bot_classifier.algorithm = selected_model # Update algorithm in loaded classifier

with train_col2:
    if st.button("📊 Show Last Trained Metrics", disabled=not st.session_state.model_trained):
        if st.session_state.model_trained and st.session_state.bot_classifier:
            current_metrics = st.session_state.bot_classifier.get_model_performance()
            if current_metrics:
                st.markdown("#### 📈 Key Metrics (Last Trained Model)")
                metrics_df = pd.DataFrame([current_metrics]).transpose()
                metrics_df.columns = ["Value"]
                st.dataframe(metrics_df)

                if st.session_state.feature_importance_display is not None and not st.session_state.feature_importance_display.empty:
                    st.markdown("#### 💡 Feature Importance (Last Trained Model)")
                    st.dataframe(st.session_state.feature_importance_display)
                else:
                    st.info("No feature importance data available for the last trained model.")

                if st.session_state.training_report_display is not None:
                    st.markdown("#### 📊 Classification Report (Last Trained Model)")
                    st.dataframe(st.session_state.training_report_display)
            else:
                st.info("No metrics available for the last trained model. Please train a model first.")
        else:
            st.warning("No model trained or loaded yet.")

with train_col3:
    if st.button("⚡ Train All Models on Combined Data"):
        st.info("Starting comprehensive training on 'training_data.xlsx' for all models...")
        all_training_results = []
        progress_bar = st.progress(0)
        total_tasks = len(model_options)
        task_count = 0

        # Create one classifier instance for this comprehensive training run
        # CORRECTED: Pass base_path=SCRIPT_DIR
        current_classifier = AdvancedBotClassifier(product_file_map, base_path=SCRIPT_DIR)

        for algo in model_options:
            task_count += 1
            progress_bar.progress(task_count / total_tasks)
            st.text(f"    Training {algo} on combined data...")

            result = current_classifier.train_model(algorithm=algo)

            if result["trained"]:
                st.success(f"    ✅ {algo.upper()}: {result['message']}")
            else:
                st.warning(f"    ⚠️ {algo.upper()}: {result['message']}")
            all_training_results.append(("Combined Data", algo, result))

        progress_bar.progress(1.0)
        st.success("Comprehensive training completed!")
        st.markdown("#### Comprehensive Training Summary")
        for prod, algo, res in all_training_results:
            status = "✅ Trained" if res["trained"] else "⚠️ Failed"
            msg = res["message"].split('\n')[0]
            st.write(f"- {status} {algo.upper()} for {prod}: {msg}")

# --- Human or Bot Entry Simulation ---
st.markdown("### 🧾 Add New Purchase Entry")
entry_type = st.radio("Choose Entry Type", ["🧍 Human Buy", "🤖 Bot Buy"])

if entry_type == "🧍 Human Buy":
    with st.form("human_form"):
        coupon_used = st.selectbox("Coupon Used?", ["No", "Yes"])
        discount_applied = st.selectbox("Discount Available?", ["No", "Yes"])
        cart_activity = st.slider("Add/Remove Count", 0, 5, 1)
        payment_method = st.selectbox("Payment Method", ["COD", "UPI", "Debit Card", "Credit Card", "Internet Banking"])
        submit_human = st.form_submit_button("💾 Add Human Entry")

    if submit_human:
        new_entry = {
            "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "BuyingTime_Seconds": random.randint(60, 300),
            "PageViewDuration": random.randint(30, 180),
            "CartTime_Seconds": random.randint(10, 60),
            "IP_Address": st.session_state.genuine_ip,
            "UserID": st.session_state.genuine_user_id,
            "CouponUsed": coupon_used,
            "DiscountApplied": discount_applied,
            "PaymentMethod": payment_method,
            "ProductViewCount": random.randint(3, 10),
            "ProductSearchCount": random.randint(1, 5),
            "AddToCart_RemoveCount": cart_activity,
            "ReviewsRead": random.randint(1, 10),
            "DeviceType": random.choice(["Desktop", "Mobile", "Tablet"]),
            "MouseClicks": random.randint(10, 50),
            "KeyboardStrokes": random.randint(20, 100),
            "ProductID": selected_product
        }
        df = pd.read_excel(selected_file, engine='openpyxl')
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_excel(selected_file, index=False, engine='openpyxl')
        st.success("✅ Human entry added!")

elif entry_type == "🤖 Bot Buy":
    with st.form("bot_form"):
        num_bots = st.slider("Number of Bot Buys", 1, 10, 3)
        submit_bot = st.form_submit_button("🚀 Simulate Bot Buys")

    if submit_bot:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        bot_entries = []

        for _ in range(num_bots):
            bot_id = "BOT-" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
            st.session_state.bot_ids.append(bot_id)

            entry = {
                "DateTime": timestamp,
                "BuyingTime_Seconds": random.randint(1, 10),
                "PageViewDuration": random.randint(0, 5),
                "CartTime_Seconds": random.randint(0, 3),
                "IP_Address": random.choice(BOT_IP_POOL),
                "UserID": bot_id,
                "CouponUsed": "Yes",
                "DiscountApplied": "Yes",
                "PaymentMethod": random.choice(["Credit Card", "Internet Banking"]),
                "ProductViewCount": random.randint(0, 2),
                "ProductSearchCount": 0,
                "AddToCart_RemoveCount": 0,
                "ReviewsRead": 0,
                "DeviceType": "Desktop",
                "MouseClicks": random.randint(3, 8),
                "KeyboardStrokes": random.randint(0, 5),
                "ProductID": selected_product
            }
            bot_entries.append(entry)

        df = pd.read_excel(selected_file, engine='openpyxl')
        df = pd.concat([df, pd.DataFrame(bot_entries)], ignore_index=True)
        df.to_excel(selected_file, index=False, engine='openpyxl')
        st.success(f"✅ {num_bots} bot entries added!")

# --- Prediction on Current File (only visible if a model is trained/loaded) ---
if st.session_state.model_trained:
    st.markdown("### 🔍 Run Detection on Current File")
    if st.button("🚨 Predict & Flag Suspicious Entries"):
        df = pd.read_excel(selected_file, engine='openpyxl')
        if df.empty:
            st.warning("No data to analyze.")
        else:
            with st.spinner("Running predictions..."):
                if st.session_state.bot_classifier:
                    # It's crucial to load the model again with the correct algorithm
                    # especially if the user has switched selected_model since the last training/load
                    st.session_state.bot_classifier.load_model(selected_model) 
                    pred_df = st.session_state.bot_classifier.predict_bot_probability(df)

                    if pd.api.types.is_numeric_dtype(pred_df["ML_Bot_Chance (%)"]):
                        pred_df_sorted = pred_df.sort_values(by="ML_Bot_Chance (%)", ascending=False)
                    else:
                        pred_df_sorted = pred_df

                    st.markdown("### 📋 Suspicious Entries (Sorted by Bot Likelihood)")
                    st.dataframe(pred_df_sorted)
                else:
                    st.error("No classifier instance available. Please train or load a model.")
else:
    st.info("Train a model to enable prediction features.")

# --- Footer ---
st.markdown("---")
st.caption("🔒 Built for detecting fraud bots during flash sales | Streamlit x Machine Learning")
