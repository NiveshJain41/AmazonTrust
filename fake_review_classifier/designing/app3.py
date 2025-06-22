import logging
import streamlit as st
import pandas as pd
import os
import time
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import google.generativeai as genai
from dataclasses import dataclass # <--- ADDED THIS LINE
import uuid
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix # Moved here from main class to avoid re-importing
import seaborn as sns # Moved here from main class to avoid re-importing
import matplotlib.pyplot as plt # Moved here from main class to avoid re-importing

# Set Streamlit's logging level to WARNING or ERROR to suppress INFO/DEBUG messages
# This helps keep your console cleaner.
logging.getLogger("streamlit").setLevel(logging.WARNING)

# Import your ML Classifier (assuming ml_classifier.py is in the same directory)
from ml_classifier import ReviewClassifier

# Configuration and Constants
class Config:
    DATA_FOLDER = "data"
    GEMINI_API_KEY = "AIzaSyB1O1HaUkHmXrakakKzU2CkQBtRi-H5kBQ"
    MODEL_NAME = "gemini-2.0-flash"

    CATEGORY_FILE_MAP = {
        "Product (Vivo Mobile)": "product_reviews.xlsx",
        "Seller (Nivesh and Yash Clothing)": "seller_reviews.xlsx",
        "Amazon Service (Prime Video)": "amazon_service_reviews.xlsx"
    }

    TRAINING_DATA_FILE = "training_data.xlsx"

    SCRIPT_TEMPLATES = [
        "Amazing product! Highly recommend it to everyone.",
        "Best value for money. Will buy again.",
        "Very satisfied. Works as described.",
        "Affordable and reliable. Go for it!",
        "Top-notch quality at this price.",
        "Excellent quality and fast delivery.",
        "Perfect for daily use. Great purchase!",
        "Outstanding performance. Worth every penny.",
        "Superb build quality. Exceeded expectations.",
        "Fantastic product. Will recommend to friends."
    ]

@dataclass
class ReviewFeatures:
    product_name: str
    review: str
    label: str
    stars: int = 0
    typing_time_seconds: float = 0
    bought: str = "No"
    ip_address: str = ""
    account_id: str = ""
    product_specific: str = "No"
    helpful_votes_exists: str = "0"
    helpful_votes_percent: str = "0%"
    device_fingerprint: str = "Reused"
    timestamp: str = ""
    review_id: str = ""

class ReviewGenerator:
    def __init__(self, api_key: str, model_name: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_ai_reviews(self, product_name: str, count: int = 10) -> List[str]:
        try:
            prompt = f"""Generate {count} realistic and diverse product reviews for '{product_name}'.
            Make them vary in:
            - Length (short to detailed)
            - Tone (positive, neutral, critical)
            - Focus areas (features, value, experience)
            - Writing style (casual to formal)

            Format: One review per line, no numbering, no quotes."""
            response = self.model.generate_content(prompt)
            # Clean up response text to ensure one review per line and remove quotes
            reviews = [r.strip().replace('"', '').replace("'", '') for r in response.text.split("\n") if r.strip()]
            return reviews[:count] # Ensure only 'count' reviews are returned
        except Exception as e:
            st.error(f"Error generating AI reviews: {str(e)}")
            return []

    def generate_hijacked_reviews(self, product_name: str, intent: str, count: int = 5) -> List[str]:
        try:
            prompt = f"""Generate {count} hijacked/misleading reviews for '{product_name}'
            based on this intent: {intent}

            Make them seem authentic but include misleading information, outdated references,
            or content that doesn't match the actual product.

            Format: One review per line, no numbering."""
            response = self.model.generate_content(prompt)
            return [r.strip() for r in response.text.split("\n") if r.strip()][:count]
        except Exception as e:
            st.error(f"Error generating hijacked reviews: {str(e)}")
            return []

class ReviewFeatureGenerator:
    SHARED_SCRIPT_IP = "192.168.10.100"
    SHARED_AI_IP = "10.0.0.50"

    @staticmethod
    def generate_account_id() -> str:
        return f"A{random.randint(10**11, 10**12-1)}"

    @staticmethod
    def generate_unique_ip() -> str:
        return f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"

    @staticmethod
    def generate_features(product_name: str, review: str, label: str,
                          rating: Optional[int] = None, typing_time: Optional[float] = None) -> ReviewFeatures:
        features = ReviewFeatures(
            product_name=product_name,
            review=review,
            label=label,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            review_id=str(uuid.uuid4())[:8],
            account_id=ReviewFeatureGenerator.generate_account_id()
        )

        if "Human" in label:
            features.stars = rating or random.randint(3, 5)
            features.typing_time_seconds = typing_time or random.uniform(250, 400)
            features.bought = "Yes"
            features.ip_address = ReviewFeatureGenerator.generate_unique_ip()
            features.product_specific = "Yes"
            features.helpful_votes_exists = "Ask user" # This is a placeholder for actual user interaction
            features.helpful_votes_percent = f"{random.randint(40, 85)}%"
            features.device_fingerprint = "Unique"
        elif "Script" in label:
            features.stars = random.choice([1, 5]) # Often extreme ratings for script reviews
            features.typing_time_seconds = random.uniform(5, 15) # Very fast typing
            features.bought = "No"
            features.ip_address = ReviewFeatureGenerator.SHARED_SCRIPT_IP
            features.product_specific = "No"
            features.helpful_votes_exists = "No"
            features.helpful_votes_percent = "0%"
            features.device_fingerprint = "Reused"
        elif "AI" in label:
            features.stars = random.choice([1, 5]) # Often extreme ratings for AI reviews
            features.typing_time_seconds = 0 # Instantaneous generation
            features.bought = "No"
            features.ip_address = ReviewFeatureGenerator.SHARED_AI_IP
            features.product_specific = "No"
            features.helpful_votes_exists = "No"
            features.helpful_votes_percent = f"{random.randint(5, 15)}%" # Low helpfulness
            features.device_fingerprint = "Reused"
        elif "Hijacked" in label:
            features.stars = random.randint(4, 5) # Hijacked reviews often try to appear positive
            features.typing_time_seconds = random.uniform(250, 400) # Similar to human for stealth
            features.bought = "Yes" # Often claimed to be bought
            features.ip_address = ReviewFeatureGenerator.generate_unique_ip()
            features.product_specific = "Misleading" # Key indicator
            features.helpful_votes_exists = "Yes" # May try to solicit votes
            features.helpful_votes_percent = f"{random.randint(20, 60)}%" # Mixed helpfulness
            features.device_fingerprint = "Unique" # Appear as new unique users
        return features

class DataManager:
    def __init__(self, data_folder: str):
        self.data_folder = data_folder
        os.makedirs(data_folder, exist_ok=True)

    def save_review(self, features: ReviewFeatures, category: str, file_map: Dict[str, str]):
        file_path = os.path.join(self.data_folder, file_map[category])
        review_dict = features.__dict__
        df_entry = pd.DataFrame([review_dict])

        if os.path.exists(file_path):
            existing = pd.read_excel(file_path)
            combined = pd.concat([existing, df_entry], ignore_index=True)
        else:
            combined = df_entry

        combined.to_excel(file_path, index=False)
        return file_path

    def load_reviews(self, category: str, file_map: Dict[str, str]) -> Optional[pd.DataFrame]:
        file_path = os.path.join(self.data_folder, file_map[category])
        if os.path.exists(file_path):
            try:
                return pd.read_excel(file_path)
            except Exception as e:
                st.error(f"Error loading {file_path}: {e}")
                return None
        return None

class MLIntegratedApp:
    def __init__(self):
        self.config = Config()
        self.generator = ReviewGenerator(self.config.GEMINI_API_KEY, self.config.MODEL_NAME)
        self.feature_gen = ReviewFeatureGenerator()
        self.data_manager = DataManager(self.config.DATA_FOLDER)
        self.ml_classifier = ReviewClassifier()

        # Initialize session state for ML training status
        if 'ml_trained' not in st.session_state:
            st.session_state.ml_trained = False

        # Attempt to load models on app start if not already trained/loaded
        if not self.ml_classifier.models_trained:
            st.session_state.ml_trained = self.ml_classifier.load_models() # Update session state based on load result

    def setup_page(self):
        st.set_page_config(page_title="ML-Enhanced Amazon Review Detection", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
        st.title("🤖 ML-Enhanced Amazon Fake Review Detection System")
        st.markdown("---")
        st.sidebar.header("🎯 Navigation")
        return st.sidebar.selectbox("Choose Action:", ["🏋️ Train ML Models", "📝 Add Reviews", "📊 ML Predictions", "📈 Results Dashboard"])

    def train_models_page(self):
        st.header("🏋️ Train ML Models")
        st.info("Train machine learning models on your review dataset to detect fake reviews automatically.")

        training_file = os.path.join(self.config.DATA_FOLDER, self.config.TRAINING_DATA_FILE)

        if not os.path.exists(training_file):
            st.warning(f"Training data file '{self.config.TRAINING_DATA_FILE}' not found.")
            st.info("Please ensure you have a training dataset with labeled reviews or generate one from existing data.")
            st.subheader("Use Existing Review Data for Training")

            all_data = []
            for category in self.config.CATEGORY_FILE_MAP:
                df = self.data_manager.load_reviews(category, self.config.CATEGORY_FILE_MAP)
                if df is not None and not df.empty:
                    all_data.append(df)

            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                # Ensure the 'label' column is consistent for training
                # This assumes labels like "Human", "Script", "AI", "Hijacked" without extra words.
                combined_df['label'] = combined_df['label'].apply(lambda x: x.split(' ')[0].strip() if isinstance(x, str) else x)
                st.write(f"Found {len(combined_df)} reviews in existing data for potential training.")
                if st.button("Create Training Dataset from Existing Data"):
                    combined_df.to_excel(training_file, index=False)
                    st.success(f"Training dataset created with {len(combined_df)} samples!")
                    st.rerun() # Rerun to show the training options now that file exists
            else:
                st.info("No existing review data found in data folder to create a training dataset. Add reviews first!")
            return

        training_df = pd.read_excel(training_file)

        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Samples", len(training_df))
        with col2: st.metric("Label Types", len(training_df['label'].value_counts()))
        with col3:
            if st.session_state.ml_trained:
                st.success("✅ Models Trained")
            else:
                st.warning("⏳ Not Trained")

        st.subheader("Training Data Distribution")
        fig = px.bar(x=training_df['label'].value_counts().index, y=training_df['label'].value_counts().values, title="Review Type Distribution in Training Data")
        fig.update_layout(xaxis_title="Review Type", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Train Models")
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Training ML models... This may take a moment."):
                if self.ml_classifier.train_models(training_file):
                    st.session_state.ml_trained = True
                    st.success("🎉 All models trained successfully!")
                    st.balloons()
                else:
                    st.error("❌ Training failed. Check your 'ml_classifier.py' and console for details.")

        if st.button("📂 Load Pre-trained Models"):
            with st.spinner("Loading pre-trained models..."):
                if self.ml_classifier.load_models():
                    st.session_state.ml_trained = True
                    st.success("✅ Pre-trained models loaded successfully!")
                    # You might want to remove balloons here as it's not a "creation"
                else:
                    st.error("❌ No pre-trained models found. Please train them first.")

    def add_reviews_page(self):
        st.header("📝 Add Reviews to Dataset")
        category = st.selectbox("Select Category:", list(self.config.CATEGORY_FILE_MAP.keys()))
        # Infer product name based on category for better user experience
        product_name_mapping = {
            "Product (Vivo Mobile)": "Vivo Mobile Phone",
            "Seller (Nivesh and Yash Clothing)": "Nivesh and Yash Cotton Shirt", # More specific product
            "Amazon Service (Prime Video)": "Amazon Prime Video Service" # More specific product
        }
        product_name = product_name_mapping.get(category, "Generic Product")
        st.subheader(f"Adding reviews for: **{product_name}**")

        tab1, tab2, tab3, tab4 = st.tabs(["👤 Human Review", "🤖 Script Review", "🔮 AI Review", "🎭 Hijacked Review"])
        with tab1: self.add_human_review(category, product_name)
        with tab2: self.add_script_review(category, product_name)
        with tab3: self.add_ai_review(category, product_name)
        with tab4: self.add_hijacked_review(category, product_name)

    def add_human_review(self, category: str, product_name: str):
        st.write("Add authentic human reviews")
        with st.form("human_review_form"):
            review_text = st.text_area("Review Text:", height=100, placeholder="Enter a realistic human review...")
            col1, col2 = st.columns(2)
            with col1: rating = st.slider("Rating:", 1, 5, 4)
            with col2: typing_time = st.number_input("Typing Time (seconds):", min_value=0.0, max_value=500.0, value=random.uniform(250, 400), step=10.0, format="%.2f")
            if st.form_submit_button("Add Human Review", type="primary"):
                if review_text:
                    features = self.feature_gen.generate_features(product_name, review_text, "Human", rating, typing_time)
                    file_path = self.data_manager.save_review(features, category, self.config.CATEGORY_FILE_MAP)
                    st.success(f"✅ Human review added to {file_path}")
                else: st.error("Please enter review text")

    def add_script_review(self, category: str, product_name: str):
        st.write("Generate script-based reviews (often short, generic, and extreme ratings)")
        col1, col2 = st.columns(2)
        with col1: count = st.number_input("Number of reviews:", 1, 20, 5)
        with col2: use_template = st.checkbox("Use predefined templates", value=True)

        if st.button("Generate Script Reviews"):
            reviews_to_add = []
            if use_template:
                # Ensure we don't try to sample more than available templates
                reviews_to_add = random.sample(self.config.SCRIPT_TEMPLATES, min(int(count), len(self.config.SCRIPT_TEMPLATES)))
            # If count is higher than templates, or not using templates, generate generic ones
            if len(reviews_to_add) < int(count):
                for i in range(int(count) - len(reviews_to_add)):
                    reviews_to_add.append(f"Generic scripted review for {product_name} #{i+1}")

            if reviews_to_add:
                progress_bar = st.progress(0)
                for i, review in enumerate(reviews_to_add):
                    features = self.feature_gen.generate_features(product_name, review, "Script")
                    self.data_manager.save_review(features, category, self.config.CATEGORY_FILE_MAP)
                    progress_bar.progress((i + 1) / len(reviews_to_add))
                st.success(f"✅ Added {len(reviews_to_add)} script reviews")
            else:
                st.warning("No reviews generated. Adjust count or templates.")

    def add_ai_review(self, category: str, product_name: str):
        st.write("Generate AI-powered reviews using Gemini (reviews often sound too perfect or generic)")
        count = st.number_input("Number of AI reviews:", 1, 15, 5)

        if st.button("Generate AI Reviews"):
            with st.spinner("Generating AI reviews..."):
                reviews = self.generator.generate_ai_reviews(product_name, int(count))
                if reviews:
                    progress_bar = st.progress(0)
                    for i, review in enumerate(reviews):
                        features = self.feature_gen.generate_features(product_name, review, "AI")
                        self.data_manager.save_review(features, category, self.config.CATEGORY_FILE_MAP)
                        progress_bar.progress((i + 1) / len(reviews))
                    st.success(f"✅ Added {len(reviews)} AI reviews")
                    with st.expander("Preview Generated Reviews"):
                        for i, review in enumerate(reviews[:min(5, len(reviews))]): # Show max 5 reviews
                            st.write(f"**{i+1}.** {review}")
                else: st.error("Failed to generate AI reviews. Check your API key or try again.")

    def add_hijacked_review(self, category: str, product_name: str):
        st.write("Generate hijacked/misleading reviews (authentic-looking but with deceptive content)")
        intent = st.text_input("Misleading Intent/Specific details to include:", placeholder="e.g., Promote competitor's battery life, complain about an old feature...")
        count = st.number_input("Number of hijacked reviews:", 1, 10, 3)

        if st.button("Generate Hijacked Reviews"):
            if intent:
                with st.spinner("Generating hijacked reviews..."):
                    reviews = self.generator.generate_hijacked_reviews(product_name, intent, int(count))
                    if reviews:
                        progress_bar = st.progress(0)
                        for i, review in enumerate(reviews):
                            features = self.feature_gen.generate_features(product_name, review, "Hijacked")
                            self.data_manager.save_review(features, category, self.config.CATEGORY_FILE_MAP)
                            progress_bar.progress((i + 1) / len(reviews))
                        st.success(f"✅ Added {len(reviews)} hijacked reviews")
                        with st.expander("Preview Generated Reviews"):
                            for i, review in enumerate(reviews[:min(5, len(reviews))]): # Show max 5 reviews
                                st.write(f"**{i+1}.** {review}")
                    else: st.error("Failed to generate hijacked reviews. Check your API key or try again.")
            else: st.error("Please specify the misleading intent.")

    def ml_predictions_page(self):
        st.header("📊 ML Predictions")
        if not st.session_state.ml_trained:
            st.warning("⚠️ ML models are not trained yet. Please train models first from '🏋️ Train ML Models' page.")
            return

        # Attempt to load models if they are not already loaded (e.g., app rerun)
        if not self.ml_classifier.models_trained:
            if not self.ml_classifier.load_models():
                st.error("❌ Failed to load trained models. Please train them again.")
                return

        category = st.selectbox("Select Category for Prediction:", list(self.config.CATEGORY_FILE_MAP.keys()))
        df = self.data_manager.load_reviews(category, self.config.CATEGORY_FILE_MAP)

        if df is None or len(df) == 0:
            st.info(f"No reviews found for {category}. Please add some reviews first using '📝 Add Reviews' tab.")
            return

        st.write(f"Found {len(df)} reviews for prediction in **{category}**.")
        col1, col2 = st.columns(2)
        with col1: predict_button_clicked = st.button("🔍 Predict All Reviews", type="primary")
        with col2: sample_size = st.number_input("Sample Size (0 = all):", 0, len(df), 0, help="Enter 0 to predict all reviews in the selected category.")

        if predict_button_clicked:
            if df.empty:
                st.warning("No reviews available to predict. Add reviews first!")
                return

            with st.spinner("Making ML predictions... This might take a moment for large datasets."):
                df_predict = df.sample(n=min(int(sample_size) if sample_size > 0 else len(df), len(df)), random_state=42).copy()
                results = self.ml_classifier.predict(df_predict)

                if results is not None:
                    st.success(f"✅ Predictions completed for {len(results)} reviews.")
                    self.display_prediction_results(results)
                    safe_category_name = category.lower().replace(' ', '_').replace('(', '').replace(')', '')
                    results_file = os.path.join(self.config.DATA_FOLDER, f"predictions_{safe_category_name}.xlsx")
                    results.to_excel(results_file, index=False)
                    st.info(f"Prediction results saved to: `{results_file}`")
                else: st.error("❌ Prediction failed. Ensure your models are trained and the 'ml_classifier.py' works correctly.")

    def display_prediction_results(self, results: pd.DataFrame):
        st.subheader("🎯 Prediction Results Summary")
        if 'final_prediction' not in results.columns:
            st.error("Prediction results do not contain 'final_prediction' column. Cannot display statistics.")
            return

        # Ensure all possible labels are accounted for even if zero
        all_possible_labels = ['Human', 'Script', 'AI', 'Hijacked']
        predicted_counts = results['final_prediction'].value_counts().reindex(all_possible_labels, fill_value=0)

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Human Reviews", predicted_counts.get('Human', 0))
        with col2: st.metric("Script Reviews", predicted_counts.get('Script', 0))
        with col3: st.metric("AI Reviews", predicted_counts.get('AI', 0))
        with col4: st.metric("Hijacked Reviews", predicted_counts.get('Hijacked', 0))

        fig = px.pie(values=predicted_counts.values, names=predicted_counts.index, title="Review Type Distribution - ML Predictions")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🎯 Confidence Analysis")
        if 'confidence' not in results.columns:
            st.warning("No 'confidence' column in prediction results. Cannot display confidence analysis.")
        else:
            if not results['confidence'].empty:
                st.metric("Average Confidence", f"{results['confidence'].mean():.2%}")
                fig2 = px.histogram(results, x='confidence', nbins=20, title="Confidence Score Distribution")
                fig2.update_layout(xaxis_title="Confidence Score", yaxis_title="Count")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No confidence data available for analysis.")


        st.subheader("📋 Detailed Results")
        col1, col2 = st.columns(2)
        with col1:
            unique_predictions = ['All'] + sorted(results['final_prediction'].unique().tolist())
            filter_type = st.selectbox("Filter by Predicted Type:", unique_predictions)
        with col2:
            min_confidence = st.slider("Minimum Confidence:", 0.0, 1.0, 0.0, 0.05)

        filtered = results.copy()
        if filter_type != 'All': filtered = filtered[filtered['final_prediction'] == filter_type]
        if 'confidence' in filtered.columns: filtered = filtered[filtered['confidence'] >= min_confidence]

        display_cols = ['review', 'label', 'final_prediction', 'confidence']
        # Dynamically add probability columns if they exist
        for prob_col_suffix in ['script', 'ai', 'hijacked', 'human']:
            prob_col_name = f'{prob_col_suffix}_probability'
            if prob_col_name in filtered.columns:
                display_cols.append(prob_col_name)

        display_cols_existing = [col for col in display_cols if col in filtered.columns]
        st.dataframe(filtered[display_cols_existing].sort_values(by='confidence', ascending=False), use_container_width=True)

        if 'label' in results.columns and not results['label'].isnull().all():
            st.subheader("📈 Model Performance (Against Original Labels)")
            self.calculate_model_accuracy(results)
        else:
            st.info("No original 'label' column found or all labels are missing. Cannot calculate model performance metrics.")

    def calculate_model_accuracy(self, results: pd.DataFrame):
        # Extract the base label (e.g., "Human" from "Human Positive")
        results['true_label'] = results['label'].apply(lambda x: x.split(' ')[0].strip() if isinstance(x, str) else 'Unknown')

        # Filter out rows where true_label is 'Unknown' or prediction is missing
        valid_predictions = results[(results['true_label'] != 'Unknown') & results['final_prediction'].notna()].copy()

        if len(valid_predictions) == 0:
            st.info("No valid labeled data with predictions to calculate accuracy.")
            return

        # Ensure both true and predicted labels have common categories for confusion matrix
        # Create a union of all unique true and predicted labels
        all_labels_union = sorted(list(set(valid_predictions['true_label'].unique()) | set(valid_predictions['final_prediction'].unique())))

        if not all_labels_union: # If still no labels, return
            st.info("No common labels found for performance calculation.")
            return

        correct = (valid_predictions['final_prediction'] == valid_predictions['true_label']).sum()
        total = len(valid_predictions)
        accuracy = correct / total if total > 0 else 0

        col1, col2 = st.columns(2)
        with col1: st.metric("Overall Accuracy", f"{accuracy:.2%}")
        with col2: st.metric("Correct Predictions", f"{correct}/{total}")

        st.markdown("---")
        st.subheader("Confusion Matrix")
        # Generate confusion matrix
        cm = confusion_matrix(valid_predictions['true_label'], valid_predictions['final_prediction'], labels=all_labels_union)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_labels_union, yticklabels=all_labels_union, ax=ax)
        ax.set_title('Confusion Matrix: Actual vs. Predicted')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('Actual Label')
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("Classification Report (Per Label Metrics)")
        # Calculate precision, recall, f1-score for each class
        # You might need to import these from sklearn.metrics at the top if you want
        # them available more broadly, or just here.
        from sklearn.metrics import classification_report
        try:
            report_dict = classification_report(valid_predictions['true_label'], valid_predictions['final_prediction'],
                                                labels=all_labels_union, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.round(2), use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate classification report: {e}")
            st.info("Ensure you have at least one sample for each relevant label in your prediction set.")


    def dashboard_page(self):
        st.header("📈 Results Dashboard")
        all_data = []
        categories = []

        for category in self.config.CATEGORY_FILE_MAP:
            df = self.data_manager.load_reviews(category, self.config.CATEGORY_FILE_MAP)
            if df is not None and not df.empty:
                df['category'] = category
                all_data.append(df)
                categories.append(category)

        if not all_data:
            st.info("No data available across all categories. Please add some reviews first using the '📝 Add Reviews' tab.")
            return

        combined_df = pd.concat(all_data, ignore_index=True)

        st.subheader("📊 Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Reviews", len(combined_df))
        with col2: st.metric("Categories", len(categories))
        with col3: st.metric("Products", combined_df['product_name'].nunique())
        with col4:
            combined_df['timestamp_dt'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
            date_range_min = combined_df['timestamp_dt'].min()
            date_range_max = combined_df['timestamp_dt'].max()
            if pd.notna(date_range_min) and pd.notna(date_range_max):
                st.metric("Data Span (Days)", (date_range_max - date_range_min).days)
            else:
                st.metric("Data Span (Days)", "N/A")


        col1, col2 = st.columns(2)
        with col1:
            label_counts = combined_df['label'].value_counts()
            fig1 = px.bar(x=label_counts.index, y=label_counts.values, title="Review Type Distribution (Original Labels)")
            fig1.update_layout(xaxis_title="Review Type", yaxis_title="Count")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            category_counts = combined_df['category'].value_counts()
            fig2 = px.pie(values=category_counts.values, names=category_counts.index, title="Reviews by Category")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📅 Timeline Analysis")
        combined_df_valid_dates = combined_df.dropna(subset=['timestamp_dt'])
        if not combined_df_valid_dates.empty:
            # Group by date and label for detailed timeline
            daily_counts = combined_df_valid_dates.groupby([combined_df_valid_dates['timestamp_dt'].dt.date, 'label']).size().reset_index(name='count')
            daily_counts.rename(columns={'timestamp_dt': 'date'}, inplace=True) # Rename column after groupby
            fig3 = px.line(daily_counts, x='date', y='count', color='label', title="Review Submission Timeline by Type")
            fig3.update_layout(xaxis_title="Date", yaxis_title="Number of Reviews")
            st.plotly_chart(fig3, use_container_width=True)
        else: st.info("No valid date data to show timeline analysis.")

        st.subheader("🔍 Feature Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig4 = px.histogram(combined_df, x='stars', color='label', barmode='group', title="Rating Distribution by Review Type")
            fig4.update_layout(xaxis_title="Stars", yaxis_title="Count")
            st.plotly_chart(fig4, use_container_width=True)
        with col2:
            # Filter out typing_time_seconds = 0 as it skews the plot for AI reviews
            filtered_df_for_typing = combined_df[combined_df['typing_time_seconds'] > 0]
            if not filtered_df_for_typing.empty:
                fig5 = px.box(filtered_df_for_typing, x='label', y='typing_time_seconds', title="Typing Time Distribution by Review Type (Excluding 0s)")
                fig5.update_layout(xaxis_title="Review Type", yaxis_title="Typing Time (seconds)")
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.info("No typing time data available for analysis (or all are zero).")

        st.subheader("💾 Export Data")
        col1, col2 = st.columns(2)
        with col1:
            csv = combined_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download All Data (CSV)", data=csv, file_name=f"all_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
        with col2:
            if st.button("📊 Export Summary Report (JSON)"):
                # Re-calculate counts just in case
                label_counts = combined_df['label'].value_counts()
                category_counts = combined_df['category'].value_counts()
                date_range_str = "N/A"
                if pd.notna(date_range_min) and pd.notna(date_range_max):
                    date_range_str = f"{date_range_min.strftime('%Y-%m-%d')} to {date_range_max.strftime('%Y-%m-%d')}"

                summary = {
                    'total_reviews': len(combined_df),
                    'categories': len(categories),
                    'label_distribution': label_counts.to_dict(),
                    'category_distribution': category_counts.to_dict(),
                    'date_range': date_range_str
                }
                st.json(summary) # Display JSON on page first
                # Also provide a download button for the JSON
                json_data = json.dumps(summary, indent=4).encode('utf-8')
                st.download_button(label="Download Summary JSON", data=json_data, file_name=f"review_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")


    def run(self):
        action = self.setup_page()
        if action == "🏋️ Train ML Models": self.train_models_page()
        elif action == "📝 Add Reviews": self.add_reviews_page()
        elif action == "📊 ML Predictions": self.ml_predictions_page()
        elif action == "📈 Results Dashboard": self.dashboard_page()

# Main execution entry point for Streamlit
if __name__ == "__main__":
    # Ensure data folder exists
    if not os.path.exists(Config.DATA_FOLDER):
        os.makedirs(Config.DATA_FOLDER)
    app = MLIntegratedApp()
    app.run()