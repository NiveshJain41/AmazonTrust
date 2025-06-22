# This is app.ui.py
import streamlit as st
import pandas as pd
import os
from typing import Optional, List
from datetime import datetime
import random

from core_logic import MLIntegratedAppCore, Config

class AppUI:
    def __init__(self, core_app: MLIntegratedAppCore):
        self.core_app = core_app
        if 'ml_trained_ui_status' not in st.session_state:
            st.session_state.ml_trained_ui_status = self.core_app.is_ml_trained
        # No need to store metrics in session_state, as they can be fetched from core_app.model_metrics

    def render_train_models_section(self):
        st.header("🏋️ Train ML Models")
        st.write("Train machine learning models to detect fake reviews.")
        training_file = os.path.join(self.core_app.config.DATA_FOLDER, self.core_app.config.TRAINING_DATA_FILE)

        training_df = None
        if os.path.exists(training_file):
            try:
                training_df = pd.read_excel(training_file)
            except Exception as e:
                st.error(f"Error loading training file: {e}")

        if training_df is None or training_df.empty:
            st.info("No training data found. Attempting to create from existing reviews if available...")
            if st.button("Attempt to Create Training Data"):
                if self.core_app.train_models_logic():
                    st.success("Training data prepared! Please click 'Start Training' again.")
                    st.session_state.ml_trained_ui_status = self.core_app.is_ml_trained
                    st.rerun()
                else:
                    st.warning("No reviews available to create training data. Add reviews first.")
            return

        st.write(f"**Total samples in training data:** {len(training_df)}")
        st.write(f"**Models Trained:** {'Yes' if st.session_state.ml_trained_ui_status else 'No'}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Start Training"):
                with st.spinner("Training models..."):
                    if self.core_app.train_models_logic():
                        st.session_state.ml_trained_ui_status = True
                        st.success("Models trained successfully!")
                    else:
                        st.error("Training failed.")
        with col2:
            if st.button("📂 Load Pre-trained Models"):
                with st.spinner("Loading models..."):
                    if self.core_app.load_trained_models_logic():
                        st.session_state.ml_trained_ui_status = True
                        st.success("Pre-trained models loaded!")
                    else:
                        st.error("Model loading failed.")

        # ADDED: Display ML Model Metrics if available
        # if st.session_state.ml_trained_ui_status:
            # metrics = self.core_app.model_metrics
            # if metrics:
            #     st.subheader("Model Performance Metrics (Simulated)")
            #     metric_cols = st.columns(len(metrics))
            #     for i, (metric_name, value) in enumerate(metrics.items()):
            #         with metric_cols[i]:
            #             st.metric(label=metric_name.replace('_', ' ').title(), value=f"{value:.2f}")
            #     st.info("Note: These metrics are simulated as the ML model is a placeholder.")
            # else:
            #     st.warning("No metrics available. Train or load models to see performance.")


    def render_add_reviews_section(self):
        st.header("📝 Add Reviews to Dataset")
        category = st.selectbox("Select Category for New Reviews:", list(self.core_app.config.CATEGORY_FILE_MAP.keys()))
        product_name = self.core_app.config.PRODUCT_NAME_MAP.get(category, "Generic Product")

        review_type = st.radio("Choose Review Type:", ["Human", "Script", "AI", "Hijacked"])

        if review_type == "Human":
            review_text = st.text_area("Enter Human Review:", height=100)
            rating = st.slider("Rating:", 1, 5, 4)
            typing_time = st.number_input("Typing Time (seconds):", value=random.uniform(250, 400), format="%.2f")
            if st.button("Add Human Review"):
                if review_text:
                    if self.core_app.add_review_to_dataset(category, product_name, review_text, "Human", rating, typing_time):
                        st.success("Review added.")
                    else:
                        st.error("Error adding review.")
                else:
                    st.warning("Review text required.")
        elif review_type == "Script":
            count = st.number_input("Number of Script Reviews:", 1, 20, 5)
            if st.button("Generate Script Reviews"):
                with st.spinner("Generating script reviews..."):
                    added = self.core_app.generate_and_add_script_reviews(category, product_name, int(count))
                    if added:
                        st.success(f"{len(added)} script reviews added.")
        elif review_type == "AI":
            count = st.number_input("Number of AI Reviews:", 1, 15, 5)
            if st.button("Generate AI Reviews"):
                with st.spinner("Generating AI reviews..."):
                    added = self.core_app.generate_and_add_ai_reviews(category, product_name, int(count))
                    if added:
                        st.success(f"{len(added)} AI reviews added.")
        elif review_type == "Hijacked":
            intent = st.text_input("Intent for Hijacked Review:")
            count = st.number_input("Number of Hijacked Reviews:", 1, 10, 3)
            if st.button("Generate Hijacked Reviews"):
                if intent:
                    with st.spinner("Generating hijacked reviews..."):
                        added = self.core_app.generate_and_add_hijacked_reviews(category, product_name, intent, int(count))
                        if added:
                            st.success(f"{len(added)} hijacked reviews added.")
                else:
                    st.warning("Please enter the misleading intent.")

    def render_ml_predictions_section(self):
        st.header("📊 ML Predictions")
        if not st.session_state.ml_trained_ui_status:
            st.warning("⚠️ Train the models first.")
            return

        category = st.selectbox("Select Category for Prediction:", list(self.core_app.config.CATEGORY_FILE_MAP.keys()))
        df = self.core_app.get_reviews_for_prediction(category)

        if df is None or df.empty:
            st.info("No reviews found for prediction.")
            return

        if st.button("🔍 Predict Reviews"):
            with st.spinner("Predicting..."):
                results = self.core_app.make_predictions(df, category)
                if results is not None:
                    st.success("Prediction complete!")
                    st.subheader("Top 10 Predicted Reviews")

                    show_cols = ['review', 'label', 'final_prediction', 'confidence']
                    display_cols = [col for col in show_cols if col in results.columns]

                    if display_cols:
                        st.dataframe(results[display_cols].head(10), use_container_width=True)
                    else:
                        st.warning("No relevant columns to display.")

                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions (CSV)",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Prediction failed.")

    def run(self):
        st.set_page_config(page_title="Fake Review Detection", layout="wide")
        st.title("🤖 Amazon Fake Review Detection System")
        st.markdown("Generate reviews, train models, predict fake reviews — all in one place.")
        st.divider()
        self.render_train_models_section()
        st.divider()
        self.render_add_reviews_section()
        st.divider()
        self.render_ml_predictions_section()