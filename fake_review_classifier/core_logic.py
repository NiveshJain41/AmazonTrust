# core_logic.py
import logging
import pandas as pd
import os
import time
import random
import json
from datetime import datetime
from typing import Dict, List, Optional
import google.generativeai as genai
from dataclasses import dataclass
import uuid

# Set up a basic logger for backend operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Placeholder for ReviewClassifier (actual model used in production)
from ml_classifier import ReviewClassifier


# --- Configuration and Constants --- (FIXED)
class Config:
    # BASE_DIR is the directory where this script is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Data folder name and full path
    DATA_FOLDER_NAME = "data"
    DATA_FOLDER = os.path.join(BASE_DIR, DATA_FOLDER_NAME)  # Add this line
    
    # Models folder path
    MODELS_FOLDER = os.path.join(BASE_DIR, "models")  # Add this line

    #ENTER YOUR GEMINI KEY HERE
    GEMINI_API_KEY = "Enter Your Gemini Key Here"
    MODEL_NAME = "gemini-2.0-flash"

    CATEGORY_FILE_MAP = {
        "Product (Vivo Mobile)": "product_reviews.xlsx",
        "Seller (Nivesh and Yash Clothing)": "seller_reviews.xlsx",
        "Amazon Service (Prime Video)": "amazon_service_reviews.xlsx"
    }

    PRODUCT_NAME_MAP = {
        "Product (Vivo Mobile)": "Vivo Mobile Phone",
        "Seller (Nivesh and Yash Clothing)": "Nivesh and Yash Cotton Shirt",
        "Amazon Service (Prime Video)": "Amazon Prime Video Service"
    }

    TRAINING_DATA_FILE = "training_data.xlsx"

    SCRIPT_TEMPLATES = [
        "Amazing product!", "Best value!", "Very satisfied.", "Affordable and reliable.",
        "Top-notch quality.", "Excellent quality.", "Perfect for daily use.",
        "Outstanding performance.", "Superb build.", "Fantastic product."
    ]

# --- Data Models ---
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

# --- Review Generation Logic ---
class ReviewGenerator:
    def __init__(self, api_key: str, model_name: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def _generate_content(self, prompt: str, count: int) -> List[str]:
        try:
            logging.info(f"Generating {count} reviews with Gemini model...")
            response = self.model.generate_content(prompt)
            reviews = [r.strip().replace('"', '').replace("'", '') for r in response.text.split("\n") if r.strip()]
            logging.info(f"Successfully generated {len(reviews)} reviews.")
            return reviews[:count]
        except Exception as e:
            logging.error(f"Error generating reviews: {str(e)}")
            return []

    def generate_ai_reviews(self, product_name: str, count: int = 5) -> List[str]:
        prompt = f"Generate {count} product reviews for '{product_name}'. Vary length and tone. Format: One review per line."
        return self._generate_content(prompt, count)

    def generate_hijacked_reviews(self, product_name: str, intent: str, count: int = 3) -> List[str]:
        prompt = f"Generate {count} misleading reviews for '{product_name}' based on: {intent}. Make them sound authentic."
        return self._generate_content(prompt, count)

# --- Feature Generation Logic ---
class ReviewFeatureGenerator:
    SHARED_SCRIPT_IP = "192.168.10.100"
    SHARED_AI_IP = "10.0.0.50"

    @staticmethod
    def _generate_ip() -> str:
        return f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"

    @staticmethod
    def generate_features(product_name: str, review: str, label: str,
                            rating: Optional[int] = None, typing_time: Optional[float] = None) -> ReviewFeatures:
        features = ReviewFeatures(
            product_name=product_name, review=review, label=label,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            review_id=str(uuid.uuid4())[:8],
            account_id=f"A{random.randint(10**11, 10**12-1)}"
        )

        if "Human" in label:
            features.stars = rating or random.randint(3, 5)
            features.typing_time_seconds = typing_time or random.uniform(250, 400)
            features.bought = "Yes"
            features.ip_address = ReviewFeatureGenerator._generate_ip()
            features.product_specific = "Yes"
            features.helpful_votes_percent = f"{random.randint(40, 85)}%"
            features.device_fingerprint = "Unique"
        elif "Script" in label:
            features.stars = random.choice([1, 5])
            features.typing_time_seconds = random.uniform(5, 15)
            features.ip_address = ReviewFeatureGenerator.SHARED_SCRIPT_IP
            features.product_specific = "No"
            features.helpful_votes_exists = "No"
            features.helpful_votes_percent = "0%"
            features.bought = "No"
            features.device_fingerprint = "Reused"
        elif "AI" in label:
            features.stars = random.choice([1, 5])
            features.typing_time_seconds = 0
            features.ip_address = ReviewFeatureGenerator.SHARED_AI_IP
            features.helpful_votes_percent = f"{random.randint(5, 15)}%"
            features.bought = "No"
            features.product_specific = "No"
            features.helpful_votes_exists = "No"
            features.device_fingerprint = "Reused"
        elif "Hijacked" in label:
            features.stars = random.randint(4, 5)
            features.typing_time_seconds = random.uniform(250, 400)
            features.bought = "Yes"
            features.ip_address = ReviewFeatureGenerator._generate_ip()
            features.product_specific = "Misleading"
            features.helpful_votes_exists = "Yes"
            features.helpful_votes_percent = f"{random.randint(20, 60)}%"
            features.device_fingerprint = "Unique"
        return features

# --- Data Management Logic ---
class DataManager:
    def __init__(self, base_dir: str, data_folder_name: str):
        # Construct the full path to the data folder relative to base_dir
        self.data_folder = os.path.join(base_dir, data_folder_name)
        os.makedirs(self.data_folder, exist_ok=True)
        logging.info(f"DataManager initialized. Data folder: {self.data_folder}")

    def save_review(self, features: ReviewFeatures, category: str, file_map: Dict[str, str]):
        file_path = os.path.join(self.data_folder, file_map[category])
        df_entry = pd.DataFrame([features.__dict__])
        # Using a context manager for ExcelFile to prevent resource leaks
        if os.path.exists(file_path):
            with pd.ExcelFile(file_path) as xls:
                # Check if the Excel file is empty or has sheets
                if not xls.sheet_names:
                    combined = pd.DataFrame() # Create empty DataFrame if no sheets
                else:
                    try:
                        combined = pd.read_excel(xls)
                    except Exception as e:
                        logging.warning(f"Could not read existing Excel file {file_path}, creating new one. Error: {e}")
                        combined = pd.DataFrame()
        else:
            combined = pd.DataFrame()

        combined = pd.concat([combined, df_entry], ignore_index=True)
        combined.to_excel(file_path, index=False)
        logging.info(f"Review saved to: {file_path}")
        return file_path

    def load_reviews(self, category: str, file_map: Dict[str, str]) -> Optional[pd.DataFrame]:
        file_path = os.path.join(self.data_folder, file_map[category])
        if os.path.exists(file_path):
            try:
                df = pd.read_excel(file_path)
                logging.info(f"Loaded {len(df)} reviews from {file_path}")
                return df
            except Exception as e:
                logging.error(f"Error loading {file_path}: {e}")
                return None
        logging.info(f"No file found at: {file_path}")
        return None

    def get_all_reviews(self) -> pd.DataFrame:
        all_data = []
        for category in Config.CATEGORY_FILE_MAP:
            df = self.load_reviews(category, Config.CATEGORY_FILE_MAP)
            if df is not None and not df.empty:
                df['category'] = category
                all_data.append(df)
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

class MLIntegratedAppCore:
    def __init__(self):
        self.config = Config()
        
        # Ensure data and models folders exist
        os.makedirs(self.config.DATA_FOLDER, exist_ok=True)
        os.makedirs(self.config.MODELS_FOLDER, exist_ok=True)
        
        self.generator = ReviewGenerator(self.config.GEMINI_API_KEY, self.config.MODEL_NAME)
        self.feature_gen = ReviewFeatureGenerator()
        
        # Pass the correct paths to DataManager
        self.data_manager = DataManager(self.config.BASE_DIR, self.config.DATA_FOLDER_NAME)
        
        # Use the MODELS_FOLDER path for ML classifier
        self.ml_classifier = ReviewClassifier(model_dir=self.config.MODELS_FOLDER)

        self._ml_trained = False

        if not self.ml_classifier.models_trained:
            if self.ml_classifier.load_models():
                self._ml_trained = True

    @property 
    def is_ml_trained(self) -> bool:
        return self._ml_trained

    def train_models_logic(self) -> bool:
        # Use the data_manager's known data_folder for training_file path
        training_file = os.path.join(self.data_manager.data_folder, self.config.TRAINING_DATA_FILE)

        if not os.path.exists(training_file):
            logging.info("Training data file not found. Attempting to create from existing reviews.")
            all_data = self.data_manager.get_all_reviews()
            if not all_data.empty:
                if 'label' in all_data.columns:
                    all_data['label'] = all_data['label'].apply(lambda x: x.split(' ')[0].strip() if isinstance(x, str) else x)
                else:
                    logging.warning("No 'label' column found in review data. Cannot create training data.")
                    return False
                all_data.to_excel(training_file, index=False)
                logging.info(f"Training data created with {len(all_data)} samples.")
            else:
                logging.warning("No reviews available to create training data.")
                return False

        if self.ml_classifier.train_models(training_file):
            self._ml_trained = True
            return True
        else:
            self._ml_trained = False
            return False

    def load_trained_models_logic(self) -> bool:
        if self.ml_classifier.load_models():
            self._ml_trained = True
            return True
        else:
            self._ml_trained = False
            return False

    def add_review_to_dataset(self, category: str, product_name: str, review_text: str, label: str,
                              rating: Optional[int] = None, typing_time: Optional[float] = None) -> bool:
        try:
            features = self.feature_gen.generate_features(product_name, review_text, label, rating, typing_time)
            self.data_manager.save_review(features, category, self.config.CATEGORY_FILE_MAP)
            return True
        except Exception as e:
            logging.error(f"Error adding review: {e}")
            return False

    def generate_and_add_script_reviews(self, category: str, product_name: str, count: int = 5) -> List[str]:
        reviews_to_add = random.sample(self.config.SCRIPT_TEMPLATES, min(count, len(self.config.SCRIPT_TEMPLATES)))
        if len(reviews_to_add) < count:
            for i in range(count - len(reviews_to_add)):
                reviews_to_add.append(f"Generic scripted review for {product_name} #{i+1}")

        added_reviews_texts = []
        for review_text in reviews_to_add:
            if self.add_review_to_dataset(category, product_name, review_text, "Script"):
                added_reviews_texts.append(review_text)
        return added_reviews_texts

    def generate_and_add_ai_reviews(self, category: str, product_name: str, count: int = 5) -> List[str]:
        reviews = self.generator.generate_ai_reviews(product_name, count)
        added_reviews_texts = []
        for review_text in reviews:
            if self.add_review_to_dataset(category, product_name, review_text, "AI"):
                added_reviews_texts.append(review_text)
        return added_reviews_texts

    def generate_and_add_hijacked_reviews(self, category: str, product_name: str, intent: str, count: int = 3) -> List[str]:
        reviews = self.generator.generate_hijacked_reviews(product_name, intent, count)
        added_reviews_texts = []
        for review_text in reviews:
            if self.add_review_to_dataset(category, product_name, review_text, "Hijacked"):
                added_reviews_texts.append(review_text)
        return added_reviews_texts

    def get_reviews_for_prediction(self, category: str) -> Optional[pd.DataFrame]:
        return self.data_manager.load_reviews(category, self.config.CATEGORY_FILE_MAP)

    def make_predictions(self, df: pd.DataFrame, category: str) -> Optional[pd.DataFrame]:
        if not self.is_ml_trained:
            logging.warning("Models not trained, cannot predict.")
            return None
        results = self.ml_classifier.predict(df)
        if results is not None:
            safe_category_name = category.lower().replace(' ', '_').replace('(', '').replace(')', '')
            # Use the data_manager's full data_folder path
            results_file = os.path.join(self.data_manager.data_folder, f"predictions_{safe_category_name}.xlsx")
            results.to_excel(results_file, index=False)
            logging.info(f"Prediction results saved to: {results_file}")
        return results

    def get_dashboard_data(self) -> pd.DataFrame:
        return self.data_manager.get_all_reviews()

# Global check for data folder creation:
# This ensures the 'data' folder is created relative to the script's location
# if core_logic.py is the primary entry point, or if this block is executed.
# It now uses Config.BASE_DIR to ensure consistency.
full_data_path = os.path.join(Config.BASE_DIR, Config.DATA_FOLDER_NAME)
if not os.path.exists(full_data_path):
    os.makedirs(full_data_path)
