# This is ml_classifier.py (OPTIMIZED - Reduced Computation)
import pandas as pd
import numpy as np
import re
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

class ReviewClassifier:
    # def __init__(self, model_dir="models"):
    #     self.model_dir = model_dir
    #     os.makedirs(model_dir, exist_ok=True)
        
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        # Initialize models
        self.script_model = None
        self.ai_model = None
        self.hijacked_model = None
        self.feature_scaler = None
        self.tfidf_vectorizer = None
        
        self.models_trained = False
        
        # Simplified product keywords for relevance scoring
        self.product_keywords = {
            "vivo mobile phone": ["phone", "mobile", "camera", "battery", "android", "screen"],
            "cotton shirt": ["shirt", "cotton", "fabric", "clothing", "fit", "size"],
            "amazon prime video": ["video", "streaming", "movie", "show", "prime", "amazon"]
        }
    
    def calculate_simple_relevance_score(self, review_text: str, product_name: str) -> float:
        """Fast product relevance calculation using basic keyword matching"""
        review_lower = review_text.lower()
        product_lower = product_name.lower()
        
        # Get relevant keywords for the product
        keywords = self.product_keywords.get(product_lower, [])
        
        if not keywords:
            return 0.5  # Default score if product not found
        
        # Count keyword matches
        matches = sum(1 for keyword in keywords if keyword in review_lower)
        
        # Calculate relevance score (0 to 1)
        relevance_score = min(matches / len(keywords), 1.0)
        
        # Boost score if product name is mentioned
        if any(word in review_lower for word in product_lower.split()):
            relevance_score = min(relevance_score + 0.3, 1.0)
        
        return relevance_score
    
    def count_simple_text_features(self, text: str) -> Dict[str, int]:
        """Fast text feature extraction without heavy NLP libraries"""
        features = {}
        
        # Basic character and word counts
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
        
        # Special character count (simplified)
        features['special_char_count'] = len(re.findall(r'[^a-zA-Z0-9\s]', text))
        
        # Exclamation and caps patterns (fake review indicators)
        features['exclamation_count'] = text.count('!')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        
        # Repetition patterns
        words = text.lower().split()
        features['repeated_words'] = len(words) - len(set(words))
        
        return features
    
    def extract_fast_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features quickly using vectorized operations where possible"""
        df = df.copy()
        
        print("Extracting features (fast method)...")
        
        # Calculate average rating for deviation
        average_rating = df['stars'].mean()
        
        # Vectorized text feature extraction
        df['review_length'] = df['review'].str.len()
        df['word_count'] = df['review'].str.split().str.len()
        df['exclamation_count'] = df['review'].str.count('!')
        df['caps_ratio'] = df['review'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)
        
        # Product relevance (vectorized where possible)
        df['product_relevance_score'] = df.apply(
            lambda row: self.calculate_simple_relevance_score(row['review'], row['product_name']), 
            axis=1
        )
        
        # Rating deviation
        df['rating_deviation'] = np.abs(df['stars'] - average_rating)
        
        # Simplified suspicious patterns
        df['extreme_rating'] = ((df['stars'] == 1) | (df['stars'] == 5)).astype(int)
        df['short_review'] = (df['review_length'] < 50).astype(int)
        df['very_fast_typing'] = (df['typing_time_seconds'] < 30).astype(int)
        
        print("Fast feature extraction completed!")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for ML models with reduced complexity"""
        # Reduced TF-IDF features (from 1000 to 200 for speed)
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=200,  # Reduced from 1000
                stop_words='english',
                ngram_range=(1, 2),  # Include bigrams for better context
                min_df=2  # Ignore rare terms
            )
            text_features = self.tfidf_vectorizer.fit_transform(df['review']).toarray()
        else:
            text_features = self.tfidf_vectorizer.transform(df['review']).toarray()
        
        # Essential numerical features only
        numerical_features = [
            'stars', 'typing_time_seconds', 'product_relevance_score',
            'rating_deviation', 'review_length', 'word_count', 
            'exclamation_count', 'caps_ratio'
        ]
        
        # Essential categorical features (encoded)
        df['bought_encoded'] = df['bought'].map({'Yes': 1, 'No': 0})
        df['product_specific_encoded'] = df['product_specific'].map({'Yes': 1, 'No': 0, 'Misleading': -1})
        df['device_fingerprint_encoded'] = df['device_fingerprint'].map({'Unique': 1, 'Reused': 0})
        
        # Convert helpful votes percentage to float
        df['helpful_votes_percent_num'] = df['helpful_votes_percent'].str.rstrip('%').astype(float) / 100
        
        # Simplified temporal features
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
        df['days_old'] = (datetime.now() - df['timestamp_dt']).dt.days
        
        # Fast IP analysis
        ip_counts = df['ip_address'].value_counts()
        df['shared_ip'] = df['ip_address'].map(lambda x: 1 if ip_counts[x] > 1 else 0)
        
        categorical_features = [
            'bought_encoded', 'product_specific_encoded', 'device_fingerprint_encoded',
            'helpful_votes_percent_num', 'days_old', 'shared_ip',
            'extreme_rating', 'short_review', 'very_fast_typing'
        ]
        
        # Combine all features
        feature_columns = numerical_features + categorical_features
        structural_features = df[feature_columns].values
        
        # Scale features
        if self.feature_scaler is None:
            self.feature_scaler = StandardScaler()
            structural_features = self.feature_scaler.fit_transform(structural_features)
        else:
            structural_features = self.feature_scaler.transform(structural_features)
        
        # Combine text and structural features
        combined_features = np.hstack([text_features, structural_features])
        
        return combined_features, text_features
    
    def train_models(self, training_file: str) -> bool:
        """Train the three-stage classification models with optimized settings"""
        try:
            # Load and prepare data
            print("Loading training data...")
            df = pd.read_excel(training_file)
            
            # Extract features using fast method
            df = self.extract_fast_features(df)
            
            # Prepare features
            X, X_text = self.prepare_features(df)
            y = df['label'].values
            
            print(f"Training on {len(df)} samples with {X.shape[1]} features")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train Script Detection Model (reduced complexity)
            print("\nTraining Script Detection Model...")
            script_labels = ['Script' if label == 'Script' else 'Non-Script' for label in y_train]
            self.script_model = RandomForestClassifier(
                n_estimators=50,  # Reduced from 100
                max_depth=10,     # Limit tree depth
                random_state=42,
                n_jobs=-1         # Use all CPU cores
            )
            self.script_model.fit(X_train, script_labels)
            
            # Train AI Detection Model (reduced complexity)
            print("Training AI Detection Model...")
            ai_labels = ['AI' if label == 'AI' else 'Non-AI' for label in y_train]
            self.ai_model = GradientBoostingClassifier(
                n_estimators=50,  # Reduced from 100
                max_depth=6,      # Limit tree depth
                random_state=42
            )
            self.ai_model.fit(X_train, ai_labels)
            
            # Train Hijacked Detection Model
            print("Training Hijacked Detection Model...")
            hijacked_labels = ['Hijacked' if label == 'Hijacked' else 'Non-Hijacked' for label in y_train]
            self.hijacked_model = LogisticRegression(
                random_state=42, 
                max_iter=500,     # Reduced from 1000
                solver='liblinear'  # Faster solver for smaller datasets
            )
            self.hijacked_model.fit(X_train, hijacked_labels)
            
            # Quick evaluation
            print("\n=== Model Evaluation ===")
            
            # Test Script Model
            script_test_labels = ['Script' if label == 'Script' else 'Non-Script' for label in y_test]
            script_pred = self.script_model.predict(X_test)
            print(f"Script Detection Accuracy: {accuracy_score(script_test_labels, script_pred):.3f}")
            
            # Test AI Model
            ai_test_labels = ['AI' if label == 'AI' else 'Non-AI' for label in y_test]
            ai_pred = self.ai_model.predict(X_test)
            print(f"AI Detection Accuracy: {accuracy_score(ai_test_labels, ai_pred):.3f}")
            
            # Test Hijacked Model
            hijacked_test_labels = ['Hijacked' if label == 'Hijacked' else 'Non-Hijacked' for label in y_test]
            hijacked_pred = self.hijacked_model.predict(X_test)
            print(f"Hijacked Detection Accuracy: {accuracy_score(hijacked_test_labels, hijacked_pred):.3f}")
            
            # Save models
            self.save_models()
            self.models_trained = True
            
            print("\n✅ All models trained successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            return False
    
    def predict(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Make predictions using the optimized three-stage classification system"""
        try:
            if not self.models_trained:
                print("Models not trained! Please train models first.")
                return None
            
            # Extract features using fast method
            df = self.extract_fast_features(df)
            
            # Prepare features
            X, X_text = self.prepare_features(df)
            
            print("Making predictions...")
            
            # Stage 1: Script Detection
            script_probs = self.script_model.predict_proba(X)
            
            # Stage 2: AI Detection  
            ai_probs = self.ai_model.predict_proba(X)
            
            # Stage 3: Hijacked Detection
            hijacked_probs = self.hijacked_model.predict_proba(X)
            
            # Combine predictions using confidence-based approach
            results = df.copy()
            
            # Extract probabilities (handle cases where classes might be missing)
            script_prob = np.array([prob[1] if len(prob) > 1 else 0.5 for prob in script_probs])
            ai_prob = np.array([prob[1] if len(prob) > 1 else 0.5 for prob in ai_probs])
            hijacked_prob = np.array([prob[1] if len(prob) > 1 else 0.5 for prob in hijacked_probs])
            
            # Calculate human probability (inverse of fake probabilities)
            human_prob = 1 - np.maximum.reduce([script_prob, ai_prob, hijacked_prob])
            
            # Assign final predictions based on highest probability
            final_predictions = []
            confidence_scores = []
            
            for i in range(len(df)):
                probs = {
                    'Script': script_prob[i],
                    'AI': ai_prob[i], 
                    'Hijacked': hijacked_prob[i],
                    'Human': human_prob[i]
                }
                
                # Find the class with highest probability
                predicted_class = max(probs, key=probs.get)
                confidence = probs[predicted_class]
                
                final_predictions.append(predicted_class)
                confidence_scores.append(confidence)
            
            # Add prediction results to dataframe
            results['script_probability'] = script_prob
            results['ai_probability'] = ai_prob
            results['hijacked_probability'] = hijacked_prob
            results['human_probability'] = human_prob
            results['final_prediction'] = final_predictions
            results['confidence'] = confidence_scores
            
            print(f"✅ Predictions completed for {len(results)} reviews")
            return results
            
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            return None
    
    def save_models(self):
        """Save trained models and preprocessing components"""
        try:
            # Save models
            joblib.dump(self.script_model, os.path.join(self.model_dir, 'script_model.pkl'))
            joblib.dump(self.ai_model, os.path.join(self.model_dir, 'ai_model.pkl'))
            joblib.dump(self.hijacked_model, os.path.join(self.model_dir, 'hijacked_model.pkl'))
            
            # Save preprocessing components
            joblib.dump(self.feature_scaler, os.path.join(self.model_dir, 'feature_scaler.pkl'))
            joblib.dump(self.tfidf_vectorizer, os.path.join(self.model_dir, 'tfidf_vectorizer.pkl'))
            
            print("✅ Models saved successfully!")
            
        except Exception as e:
            print(f"❌ Failed to save models: {str(e)}")
    
    def load_models(self) -> bool:
        """Load pre-trained models and preprocessing components"""
        try:
            model_files = {
                'script_model.pkl': 'script_model',
                'ai_model.pkl': 'ai_model',
                'hijacked_model.pkl': 'hijacked_model',
                'feature_scaler.pkl': 'feature_scaler',
                'tfidf_vectorizer.pkl': 'tfidf_vectorizer'
            }
            
            for filename, attr_name in model_files.items():
                filepath = os.path.join(self.model_dir, filename)
                if os.path.exists(filepath):
                    setattr(self, attr_name, joblib.load(filepath))
                else:
                    print(f"❌ Model file not found: {filename}")
                    return False
            
            self.models_trained = True
            print("✅ Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load models: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict:
        """Get information about trained models"""
        info = {
            'models_trained': self.models_trained,
            'script_model': str(type(self.script_model).__name__) if self.script_model else None,
            'ai_model': str(type(self.ai_model).__name__) if self.ai_model else None,
            'hijacked_model': str(type(self.hijacked_model).__name__) if self.hijacked_model else None,
            'feature_scaler': str(type(self.feature_scaler).__name__) if self.feature_scaler else None,
            'tfidf_vectorizer': str(type(self.tfidf_vectorizer).__name__) if self.tfidf_vectorizer else None
        }
        return info
    
    def analyze_review_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze patterns in the review data for insights (simplified)"""
        analysis = {}
        
        # IP address patterns
        ip_counts = df['ip_address'].value_counts()
        analysis['shared_ips'] = (ip_counts > 1).sum()
        analysis['max_reviews_per_ip'] = ip_counts.max()
        
        # Device fingerprint patterns
        device_counts = df['device_fingerprint'].value_counts()
        analysis['device_patterns'] = device_counts.to_dict()
        
        # Rating patterns
        analysis['rating_distribution'] = df['stars'].value_counts().to_dict()
        analysis['extreme_ratings'] = ((df['stars'] == 1) | (df['stars'] == 5)).sum()
        
        # Quick text patterns
        analysis['avg_review_length'] = df['review'].str.len().mean()
        analysis['short_reviews'] = (df['review'].str.len() < 50).sum()
        
        return analysis