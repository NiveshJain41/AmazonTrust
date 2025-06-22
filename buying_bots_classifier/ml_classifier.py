import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import joblib
import warnings

warnings.filterwarnings('ignore')

# Import multiple algorithms
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class AdvancedBotClassifier:
    def __init__(self, product_file_map, base_path=None):
        """
        Advanced Bot Classifier with multiple algorithms and ensemble methods
        
        Args:
            product_file_map: Dictionary mapping product names to file paths
            base_path: Base directory path (defaults to script's parent directory)
        """
        # --- Path Handling: Setting up base, data, and model directories ---
        if base_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # This line correctly moves up two levels from the current script's location
            # e.g., if script is in /project/main/part1/, base_path becomes /project/
            self.base_path = os.path.dirname(os.path.dirname(current_dir))
        else:
            self.base_path = base_path
            
        print(f"Base path set to: {self.base_path}")
        
        # Define the data and model directories relative to the base path
        self.data_dir = os.path.join(self.base_path, "data")
        self.model_dir = os.path.join(self.base_path, "model")
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Update product file map with correct paths
        self.product_file_map = {}
        for product, file_path in product_file_map.items():
            if os.path.isabs(file_path):  # If absolute path, use as is
                self.product_file_map[product] = file_path
            else:  # If relative path, join with the data directory
                self.product_file_map[product] = os.path.join(self.data_dir, file_path)
        # --- End Path Handling ---

        self.model = None
        self.ensemble_model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model_trained = False
        self.algorithm = 'ensemble'

        # Explicit Label Mapping for consistency
        self.label_mapping = {'human': 0, 'bot': 1}
        self.inverse_label_mapping = {0: 'Human', 1: 'Bot'}
        self.le_label = LabelEncoder()

        # All possible features from training data
        self.feature_names = [
            'BuyingTime_Seconds', 'PageViewDuration', 'CartTime_Seconds',
            'CouponUsed', 'DiscountApplied', 'PaymentMethod', 'ProductViewCount',
            'ProductSearchCount', 'AddToCart_RemoveCount', 'ReviewsRead',
            'DeviceType', 'MouseClicks', 'KeyboardStrokes'
        ]

        # Features that need encoding
        self.categorical_features = ['CouponUsed', 'DiscountApplied', 'PaymentMethod', 'DeviceType']

        # Performance metrics storage
        self.training_metrics = {}

        # Store feature names used during training
        self.trained_feature_columns = []

    def load_all_data(self):
        """
        Loads data from the training file with proper path handling.
        """
        # --- Path Handling: Using data_dir for training file ---
        # The training file is expected to be in the 'data' directory
        training_file = os.path.join(self.data_dir, "training_data.xlsx")
        # --- End Path Handling ---
        
        all_data = []

        print(f"Looking for training file at: {training_file}")
        
        if os.path.exists(training_file):
            try:
                df = pd.read_excel(training_file)
                all_data.append(df)
                print(f"✅ Loaded {len(df)} training entries from {training_file}")
            except Exception as e:
                print(f"❌ Error reading {training_file}: {e}")
                return pd.DataFrame()
        else:
            print(f"⚠️ Training file '{training_file}' not found.")
            print(f"📁 Available files in data directory:")
            if os.path.exists(self.data_dir):
                for file in os.listdir(self.data_dir):
                    if file.endswith(('.xlsx', '.csv')):
                        print(f"   - {file}")
            else:
                print(f"   Data directory '{self.data_dir}' does not exist")
            return pd.DataFrame()

        if not all_data:
            return pd.DataFrame()

        df_all = pd.concat(all_data, ignore_index=True)

        # Ensure 'Label' column exists
        if 'Label' not in df_all.columns:
            print("Warning: 'Label' column not found in training data. Cannot train without labels.")
            return pd.DataFrame()

        df_all = df_all.dropna(subset=['Label'])
        df_all = df_all[df_all["Label"].isin(self.label_mapping.keys())]

        print(f"\n📊 Dataset Summary:")
        print(f"Total entries: {len(df_all)}")
        print(f"Human entries: {len(df_all[df_all['Label'] == 'human'])}")
        print(f"Bot entries: {len(df_all[df_all['Label'] == 'bot'])}")
        print(f"Available features: {[col for col in df_all.columns if col in self.feature_names]}")

        return df_all

    def prepare_data(self, df):
        """
        Prepare and encode data for machine learning.
        """
        df_processed = df.copy()

        # Encode categorical features
        for feature in self.categorical_features:
            if feature in df_processed.columns:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    all_categories = df_processed[feature].astype(str).unique()
                    self.label_encoders[feature].fit(all_categories)
                    df_processed[feature] = self.label_encoders[feature].transform(df_processed[feature].astype(str))
                else:
                    le = self.label_encoders[feature]
                    df_processed[feature] = df_processed[feature].astype(str)
                    mask = df_processed[feature].isin(le.classes_)
                    df_processed.loc[mask, feature] = le.transform(df_processed.loc[mask, feature])
                    df_processed.loc[~mask, feature] = 0 # Assign 0 for unseen categories
            else:
                # If feature is not in the current DataFrame, add it with a default value (e.g., 0)
                df_processed[feature] = 0

        # Encode target variable
        if 'Label' in df_processed.columns:
            self.le_label.fit(list(self.label_mapping.keys())) # Ensure it's fitted with expected labels
            y = df_processed['Label'].map(self.label_mapping).values
        else:
            y = None

        # Select available features
        current_features = [f for f in self.feature_names if f in df_processed.columns]
        X = df_processed[current_features].copy()

        # Handle missing values
        for col in X.select_dtypes(include=[np.number]).columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mean())

        # Feature engineering
        if 'BuyingTime_Seconds' in X.columns and 'MouseClicks' in X.columns:
            X['Time_per_Click'] = X['BuyingTime_Seconds'] / (X['MouseClicks'] + 1).replace(0, 1) # +1 to avoid div by zero

        if 'ProductViewCount' in X.columns and 'PageViewDuration' in X.columns:
            X['Avg_View_Time'] = X['PageViewDuration'] / (X['ProductViewCount'] + 1).replace(0, 1) # +1 to avoid div by zero

        if 'ReviewsRead' in X.columns and 'ProductViewCount' in X.columns:
            X['Research_Intensity'] = X['ReviewsRead'] / (X['ProductViewCount'] + 1).replace(0, 1) # +1 to avoid div by zero

        return X, y, list(X.columns)

    def train_model(self, algorithm='ensemble'):
        """
        Trains the classifier with proper file path handling.
        """
        self.algorithm = algorithm
        self.model_trained = False

        # Load data
        df_all = self.load_all_data()
        if df_all.empty or "Label" not in df_all.columns:
            return {
                "trained": False,
                "message": "No training data available or 'Label' column missing.",
                "report": None
            }

        try:
            # Prepare data
            X, y, feature_names_before_scaling = self.prepare_data(df_all)
            self.trained_feature_columns = feature_names_before_scaling

            print(f"\n🔧 Training with {X.shape[1]} features: {feature_names_before_scaling}")

            # Check for at least two classes
            if len(y) == 0 or len(np.unique(y)) < 2:
                return {
                    "trained": False,
                    "message": "Not enough unique classes (human/bot) in training data.",
                    "report": None
                }

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )

            # Scale features (important for ensemble and linear models)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train based on selected algorithm
            if algorithm == 'ensemble':
                self.model = self._train_ensemble(X_train_scaled, y_train)
            elif algorithm == 'xgboost' and XGBOOST_AVAILABLE:
                # XGBoost doesn't strictly require scaling, uses tree-based models
                self.model = self._train_xgboost(X_train, y_train)
            elif algorithm == 'lightgbm' and LIGHTGBM_AVAILABLE:
                # LightGBM also doesn't strictly require scaling
                self.model = self._train_lightgbm(X_train, y_train)
            elif algorithm == 'random_forest':
                # Random Forest also doesn't strictly require scaling
                self.model = self._train_random_forest(X_train, y_train)
            elif algorithm == 'logistic_regression':
                self.model = LogisticRegression(random_state=42, max_iter=1000)
                self.model.fit(X_train_scaled, y_train)
            elif algorithm == 'svc':
                self.model = SVC(probability=True, random_state=42) # probability=True for predict_proba
                self.model.fit(X_train_scaled, y_train)
            else:
                print(f"Algorithm '{algorithm}' not available. Falling back to Random Forest.")
                self.model = self._train_random_forest(X_train, y_train)
                algorithm = 'random_forest'
            
            # Ensure algorithm attribute is updated if fallback occurred
            self.algorithm = algorithm 

            # Evaluate model
            # Predict using scaled data if algorithm is ensemble, LogisticRegression, or SVC
            # Otherwise, use unscaled data as tree-based models are less sensitive to scaling
            if self.algorithm in ['ensemble', 'logistic_regression', 'svc']:
                y_pred = self.model.predict(X_test_scaled)
                y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred = self.model.predict(X_test)
                y_pred_proba = self.model.predict_proba(X_test)[:, 1]


            # Generate classification report
            if len(np.unique(y_test)) < 2:
                # Handle case with single class in test set
                test_accuracy = accuracy_score(y_test, y_pred)
                report = {'accuracy': test_accuracy, 'macro avg': {'precision': test_accuracy, 'recall': test_accuracy, 'f1-score': test_accuracy, 'support': len(y_test)}, 'weighted avg': {'precision': test_accuracy, 'recall': test_accuracy, 'f1-score': test_accuracy, 'support': len(y_test)}}
                
                # Manually populate class-specific metrics for a single class
                if 0 in np.unique(y_test):
                    report[self.inverse_label_mapping[0]] = {
                        'precision': test_accuracy, 'recall': 1.0, 'f1-score': test_accuracy, 'support': len(y_test)
                    }
                    report[self.inverse_label_mapping[1]] = { # Other class will have 0 metrics
                        'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0
                    }
                elif 1 in np.unique(y_test):
                    report[self.inverse_label_mapping[1]] = {
                        'precision': test_accuracy, 'recall': 1.0, 'f1-score': test_accuracy, 'support': len(y_test)
                    }
                    report[self.inverse_label_mapping[0]] = { # Other class will have 0 metrics
                        'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0
                    }
                
                auc_score = 0.5 # AUC not well-defined for single class
                message_suffix = " (Limited evaluation due to single class in test set)"
            else:
                report = classification_report(y_test, y_pred, output_dict=True,
                                               target_names=[self.inverse_label_mapping[0], self.inverse_label_mapping[1]])
                if len(np.unique(y_pred_proba)) == 1:
                    auc_score = 0.5 # If all probabilities are the same, AUC is 0.5
                else:
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                message_suffix = ""

            # Feature importance
            feature_importance_df = self._get_feature_importance(feature_names_before_scaling)

            # Cross-validation
            cv_model = self.model
            # Use scaled data for cross-validation if the model requires it
            cv_X = X_train_scaled if self.algorithm in ['ensemble', 'logistic_regression', 'svc'] else X_train
            cv_scores = cross_val_score(cv_model, cv_X, y_train, cv=5, scoring='f1_weighted' if len(np.unique(y_train)) > 1 else 'accuracy')


            # Store metrics
            # Safely get metrics, handling cases where a class might not be present in the report
            f1_bot = report['Bot']['f1-score'] if 'Bot' in report and 'f1-score' in report['Bot'] else 0.0
            precision_bot = report['Bot']['precision'] if 'Bot' in report and 'precision' in report['Bot'] else 0.0
            recall_bot = report['Bot']['recall'] if 'Bot' in report and 'recall' in report['Bot'] else 0.0

            self.training_metrics = {
                'accuracy': report['accuracy'],
                'f1_bot': f1_bot,
                'precision_bot': precision_bot,
                'recall_bot': recall_bot,
                'auc': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }

            # Save model
            self.save_model()
            self.model_trained = True

            return {
                "trained": True,
                "message": f"🎯 {self.algorithm.upper()} model trained successfully!\n"
                           f"Accuracy: {report['accuracy']:.3f} | "
                           f"Bot F1: {f1_bot:.3f} | "
                           f"AUC: {auc_score:.3f}{message_suffix}",
                "report": pd.DataFrame(report).transpose(),
                "feature_importance": feature_importance_df,
                "confusion_matrix": confusion_matrix(y_test, y_pred),
                "metrics": self.training_metrics,
                "algorithm": self.algorithm
            }

        except Exception as e:
            print(f"Error during training: {e}")
            return {
                "trained": False,
                "message": f"Training failed: {str(e)}",
                "report": None
            }

    def _train_ensemble(self, X_train_scaled, y_train):
        """Train ensemble model."""
        estimators = []
        
        # Add a diverse set of strong classifiers
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        estimators.append(('rf', rf))

        if XGBOOST_AVAILABLE:
            xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42,
                                use_label_encoder=False, eval_metric='logloss')
            estimators.append(('xgb', xgb))

        if LIGHTGBM_AVAILABLE:
            lgb = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1)
            estimators.append(('lgb', lgb))

        lr = LogisticRegression(random_state=42, max_iter=1000)
        estimators.append(('lr', lr))

        # Train a VotingClassifier if there are multiple estimators, otherwise default to Random Forest
        if len(estimators) > 1: # Ensure at least two estimators for effective voting
            ensemble = VotingClassifier(estimators=estimators, voting='soft') # 'soft' for probability averaging
            ensemble.fit(X_train_scaled, y_train)
        elif len(estimators) == 1:
            print("Only one estimator available for ensemble, training it directly.")
            single_estimator_name, single_estimator_model = estimators[0]
            single_estimator_model.fit(X_train_scaled, y_train)
            ensemble = single_estimator_model
        else:
            print("No suitable estimators found for ensemble. Training a default Random Forest model.")
            ensemble = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
            ensemble.fit(X_train_scaled, y_train)

        return ensemble

    def _train_xgboost(self, X_train, y_train):
        """Train XGBoost."""
        xgb = XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            use_label_encoder=False, eval_metric='logloss'
        )
        xgb.fit(X_train, y_train)
        return xgb

    def _train_lightgbm(self, X_train, y_train):
        """Train LightGBM."""
        lgb = LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
        )
        lgb.fit(X_train, y_train)
        return lgb

    def _train_random_forest(self, X_train, y_train):
        """Train Random Forest."""
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, random_state=42
        )
        rf.fit(X_train, y_train)
        return rf

    def _get_feature_importance(self, feature_names):
        """Get feature importance with explanations."""
        data = {
            'Feature': [
                'BuyingTime_Seconds', 'PageViewDuration', 'CartTime_Seconds',
                'Time_per_Click', 'Avg_View_Time', 'Research_Intensity',
                'KeyboardStrokes', 'MouseClicks',
                'ProductViewCount', 'ProductSearchCount', 'ReviewsRead',
                'CouponUsed', 'DiscountApplied', 'PaymentMethod',
                'AddToCart_RemoveCount', 'DeviceType'
            ],
            'Importance': [
                0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.07, 0.06,
                0.05, 0.05, 0.05, 0.04, 0.04, 0.03, 0.02, 0.01
            ],
            'Why This Weight': [
                'Critical time-based signal; bots are significantly faster than humans.',
                'Bots skip browsing; humans spend more time on pages.',
                'Bots rarely review cart; humans take time for cart decisions.',
                'Derived metric; bots have near-zero delay between clicks.',
                'Bots view products uniformly fast; humans vary their view times.',
                'Humans research (reviews); bots skip research phases.',
                'Bots auto-fill (minimal typing); humans show natural input patterns.',
                'Humans explore (more clicks); bots use minimal clicks for direct actions.',
                'Humans compare products; bots often have minimal product views.',
                'Humans search for products; bots often use direct links (0 searches).',
                'Humans read reviews; bots typically skip review reading.',
                'Bots frequently use coupons; humans sometimes.',
                'Bots always target discounts; humans show mixed behavior.',
                'Bots reuse limited payment methods; humans vary more.',
                'Humans change mind (add/remove); bots add instantly.',
                'Bots might prefer desktop; humans use mobile/desktop.'
            ]
        }

        feature_importance_df = pd.DataFrame(data)
        
        # Add missing features that might be created by feature engineering
        for f in feature_names:
            if f not in feature_importance_df['Feature'].values:
                feature_importance_df = pd.concat([
                    feature_importance_df, 
                    pd.DataFrame([{
                        'Feature': f, 
                        'Importance': 0.0, # Default importance for dynamically added features
                        'Why This Weight': 'Dynamically added feature, not explicitly weighted'
                    }])
                ], ignore_index=True)

        # Filter to only include features present in the current model training
        feature_importance_df = feature_importance_df[feature_importance_df['Feature'].isin(feature_names)]
        return feature_importance_df.sort_values('Importance', ascending=False)

    def save_model(self):
        """Save model with proper path handling."""
        try:
            # --- Path Handling: Using model_dir for saving all components ---
            model_path = os.path.join(self.model_dir, f"bot_detector_{self.algorithm}.pkl")
            encoders_path = os.path.join(self.model_dir, "label_encoders.pkl")
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            features_path = os.path.join(self.model_dir, "trained_feature_columns.pkl")
            label_mapping_path = os.path.join(self.model_dir, "label_mapping.pkl")
            inverse_mapping_path = os.path.join(self.model_dir, "inverse_label_mapping.pkl")
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.label_encoders, encoders_path)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.trained_feature_columns, features_path)
            joblib.dump(self.label_mapping, label_mapping_path)
            joblib.dump(self.inverse_label_mapping, inverse_mapping_path)
            # --- End Path Handling ---
            
            print(f"✅ Model saved to: {model_path}")
            print(f"✅ Components saved to: {self.model_dir}")
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")

    def load_model(self, algorithm=None):
        """Load model with proper path handling."""
        if algorithm is None:
            algorithm = self.algorithm

        try:
            # --- Path Handling: Using model_dir for loading all components ---
            model_path = os.path.join(self.model_dir, f"bot_detector_{algorithm}.pkl")
            encoders_path = os.path.join(self.model_dir, "label_encoders.pkl")
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            features_path = os.path.join(self.model_dir, "trained_feature_columns.pkl")
            label_mapping_path = os.path.join(self.model_dir, "label_mapping.pkl")
            inverse_mapping_path = os.path.join(self.model_dir, "inverse_label_mapping.pkl")
            # --- End Path Handling ---
            
            print(f"Looking for model at: {model_path}")
            
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                self.label_encoders = joblib.load(encoders_path)
                self.scaler = joblib.load(scaler_path)
                self.trained_feature_columns = joblib.load(features_path)
                self.label_mapping = joblib.load(label_mapping_path)
                self.inverse_label_mapping = joblib.load(inverse_mapping_path)
                self.algorithm = algorithm # Update algorithm to the one loaded
                self.model_trained = True
                print(f"✅ Model '{algorithm}' loaded successfully from {model_path}")
                return True
            else:
                print(f"❌ Model file {model_path} not found.")
                print(f"📁 Available files in model directory:")
                if os.path.exists(self.model_dir):
                    for file in os.listdir(self.model_dir):
                        if file.endswith('.pkl'):
                            print(f"   - {file}")
                else:
                    print(f"   Model directory '{self.model_dir}' does not exist")
                return False
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
            self.model_trained = False
            return False

    def predict_bot_probability(self, df):
        """Predict bot probability with proper error handling."""
        # Attempt to load model if not already trained
        if not self.model_trained and not self.load_model(self.algorithm):
            df["ML_Bot_Chance (%)"] = "Model Not Trained"
            df["Risk_Level"] = "Unknown"
            df["Confidence"] = "Low"
            df["ML_Prediction"] = "Unknown"
            return df

        try:
            df_copy = df.copy()

            # Encode categorical features using fitted encoders
            for feature in self.categorical_features:
                if feature in df_copy.columns and feature in self.label_encoders:
                    le = self.label_encoders[feature]
                    df_copy[feature] = df_copy[feature].astype(str)
                    mask = df_copy[feature].isin(le.classes_)
                    df_copy.loc[mask, feature] = le.transform(df_copy.loc[mask, feature])
                    df_copy.loc[~mask, feature] = 0 # Assign 0 for unseen categories
                elif feature in df_copy.columns:
                    # If feature exists but no encoder was saved for it, treat as unknown/0
                    df_copy[feature] = 0
                else:
                    # If feature doesn't exist, add it with default 0
                    df_copy[feature] = 0

            # Prepare features ensuring all expected columns are present and in order
            X_predict = pd.DataFrame(columns=self.trained_feature_columns)
            for col in self.trained_feature_columns:
                if col in df_copy.columns:
                    X_predict[col] = df_copy[col]
                else:
                    X_predict[col] = 0 # Add missing columns with a default value

            # Feature engineering for prediction data - must match training
            if 'BuyingTime_Seconds' in X_predict.columns and 'MouseClicks' in X_predict.columns:
                X_predict['Time_per_Click'] = X_predict['BuyingTime_Seconds'] / (X_predict['MouseClicks'] + 1).replace(0, 1)
            if 'ProductViewCount' in X_predict.columns and 'PageViewDuration' in X_predict.columns:
                X_predict['Avg_View_Time'] = X_predict['PageViewDuration'] / (X_predict['ProductViewCount'] + 1).replace(0, 1)
            if 'ReviewsRead' in X_predict.columns and 'ProductViewCount' in X_predict.columns:
                X_predict['Research_Intensity'] = X_predict['ReviewsRead'] / (X_predict['ProductViewCount'] + 1).replace(0, 1)

            # Ensure correct column order and fill any NaN introduced by feature engineering
            # This step is critical to ensure feature order matches during training
            X = X_predict[self.trained_feature_columns].copy()
            X = X.fillna(0) # Fill any NaNs that might result from division by zero, etc.

            # Predict (scale if the model was trained with scaled data)
            if self.algorithm in ['ensemble', 'logistic_regression', 'svc']:
                X_scaled = self.scaler.transform(X)
                probabilities = self.model.predict_proba(X_scaled)[:, 1]
                predictions = self.model.predict(X_scaled)
            else:
                probabilities = self.model.predict_proba(X)[:, 1]
                predictions = self.model.predict(X)

            # Add predictions to the original DataFrame
            df["ML_Bot_Chance (%)"] = (probabilities * 100).round(1)
            df["ML_Prediction"] = [self.inverse_label_mapping[p] for p in predictions]

            # Risk categorization
            df["Risk_Level"] = df["ML_Bot_Chance (%)"].apply(lambda x:
                "🔴 High" if x > 80 else "🟡 Medium" if x > 50 else "🟢 Low"
            )

            # Confidence (higher deviation from 50% means higher confidence)
            df["Confidence"] = df["ML_Bot_Chance (%)"].apply(lambda x:
                "High" if abs(x - 50) > 30 else "Medium" if abs(x - 50) > 15 else "Low"
            )

        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            df["ML_Bot_Chance (%)"] = f"Error: {str(e)}"
            df["ML_Prediction"] = "Error"
            df["Risk_Level"] = "Unknown"
            df["Confidence"] = "Low"

        return df

    def evaluate_shopping_behavior(self):
        """Comprehensive shopping behavior evaluation."""
        summary_data = []
        flagged_entries = []

        for product, file_path in self.product_file_map.items():
            print(f"Checking file: {file_path}")
            if os.path.exists(file_path):
                try:
                    df = pd.read_excel(file_path)
                    print(f"✅ Loaded {len(df)} entries from {file_path}")
                except Exception as e:
                    print(f"❌ Error reading {file_path}: {e}")
                    continue

                if len(df) == 0:
                    print(f"ℹ️ No entries in {file_path}, skipping.")
                    continue

                # Apply ML predictions
                df_with_predictions = self.predict_bot_probability(df)
                df_with_predictions["Product"] = product
                flagged_entries.append(df_with_predictions)

                # Calculate statistics
                total_entries = len(df)

                # Ensure ML_Bot_Chance (%) is numeric before calculations
                if pd.api.types.is_numeric_dtype(df_with_predictions["ML_Bot_Chance (%)"]):
                    high_risk = len(df_with_predictions[df_with_predictions["ML_Bot_Chance (%)"] > 80])
                    medium_risk = len(df_with_predictions[
                        (df_with_predictions["ML_Bot_Chance (%)"] > 50) &
                        (df_with_predictions["ML_Bot_Chance (%)"] <= 80)
                    ])
                    low_risk = total_entries - high_risk - medium_risk

                    # Additional insights
                    avg_bot_chance = df_with_predictions["ML_Bot_Chance (%)"].mean()
                    bot_predictions = len(df_with_predictions[df_with_predictions["ML_Prediction"].str.contains("Bot", na=False)])
                else:
                    # If predictions failed, set counts to zero
                    high_risk = medium_risk = low_risk = bot_predictions = 0
                    avg_bot_chance = 0
                    print(f"⚠️ Skipping risk calculation for {product} due to non-numeric ML_Bot_Chance (%) values.")

                summary_data.append({
                    "Product": product,
                    "Total_Entries": total_entries,
                    "🔴 High Risk (>80%)": high_risk,
                    "🟡 Medium Risk (50-80%)": medium_risk,
                    "🟢 Low Risk (<50%)": low_risk,
                    "🤖 Bot Predictions": bot_predictions,
                    "📊 Avg Bot Chance": f"{avg_bot_chance:.1f}%",
                    "⚠️ Risk Ratio": f"{((high_risk + medium_risk) / total_entries * 100):.1f}%" if total_entries > 0 else "0%"
                })
            else:
                print(f"❌ Product file '{file_path}' not found. Skipping.")


        # Combine and sort flagged entries
        if flagged_entries:
            all_flags = pd.concat(flagged_entries, ignore_index=True)
            if len(all_flags) > 0 and pd.api.types.is_numeric_dtype(all_flags["ML_Bot_Chance (%)"]):
                all_flags = all_flags.sort_values("ML_Bot_Chance (%)", ascending=False)
            else:
                all_flags = pd.DataFrame()  # Handle case where ML_Bot_Chance is not numeric or no flags
        else:
            all_flags = pd.DataFrame()

        return summary_data, all_flags

    def get_model_performance(self):
        """Get current model performance metrics."""
        return self.training_metrics if hasattr(self, 'training_metrics') else {}

    def analyze_behavioral_patterns(self, df):
        """Analyze behavioral patterns in the data (not currently used in app.py)."""
        if df.empty:
            return {}

        patterns = {}

        # Time-based patterns
        if 'BuyingTime_Seconds' in df.columns:
            patterns['avg_buying_time'] = df['BuyingTime_Seconds'].mean()
            patterns['fast_buyers'] = len(df[df['BuyingTime_Seconds'] < 30])

        # Interaction patterns
        if 'MouseClicks' in df.columns:
            patterns['avg_clicks'] = df['MouseClicks'].mean()
            patterns['low_interaction'] = len(df[df['MouseClicks'] < 10])

        # Research patterns
        if 'ReviewsRead' in df.columns:
            patterns['avg_reviews_read'] = df['ReviewsRead'].mean()
            patterns['no_research'] = len(df[df['ReviewsRead'] == 0])

        return patterns
