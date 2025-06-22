🛡️ Amazon Trust (HackOn 2025 Prototype)

Overview

**Amazon Trust** is a machine learning-based prototype built for **Amazon HackOn 2025**.  
It helps detect **fake product reviews** and **bot-based buying activity** on Amazon using two intelligent classifiers:

🔍 **Fake Review Classifier**
🤖 **Bot Buying Classifier**

---

✅ Features

🔍 Fake Review Classifier
- Generate 4 types of reviews: Human, Scripted, AI-generated, Hijacked
- Train ML models (Random Forest, XGBoost, Logistic Regression)
- Predict fake vs real reviews with confidence score
- Store and analyze review data in Excel

🤖 Bot Buying Classifier
- Simulate genuine and bot purchases with labeled behavior
- Extract features like burst buying, IP reuse, coupon usage, interaction time
- Train ML models to classify bot vs human buyersComponent
-Train ML models (RandomForest, VotingClassifier, StandardScaler, joblib)

---

📁 Folder Structure

Amazon Trust/
├── main.py        ( Entry point for both classifiers )
├── fake_review_classifier/       ( Review detection module )
│ ├── app.py, app_ui.py       ( Streamlit UI and control )
│ ├── core_logic.py      ( Data and AI integration )
│ ├── ml_classifier.py     ( ML logic )
│ └── style.css     ( UI styling )
| ├── data/     ( stores .xlsx files )
| └── models/     (stores trained .pkl models )
|
├── buying_bots_classifier/     ( Bot buying detection module )
│ ├── app.py     ( Bot simulation and prediction UI )
│ ├── ml_classifier.py     ( ML logic for bot detection )
| ├── data/     ( stores .xlsx files )
| └── models/     (stores trained .pkl models )


---

SETUP

1. Clone the project
bash
git clone https://github.com/your-username/amazon-trust.git
cd amazon-trust

2. Install required packages
bash
pip install streamlit pandas numpy scikit-learn joblib xgboost lightgbm openpyxl google-generativeai

3. Run the app
bash
streamlit run main.py


Gemini AI Setup (Optional)
 core_logic.py, replace:
GEMINI_API_KEY = "your-api-key"



Notes
This is a prototype project for HackOn — not for production use
Review and purchase data are saved as .xlsx files
Trained models are saved as .pkl in models/

