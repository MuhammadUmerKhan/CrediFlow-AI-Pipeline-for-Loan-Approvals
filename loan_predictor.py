import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib as jb
from langchain.schema import HumanMessage
import os, langchain_groq, re
import mlflow
from dotenv import load_dotenv
from scripts.config import PREPROCESSING_CONFIG, SCALER_PATH, MODEL_DIR

load_dotenv()

# ----------------------------------Streamlit page configuration-----------------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------Custom CSS for styling-----------------------------------------
st.markdown("""
    <style>
        /* Financial-Themed Dark Styles */
        .stApp {
            background: linear-gradient(rgba(31, 41, 55, 0.9), rgba(31, 41, 55, 0.9)), url('https://media.licdn.com/dms/image/v2/D5612AQHy0wPDANw36g/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1680808534854?e=2147483647&v=beta&t=8YIjZWTBts-oy1tiH4ukyPleGJcs1_PIdYhO0oDmeGY');
            background-size: cover;
            background-attachment: fixed;
            color: #f3f4f6;
            font-family: 'Poppins', sans-serif;
        }
        .main-container {
            background: linear-gradient(135deg, rgba(29, 78, 216, 0.85), rgba(4, 120, 87, 0.85));
            border-radius: 15px;
            padding: 30px;
            margin: 20px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.6);
            border: 2px solid #eab308;
            backdrop-filter: blur(10px);
        }
        .main-title {
            font-size: 3.2em;
            font-weight: 700;
            color: #eab308;
            text-align: center;
            margin-bottom: 35px;
            text-shadow: 0 0 12px rgba(234, 179, 8, 0.8);
            animation: pulseGlow 2s ease-in-out infinite;
        }
        .section-title {
            font-size: 2.2em;
            font-weight: 600;
            color: #1d4ed8;
            margin: 40px 0 20px;
            text-shadow: 0 0 10px rgba(29, 78, 216, 0.8);
            border-left: 6px solid #1d4ed8;
            padding-left: 18px;
            animation: slideInLeft 0.6s ease-in-out;
        }
        .system-content {
            font-size: 2.2em;
            font-weight: 600;
            color: #047857;
            text-align: center;
            text-shadow: 0 0 10px rgba(4, 120, 87, 0.8);
            animation: slideInLeft 0.6s ease-in-out;
        }
        .intro-title {
            font-size: 2.5em;
            color: #eab308;
            font-weight: bold;
            text-align: center;
        }
        .intro-subtitle {
            font-size: 1.5em;
            color: #1d4ed8;
            text-align: center;
            text-shadow: 0 0 8px rgba(29, 78, 216, 0.8);
        }
        .content {
            font-size: 1.15em;
            color: #f3f4f6;
            line-height: 1.9;
            text-align: justify;
        }
        .highlight {
            color: #eab308;
            font-weight: bold;
        }
        .separator {
            height: 2px;
            background: linear-gradient(to right, #1d4ed8, #047857);
            margin: 20px 0;
        }
        .stButton>button {
            background: linear-gradient(45deg, #1d4ed8, #047857);
            color: #eab308;
            border-radius: 12px;
            padding: 14px 30px;
            font-weight: 600;
            font-size: 1.1em;
            border: none;
            box-shadow: 0 0 15px rgba(234, 179, 8, 0.8);
            transition: all 0.4s ease;
            position: relative;
            overflow: hidden;
        }
        .stButton>button:hover {
            background: linear-gradient(45deg, #1e40af, #065f46);
            box-shadow: 0 0 25px rgba(234, 179, 8, 1);
            transform: scale(1.1);
            color: #f3f4f6;
        }
        .stButton>button::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 300%;
            height: 300%;
            background: rgba(234, 179, 8, 0.2);
            transition: all 0.6s ease;
            transform: translate(-50%, -50%) scale(0);
            border-radius: 50%;
        }
        .stButton>button:hover::after {
            transform: translate(-50%, -50%) scale(1);
        }
        .stSelectbox, .stNumberInput, .stTextArea {
            background: linear-gradient(135deg, rgba(29, 78, 216, 0.9), rgba(4, 120, 87, 0.9));
            border-radius: 10px;
            padding: 12px;
            border: 1px solid #eab308;
            color: #f3f4f6;
            transition: all 0.3s ease;
        }
        .stSelectbox:hover, .stNumberInput:hover, .stTextArea:hover {
            border-color: #facc15;
            box-shadow: 0 0 8px rgba(234, 179, 8, 0.5);
        }
        .stSelectbox label, .stNumberInput label, .stTextArea label {
            color: #eab308;
            font-weight: 500;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.3em;
            font-weight: 500;
            color: #f3f4f6;
            padding: 15px 30px;
            border-radius: 12px 12px 0 0;
            transition: all 0.3s ease;
            background: linear-gradient(135deg, rgba(29, 78, 216, 0.9), rgba(4, 120, 87, 0.9));
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(45deg, #1d4ed8, #047857);
            color: #eab308;
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: linear-gradient(135deg, #1e40af, #065f46);
            color: #f3f4f6;
        }
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
            background-color: rgba(31, 41, 55, 0.95);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        .stDataFrame table {
            color: #f3f4f6;
        }
        .footer {
            font-size: 0.95em;
            color: #f3f4f6;
            margin-top: 50px;
            text-align: center;
            padding: 25px;
            background: linear-gradient(135deg, rgba(29, 78, 216, 0.85), rgba(4, 120, 87, 0.85));
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            border: 2px solid #eab308;
            backdrop-filter: blur(10px);
        }
        .footer a {
            color: #facc15;
            text-decoration: none;
            font-weight: 600;
            transition: color 0.3s ease;
        }
        .footer a:hover {
            color: #1d4ed8;
            text-decoration: underline;
        }
        .content ul li::marker {
            color: #eab308;
        }
        .prediction-text {
            font-size: 2em;
            font-weight: bold;
            text-align: center;
            text-shadow: 0 0 10px rgba(234, 179, 8, 0.8);
        }
        .prediction-text-approved {
            color: #047857;
        }
        .prediction-text-denied {
            color: #ef4444;
        }
        /* Animations */
        @keyframes pulseGlow {
            0% { text-shadow: 0 0 10px rgba(234, 179, 8, 0.8); }
            50% { text-shadow: 0 0 20px rgba(234, 179, 8, 1); }
            100% { text-shadow: 0 0 10px rgba(234, 179, 8, 0.8); }
        }
        @keyframes slideInLeft {
            from { transform: translateX(-30px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes scaleIn {
            from { transform: scale(0.95); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------Load the sample data----------------------------------
data = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "train.csv")), index_col='id')

approval_1 = data[data['loan_status'] == 1]
approval_0 = data[data['loan_status'] == 0]

if "df_sample_tab1" not in st.session_state:
    approval_1_sample = approval_1.sample(3)
    approval_0_sample = approval_0.sample(3)
    st.session_state.df_sample_tab1 = pd.concat([approval_1_sample, approval_0_sample])

# ----------------------------------Page Title-----------------------------------
st.markdown('<div class="intro-title">💡 Unlock Your Loan Approval Potential! 💡</div>', unsafe_allow_html=True)
st.markdown('<div class="intro-subtitle">Smart insights for confident financial decisions. 🏦✨</div>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📋 Get Loan Approval", "📤 Batch Prediction", "💬 LLM Review Analysis"])

# ----------------------------------Tab 1----------------------------------
with tab1:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="system-content">👋 About Me</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            Hi! I’m <span class="highlight">Muhammad Umer Khan</span>, a dedicated Data Scientist and Machine Learning enthusiast with a Bachelor’s in Computer Science. 
            With hands-on experience in <span class="highlight">🤖 Natural Language Processing (NLP)</span>, 🧠 Machine Learning, and MLOps, I specialize in building intelligent systems, 
            from data pipelines to deployable applications. My journey includes developing recommendation systems, optimizing ANN models, and integrating advanced LLMs, 
            all while pursuing excellence in real-world problem-solving. 🚀
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🎯 Project Overview</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            This project is a state-of-the-art loan approval prediction system, showcasing a complete MLOps pipeline and advanced AI integration. Here's what I've achieved:
            <ul>
                <li><span class="highlight">📊 Exploratory Data Analysis (EDA)</span>: Analyzed the dataset to uncover insights, patterns, and ensure data quality.</li>
                <li><span class="highlight">🛠 Data Preprocessing</span>: Cleaned, transformed, encoded features, and balanced data with SMOTEENN for robust training.</li>
                <li><span class="highlight">🔗 Model Development</span>: Built an Artificial Neural Network (ANN) for classifying loan applications into approved or denied categories.</li>
                <li><span class="highlight">⚙️ Model Optimization</span>: Tuned hyperparameters and applied dropout layers to enhance performance metrics (accuracy, precision, recall, F1-score).</li>
                <li><span class="highlight">📈 Evaluation</span>: Achieved ~94% accuracy with comprehensive metrics, logged via MLflow for tracking.</li>
                <li><span class="highlight">📦 Model Registry</span>: Registered the model in MLflow with versioning and aliases for production readiness.</li>
                <li><span class="highlight">🌐 Deployment</span>: Developed an interactive Streamlit app with real-time predictions, batch processing, and LLM-powered analysis.</li>
                <li><span class="highlight">💬 LLM Integration</span>: Added LLM (Mixtral-8x7B via Grok API) for loan approval predictions and customer sentiment analysis.</li>
                <li><span class="highlight">🧩 MLOps Pipeline</span>: Designed a modular pipeline (ingestion to deployment) with logging and error handling.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📂 Data Overview</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            The
            <a href="https://www.kaggle.com/competitions/playground-series-s4e10" target="_blank" style="color: #facc15;">Dataset</a>
            used in this project contains key attributes for loan approval prediction. Here's a summary:
            <ul>
                <li><span class="highlight">📜 Features</span>: Includes age, income, home ownership, employment length, loan amount, interest rate, credit history, and more.</li>
                <li><span class="highlight">⚖️ Class Balance</span>: Balanced with SMOTEENN to ensure fair evaluation.</li>
                <li><span class="highlight">🔍 Feature Engineering</span>: Derived loan-to-income ratio and other features to boost prediction accuracy.</li>
                <li><span class="highlight">📊 Insights</span>: 
                    <ul>
                        <li>Higher incomes positively correlate with loan approvals.</li>
                        <li>Employment stability significantly influences decisions.</li>
                        <li>High-interest rates increase the likelihood of denial.</li>
                    </ul>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">💻 Technologies & Tools</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            <ul>
                <li><span class="highlight">🔤 Languages & Libraries</span>: Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Imbalanced-learn, Matplotlib, Seaborn, LangChain, MLflow, Joblib.</li>
                <li><span class="highlight">⚙️ Methods</span>: Feature Engineering, Artificial Neural Networks (ANN), SMOTEENN, Hyperparameter Tuning, MLOps.</li>
                <li><span class="highlight">🌐 Deployment</span>: Streamlit for interactive web apps, deployable on cloud platforms.</li>
                <li><span class="highlight">📊 Visualization Tools</span>: Matplotlib and Seaborn for EDA and insights.</li>
                <li><span class="highlight">🧠 NLP & LLM</span>: Grok API (Mixtral-8x7B) for advanced predictions and sentiment analysis.</li>
                <li><span class="highlight">📦 MLOps Tools</span>: MLflow for model tracking, versioning, and registry.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🌟 Why This Project?</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            This project exemplifies my ability to design, implement, and deploy a full MLOps pipeline, integrating cutting-edge AI technologies like ANN and LLM. 
            By solving a real-world loan approval challenge, it highlights my skills in data science, software engineering, and user-focused development. 
            My goal is to empower data-driven decisions with scalable, accessible solutions. ✨
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------Tab 2----------------------------------
with tab2:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Enter Your Details for Loan Approval Prediction</div>', unsafe_allow_html=True)

    # First row with 3 inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        person_age = st.number_input("Enter Your Age", min_value=20.0, max_value=100.0, value=37.0)
    with col2:
        person_income = st.number_input("Enter Your Income", min_value=4200.0, max_value=1900000.0, value=35000.0)
    with col3:
        person_home_ownership_input = st.selectbox("Your Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

    # Second row with 3 inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        person_emp_length = st.number_input("Employment Length (years)", min_value=0.0, max_value=60.0, value=0.0)
    with col2:
        loan_intent_input = st.selectbox("Your Loan Intent", ['EDUCATION', 'MEDICAL', 'PERSONAL', 'VENTURE', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])
    with col3:
        loan_grade_input = st.selectbox("Loan Grade", ["B", "A", "C", "D", "E", "F", "G"])

    # Third row with 3 inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        loan_amnt = st.number_input("Loan Amount", min_value=500.0, max_value=35000.0, value=6000.0)
    with col2:
        loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=4.0, max_value=24.0, value=11.49)
    with col3:
        loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=0.83, value=0.17)

    # Fourth row with 2 inputs
    col1, col2 = st.columns(2)
    with col1:
        cb_person_default_on_file_input = st.selectbox("Default on File", ["No", "Yes"])
    with col2:
        cb_person_cred_hist_length = st.number_input("Credit History (in years)", min_value=2.0, max_value=30.0, value=14.0)

    # ----------------------------------User Input Data------------------------------------
    if st.button("See My Inputs 👀"):
        user_data = pd.DataFrame({
            'person_age': [person_age],
            'person_income': [person_income],
            'person_emp_length': [person_emp_length],
            'loan_grade': [loan_grade_input],
            'loan_amnt': [loan_amnt],
            'loan_int_rate': [loan_int_rate],
            'loan_percent_income': [loan_percent_income],
            'cb_person_default_on_file': [cb_person_default_on_file_input],
            'cb_person_cred_hist_length': [cb_person_cred_hist_length],
        })
        st.markdown('<div class="content">Your Inputs:  📝</div>', unsafe_allow_html=True)
        st.table(user_data)

    # ---------------------------------Apply Mappings------------------------------------
    default_mapping = {'No': 0, 'Yes': 1}
    loan_grade_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}

    # Apply mappings
    cb_person_default_on_file = default_mapping[cb_person_default_on_file_input]
    loan_grade = loan_grade_mapping[loan_grade_input]

    # ---------------------------------One-Hot Encoding manually based on inputs-----------------------------------
    # Initialize one-hot columns with 0
    home_ownership = ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
    loan_intent = ['EDUCATION', 'MEDICAL', 'PERSONAL', 'VENTURE', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT']

    home_ownership_encoded = {f'person_home_ownership_{col}': (1 if col == person_home_ownership_input else 0) for col in home_ownership}
    loan_intent_encoded = {f'loan_intent_{col}': (1 if col == loan_intent_input else 0) for col in loan_intent}

    # ----------------------------------Input DataFrame-----------------------------------
    input_data = pd.DataFrame({
        'person_age': [person_age],
        'person_income': [person_income],
        'person_emp_length': [person_emp_length],
        'loan_grade': [loan_grade],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_default_on_file': [cb_person_default_on_file],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length],
        **home_ownership_encoded,  # Add one-hot encoded columns for home ownership
        **loan_intent_encoded      # Add one-hot encoded columns for loan intent
    })
    # Drop first category for one-hot encoding (match preprocess.py drop_first=True)
    input_data.drop(columns=['person_home_ownership_RENT', 'loan_intent_EDUCATION'], inplace=True)

    # Reorder columns to match preprocess.py output (based on X_train_scaled.csv)
    reordered_columns = [
        'person_age', 'person_income', 'person_emp_length', 'loan_grade',
        'loan_amnt', 'loan_int_rate', 'loan_percent_income',
        'cb_person_default_on_file', 'cb_person_cred_hist_length',
        'person_home_ownership_OWN', 'person_home_ownership_MORTGAGE',
        'person_home_ownership_OTHER', 'loan_intent_MEDICAL',
        'loan_intent_PERSONAL', 'loan_intent_VENTURE',
        'loan_intent_DEBTCONSOLIDATION', 'loan_intent_HOMEIMPROVEMENT'
    ]
    input_data = input_data[reordered_columns]

    # ---------------------------------Log Transformation for Numerical Features------------------------------------
    num_cols = PREPROCESSING_CONFIG['numerical_columns']  # ['loan_amnt', 'loan_int_rate', 'person_income', 'person_age', 'person_emp_length']
    input_data[num_cols] = np.log1p(input_data[num_cols])

    # ---------------------------------Data Normalization with Standard Scaler------------------------------------
    scaler = jb.load(os.path.join(SCALER_PATH, "scaler.pkl"))
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # ---------------------------------Load Models------------------------------------
    model_predictor = tf.keras.models.load_model(MODEL_DIR)
    
    # def load_model(model_name = "LoanApprovalModel", alias = "ReadyForProduction", version = None):
    #     try:
    #         mlflow.set_tracking_uri(f"sqlite:///database/mlflow.db")
    #         mlflow.set_experiment("Loan_Prediction")
    # 
    #         # Load model
    #         client = mlflow.tracking.MlflowClient()
    #         if alias:
    #         
    #             version_info = client.get_model_version_by_alias(model_name, alias)
    #             version = version_info.version
    #             model_uri = f"models:/{model_name}@{alias}"
    #         else:
    #             if not version:
    #                 version = client.get_latest_versions(model_name)[0].version
    #                 
    #             model_uri = f"models:/{model_name}/{version}"
    # 
    #         try:
    #             model = mlflow.tensorflow.load_model(model_uri)
    #         except Exception as e:
    #             raise    
    #     except Exception as e:
    #         raise
    #     return model
    # 
    # model_predictor = load_model()
    
    #----------------------------------Make Prediction------------------------------------
    if st.button("✨ Get Prediction"):
        prediction_prob = model_predictor.predict(input_data)
        prediction = (prediction_prob > 0.5).astype(int)[0][0]
        
        if prediction == 1:
            st.text("")
            st.markdown("""
                            <h5>
                            <div class="content">
                                🚫 Approval Status:
                                <span class="prediction-text-denied">
                                Sorry, Your Request is Denied! 😢
                                </span>
                            </div>
                            </h5>
                        """, unsafe_allow_html=True)
        else:
            st.text("")
            st.markdown("""
                            <h5>
                            <div class="content">
                                🎉 Approval Status:
                                <span class="prediction-text-approved">
                                Congratulations! Your Loan is Approved! 💰😊
                                </span>
                            </div>
                            </h5>
                        """, unsafe_allow_html=True)
    
    st.markdown('<div class="content">Sample Data:  📝</div>', unsafe_allow_html=True)
    sample_display = st.empty()
    sample_display.dataframe(st.session_state.df_sample_tab1)
    if st.button("Refresh Sample"):
            approval_1_sample = approval_1.sample(3)
            approval_0_sample = approval_0.sample(3)
            st.session_state.df_sample_tab1 = pd.concat([approval_1_sample, approval_0_sample])
            sample_display.dataframe(st.session_state.df_sample_tab1)

# ----------------------------------Tab 3: Batch Prediction----------------------------------
with tab3:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📤 Batch Loan Approval Prediction</div>', unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    st.markdown('<div class="content">Required columns: person_age, person_income, person_emp_length, loan_grade, loan_amnt, loan_int_rate, loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length, person_home_ownership, loan_intent</div>', unsafe_allow_html=True)
    if uploaded_file is not None:
        # Load and validate the uploaded CSV
        uploaded_data = pd.read_csv(uploaded_file)
        df = uploaded_data.copy()
        required_columns = ['person_age', 'person_income', 'person_emp_length', 'loan_grade',
                           'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                           'cb_person_default_on_file', 'cb_person_cred_hist_length',
                           'person_home_ownership', 'loan_intent']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}. Please ensure all columns are present.")
        else:
            st.success("All required columns are present. Processing...")

            # Apply mappings
            df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'No': 0, 'Yes': 1})
            df['loan_grade'] = df['loan_grade'].map({"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6})

            # One-hot encoding
            df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent'], drop_first=True)

            # Ensure all expected one-hot columns are present (fill missing with 0)
            expected_columns = [
                'person_age', 'person_income', 'person_emp_length', 'loan_grade',
                'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                'cb_person_default_on_file', 'cb_person_cred_hist_length',
                'person_home_ownership_OWN', 'person_home_ownership_MORTGAGE',
                'person_home_ownership_OTHER', 'loan_intent_MEDICAL',
                'loan_intent_PERSONAL', 'loan_intent_VENTURE',
                'loan_intent_DEBTCONSOLIDATION', 'loan_intent_HOMEIMPROVEMENT'
            ]
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = 0

            # Reorder columns
            df = df[expected_columns]

            # Log transformation
            num_cols = PREPROCESSING_CONFIG['numerical_columns']
            df[num_cols] = np.log1p(df[num_cols])

            # Scaling
            scaler = jb.load(os.path.join(SCALER_PATH, "scaler.pkl"))
            df[num_cols] = scaler.transform(df[num_cols])

            # Make predictions
            predictions = model_predictor.predict(df[expected_columns])
            df['Loan Status'] = (predictions > 0.5).astype(int)

            # Display results
            st.markdown('<div class="content">Prediction Results: 📊</div>', unsafe_allow_html=True)
            prediction_df = pd.concat([uploaded_data, df[['Loan Status']]], axis=1)
            st.dataframe(prediction_df, height=300)

            # Download button
            csv = prediction_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="loan_predictions.csv",
                mime="text/csv"
            )
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------Tab 4: LLM Review Analysis----------------------------------
with tab4:
    
    REVIEWS_PATH = os.path.join(os.path.dirname(__file__), "data", "reviews.csv")
    
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💬 Analyze Customer Review with LLM</div>', unsafe_allow_html=True)

    user_feedback = st.text_area("Enter customer feedback for loan approval experience:")
    if st.button("Predict with LLM 🚀"):
        if not user_feedback:
            st.warning("⚠️ Please enter some feedback!")
        else:
            try:
                # Load LLM (assuming GROK_API_KEY is in config or environment)
                llm = langchain_groq.ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="openai/gpt-oss-20b")

                # Construct prompt
                prompt = f"""
                You are the Pro Loan Approval Model, an expert in loan approval prediction. Given the following user input data representing features for a loan application, predict whether the loan will be approved or denied:

                🔹 **User Input Data:** "{user_feedback}"

                🎯 **Your Task:**
                - Analyze the provided data features to predict the loan approval outcome.
                - Determine if the loan is approved or denied.
                - Provide a short, engaging explanation for your decision, highlighting key factors.

                📌 **Format your response as follows:**
                - **Prediction:** ("Loan Approved" or "Loan Denied")
                - **Reasoning:** A brief but engaging analysis explaining the decision.

                🚀 **Make it sound professional yet interesting!**
                """

                # Get LLM response
                response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
                response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

                # Parse LLM response
                if "Customer is dissatisfied" in response:
                    llm_prediction = "Dissatisfied"
                else:
                    llm_prediction = "Satisfied"

                # Save to CSV
                reviews_df = pd.DataFrame({
                    "Feedback": [user_feedback],
                    "Prediction": [llm_prediction],
                    "Reasoning": [response.split("**Reasoning:**")[-1].strip()]
                })
                if os.path.exists(REVIEWS_PATH):
                    reviews_df.to_csv(REVIEWS_PATH, mode='a', header=False, index=False)
                else:
                    reviews_df.to_csv(REVIEWS_PATH, mode='w', header=True, index=False)

                # Display result
                st.write(f"🔮 {response}")
                
            except Exception as e:
                st.error(f"❌ LLM Error: {str(e)}")
        
    st.dataframe(pd.read_csv(REVIEWS_PATH), width=1500)
    st.markdown('</div>', unsafe_allow_html=True)
        
# Footer
st.markdown("""
    <div class="footer">
        Developed by <a href="https://portfolio-sigma-mocha-67.vercel.app/" target="_blank">Muhammad Umer Khan</a>. Powered by Artificial Neural Network. 🧠
    </div>""", unsafe_allow_html=True)