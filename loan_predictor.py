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
        .prediction-text.approved {
            color: #047857;
        }
        .prediction-text.denied {
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
                <li><span class="highlight">📊 Data Pipeline & Preprocessing</span>: Automated data ingestion, cleaning, and transformation with custom pipelines, ensuring data quality and handling outliers effectively.</li>
                <li><span class="highlight">🧠 Model Development</span>: Built an optimized Artificial Neural Network (ANN) model with hyperparameter tuning using Optuna, achieving high accuracy in loan approval predictions.</li>
                <li><span class="highlight">🔄 MLOps Integration</span>: Implemented MLflow for experiment tracking, model versioning, and deployment, enabling seamless transitions from development to production.</li>
                <li><span class="highlight">💬 LLM Review Analysis</span>: Integrated a Large Language Model (LLM) for sentiment analysis on customer feedback, providing actionable insights into user experiences.</li>
                <li><span class="highlight">🌐 Deployment</span>: Deployed the system using Streamlit for an interactive, user-friendly interface, supporting single and batch predictions.</li>
            </ul>
            This project demonstrates my ability to deliver end-to-end ML solutions, combining robust engineering with innovative AI techniques. 🌟
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">💻 Technologies & Tools</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            <ul>
                <li><span class="highlight">🔤 Languages & Libraries</span>: Python, Pandas, NumPy, TensorFlow, Scikit-Learn, Optuna, LangChain.</li>
                <li><span class="highlight">⚙️ Approaches</span>: Data Preprocessing, ANN Modeling, Hyperparameter Tuning, LLM Integration, MLOps with MLflow.</li>
                <li><span class="highlight">🌐 Deployment</span>: Streamlit for web-based interactive systems.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------Tab 2----------------------------------
with tab2:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Loan Approval Prediction</div>', unsafe_allow_html=True)
    
    # Input form
    with st.form("loan_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
            person_income = st.number_input("Income", min_value=0, value=50000)
            person_emp_length = st.number_input("Employment Length (years)", min_value=0, value=5)
        with col2:
            loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
            loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0)
            loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, value=0.2)
        with col3:
            cb_person_default_on_file = st.selectbox("Default on File", ["N", "Y"])
            cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=5)
            person_home_ownership = st.selectbox("Home Ownership", ["OWN", "MORTGAGE", "RENT", "OTHER"])
            loan_intent = st.selectbox("Loan Intent", ["MEDICAL", "PERSONAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT", "EDUCATION"])
        
        submitted = st.form_submit_button("Predict Loan Approval 🚀")
        
        if submitted:
            # Prepare input data
            input_data = {
                'person_age': person_age,
                'person_income': person_income,
                'person_emp_length': person_emp_length,
                'loan_amnt': loan_amnt,
                'loan_int_rate': loan_int_rate,
                'loan_percent_income': loan_percent_income,
                'cb_person_default_on_file': 1 if cb_person_default_on_file == "Y" else 0,
                'cb_person_cred_hist_length': cb_person_cred_hist_length,
                'person_home_ownership_OWN': 1 if person_home_ownership == "OWN" else 0,
                'person_home_ownership_MORTGAGE': 1 if person_home_ownership == "MORTGAGE" else 0,
                'person_home_ownership_OTHER': 1 if person_home_ownership == "OTHER" else 0,
                'loan_intent_MEDICAL': 1 if loan_intent == "MEDICAL" else 0,
                'loan_intent_PERSONAL': 1 if loan_intent == "PERSONAL" else 0,
                'loan_intent_VENTURE': 1 if loan_intent == "VENTURE" else 0,
                'loan_intent_DEBTCONSOLIDATION': 1 if loan_intent == "DEBTCONSOLIDATION" else 0,
                'loan_intent_HOMEIMPROVEMENT': 1 if loan_intent == "HOMEIMPROVEMENT" else 0,
            }
            df = pd.DataFrame([input_data])

            # Preprocess
            num_cols = PREPROCESSING_CONFIG['numerical_columns']
            df[num_cols] = np.log1p(df[num_cols])
            scaler = jb.load(os.path.join(SCALER_PATH, "scaler.pkl"))
            df[num_cols] = scaler.transform(df[num_cols])

            # Load model
            model_predictor = tf.keras.models.load_model(os.path.join(MODEL_DIR, "loan_approval_ann_model.h5"))

            # Predict
            prediction = model_predictor.predict(df)[0][0]
            status = "Approved" if prediction > 0.5 else "Denied"
            status_class = "approved" if status == "Approved" else "denied"
            st.markdown(f'''
                <div class="prediction-text {status_class}">
                    Loan {status} 📊
                </div>
            ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------Tab 3: Batch Prediction----------------------------------
with tab3:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📤 Batch Loan Approval Prediction</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV for Batch Prediction", type="csv")
    
    if uploaded_file is not None:
        uploaded_data = pd.read_csv(uploaded_file)
        st.markdown('<div class="content">Uploaded Data Preview: 📊</div>', unsafe_allow_html=True)
        st.dataframe(uploaded_data.head(), height=200)

        if st.button("Predict Batch 🚀"):
            df = uploaded_data.copy()
            expected_columns = [
                'person_age', 'person_income', 'person_emp_length',
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

            # Load model
            model_predictor = tf.keras.models.load_model(os.path.join(MODEL_DIR, "loan_approval_ann_model.h5"))

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
                llm = langchain_groq.ChatGroq(groq_api_key=os.getenv("GROK_API_KEY"), model_name="qwen-qwq-32b")

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