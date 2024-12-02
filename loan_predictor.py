import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb
import tensorflow as tf

# ----------------------------------Streamlit page configuration-----------------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)
# ----------------------------------Custom CSS for styling-----------------------------------------
st.markdown("""
    <style>
        /* Main Title */
        .main-title {
            font-size: 2.5em;
            font-weight: bold;
            color: #2C3E50;
            text-align: center;
            margin-bottom: 20px;
        }
        /* Section Titles */
        .section-title {
            font-size: 1.8em;
            color: #3498DB;
            font-weight: bold;
            margin-top: 30px;
            text-align: left;
        }
        /* System Content */
        .system-content {
            font-size: 1.8em;
            color: #3498DB;
            font-weight: bold;
            margin-top: 30px;
            text-align: center;
        }
        /* Section Content */
        .section-content{
            text-align: center;
        }
        /* Home Page Content */
        .intro-title {
            font-size: 2.5em;
            color: #2C3E50;
            font-weight: bold;
            text-align: center;
        }
        .intro-subtitle {
            font-size: 1.2em;
            color: #34495E;
            text-align: center;
        }
        .content {
            font-size: 1em;
            color: #7F8C8D;
            text-align: justify;
            line-height: 1.6;
        }
        .highlight {
            color: #2E86C1;
            font-weight: bold;
        }
        /* Recommendation Titles and Descriptions */
        .recommendation-title {
            font-size: 22px;
            color: #2980B9;
        }
        .recommendation-desc {
            font-size: 16px;
            color: #7F8C8D;
        }
        /* Separator Line */
        .separator {
            margin-top: 10px;
            margin-bottom: 10px;
            border-top: 1px solid #BDC3C7;
        }
        /* Footer */
        .footer {
            font-size: 14px;
            color: #95A5A6;
            margin-top: 20px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)
# ----------------------------------Load the sample data----------------------------------
data = pd.read_csv("./Data/playground-series-s4e10/train.csv", index_col='id')

approval_1 = data[data['loan_status'] == 1]
approval_0 = data[data['loan_status'] == 0]

if "df_sample_tab1" not in st.session_state:
    approval_1_sample = approval_1.sample(4)
    approval_0_sample = approval_0.sample(4)
    st.session_state.df_sample_tab1 = pd.concat([approval_1_sample, approval_0_sample])


# ----------------------------------Page Title-----------------------------------
st.markdown('<div class="intro-title">📊 Welcome to the Loan Approval Predictor 📊</div>', unsafe_allow_html=True)
st.markdown('<div class="intro-subtitle">Your one-stop solution for loan approval predictions! 🚀</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🏠 Home", "📋 Get Loan Approval"])

# ----------------------------------Tab 1----------------------------------
with tab1:
    st.markdown('<div class="system-content">👋 About Me</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            Hi! I’m <span class="highlight">Muhammad Umer Khan</span>, an aspiring Data Scientist passionate about 
            <span class="highlight">🤖 Natural Language Processing (NLP)</span> and 🧠 Machine Learning. 
            Currently pursuing my Bachelor’s in Computer Science, I bring hands-on experience in developing intelligent recommendation systems, 
            performing data analysis, and building machine learning models. 🚀
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🎯 Project Overview</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            This project is a comprehensive loan approval prediction system that demonstrates my expertise in machine learning, data preprocessing, and deployment. Here's what I've achieved:
            <ul>
                <li><span class="highlight">📊 Exploratory Data Analysis (EDA)</span>: Analyzed the dataset to uncover insights and patterns, ensuring data quality and feature selection.</li>
                <li><span class="highlight">🛠 Data Preprocessing</span>: Cleaned, transformed, and encoded features to prepare the dataset for model training.</li>
                <li><span class="highlight">🔗 Model Development</span>: Built an Artificial Neural Network (ANN) to classify loan applications into approved or denied categories.</li>
                <li><span class="highlight">⚙️ Model Optimization</span>: Tuned hyperparameters to improve performance metrics like accuracy, precision, recall, and F1-score.</li>
                <li><span class="highlight">📈 Evaluation</span>: Achieved excellent classification results with robust evaluation metrics, ensuring reliability of predictions.</li>
                <li><span class="highlight">🌐 Deployment</span>: Developed an interactive and user-friendly Streamlit app, enabling seamless loan predictions for users.</li>
                <li><span class="highlight">🧩 Sequential Workflow</span>: Designed a modular codebase to ensure scalability and adaptability for future enhancements.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📂 Data Overview</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            The
            <a href="https://www.kaggle.com/competitions/playground-series-s4e10" target="_blank" style="color: #2980B9;">Dataset</a>
            used in this project contains key attributes for loan approval prediction. Here's a summary:
            <ul>
                <li><span class="highlight">📜 Features</span>: Includes attributes such as age, income, home ownership, employment length, loan amount, interest rate, and credit history.</li>
                <li><span class="highlight">⚖️ Class Balance</span>: Balanced classes to ensure fair model evaluation.</li>
                <li><span class="highlight">🔍 Feature Engineering</span>: Derived additional features like loan-to-income ratio for enhanced prediction accuracy.</li>
                <li><span class="highlight">📊 Insights</span>: 
                    <ul>
                        <li>Higher incomes are positively correlated with loan approvals.</li>
                        <li>Employment stability (length) impacts loan decisions significantly.</li>
                        <li>High-interest rates are associated with a greater chance of loan denial.</li>
                    </ul>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">💻 Technologies & Tools</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            <ul>
                <li><span class="highlight">🔤 Languages & Libraries</span>: Python, Pandas, Scikit-Learn, TensorFlow/Keras, Matplotlib, Seaborn.</li>
                <li><span class="highlight">⚙️ Methods</span>: Feature Engineering, Artificial Neural Networks (ANN), and Hyperparameter Tuning</li>
                <li><span class="highlight">🌐 Deployment</span>: Streamlit for web-based interactive systems</li>
                <li><span class="highlight">📊 Visualization Tools</span>: Seaborn and Matplotlib for EDA and performance insights</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🌟 Why This Project?</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            This project highlights my ability to build end-to-end solutions, from data analysis and model development to deployment. 
            By addressing a real-world use case of loan approval prediction, it showcases my technical skills and understanding of user needs. 
            I aim to make data-driven decision-making accessible and impactful. ✨
        </div>
    """, unsafe_allow_html=True)
# ----------------------------------Tab 2----------------------------------
with tab2:
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
        st.table(user_data)
    # ---------------------------------Apply Mappings------------------------------------
    default_mapping = {'No': 0, 'Yes': 1}
    loan_grade_mapping = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6
    }

    # Apply mappings
    cb_person_default_on_file = default_mapping[cb_person_default_on_file_input]
    loan_grade = loan_grade_mapping[loan_grade_input]

    # ---------------------------------One-Hot Encoding manually based on inputs-----------------------------------
    # Initialize one-hot columns with 0
    home_ownership = ['OWN', 'MORTGAGE', 'RENT', 'OTHER']
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
        **loan_intent_encoded       # Add one-hot encoded columns for loan intent
    })
    input_data.drop(columns=['person_home_ownership_MORTGAGE', 'loan_intent_DEBTCONSOLIDATION'], inplace=True)
    rearranged = ['person_age', 'person_income', 'person_emp_length', 'loan_grade',
       'loan_amnt', 'loan_int_rate', 'loan_percent_income',
       'cb_person_default_on_file', 'cb_person_cred_hist_length',
       'person_home_ownership_OTHER', 'person_home_ownership_OWN',
       'person_home_ownership_RENT', 'loan_intent_EDUCATION',
       'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
       'loan_intent_PERSONAL', 'loan_intent_VENTURE']

    input_data = input_data[rearranged]

    # ---------------------------------Load Models------------------------------------
    model_predictor = tf.keras.models.load_model('./Model/loan_approval_model.h5')
    scaler = jb.load('./Model/scaler.pkl')

    # ---------------------------------Log Transformation for Numerical Features------------------------------------
    num_cols = ['person_age',
                'person_income',
                'person_emp_length',
                'loan_amnt',
                'loan_int_rate',
                'loan_percent_income',
                'cb_person_cred_hist_length']
    # # Apply log transformation (same as during training)
    input_data[num_cols] = np.log1p(input_data[num_cols])

    # # ---------------------------------Data Normalization with Standard Scaler------------------------------------
    input_data[num_cols] = scaler.transform(input_data[num_cols])
    st.table(input_data)
    #----------------------------------Make Prediction------------------------------------
    if st.button("✨ Get Prediction"):
        prediction_prob = model_predictor.predict(input_data)
        prediction = (prediction_prob > 0.5).astype(int)
        prediction = prediction[0][0]
        
        if prediction == 1:
            
            st.text("")
            st.markdown("""
                            <h5>
                            <div class="content">
                                🚫 Approval Status:
                                <span class="highlight">
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
                                <span class="highlight">
                                Congratulations! Your Loan is Approved! 💰😊
                                </span>
                            </div>
                            </h5>
                        """, unsafe_allow_html=True)

    sample_display = st.empty()
    sample_display.dataframe(st.session_state.df_sample_tab1)
    if st.button("Refresh Sample"):
            approval_1_sample = approval_1.sample(3)
            approval_0_sample = approval_0.sample(3)
            st.session_state.df_sample_tab1 = pd.concat([approval_1_sample, approval_0_sample])
            sample_display.dataframe(st.session_state.df_sample_tab1)
# Footer
st.markdown("""
    <div class="footer">
        Developed by <a href="https://portfolio-sigma-mocha-67.vercel.app/" target="_blank" style="color: #2980B9;">Muhammad Umer Khan</a>. Powered by Artificial Neural Network. 🧠
    </div>""", unsafe_allow_html=True)