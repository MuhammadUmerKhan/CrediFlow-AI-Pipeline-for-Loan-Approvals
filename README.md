# 📊 CrediFlow: AI Pipeline for Loan Approvals 🎉
![Loan Approval](https://www.idfcfirstbank.com/content/dam/idfcfirstbank/images/blog/personal-loan/how-to-apply-for-firstmoney-personal-loan-a-step-by-step-guide-717X404.jpg)

Welcome to the **CrediFlow: Loan Approval Predictions**, a robust end-to-end MLOps project designed to predict loan approval outcomes using machine learning and natural language processing (NLP). This project demonstrates expertise in **data ingestion**, **preprocessing**, **model training**, **evaluation**, **registry**, and **interactive deployment** using **Streamlit** and **MLflow**. 🚀

---

## 🌟 Project Highlights

### 🎯 Key Features
- 🚀 **End-to-End MLOps Pipeline**:
  - 📂 **Data Ingestion**: Loaded and validated datasets from CSV files.
  - 📊 **Exploratory Data Analysis (EDA)**: Uncovered insights and patterns.
  - 🛠 **Data Preprocessing**: Cleaned, transformed, and balanced the dataset.
  - 🎨 **Feature Engineering**: Enhanced model accuracy with derived features.
  - 🤖 **Model Training**: Built an Artificial Neural Network (ANN) for binary classification.
  - 📈 **Evaluation**: Assessed model performance with metrics like accuracy, AUC, precision, and recall.
  - 📦 **Model Registry**: Registered models using MLflow for versioning and tracking.
  - 🌐 **Deployment**: Deployed an interactive Streamlit app for real-time predictions.
- 🧠 **Artificial Neural Network (ANN)**: Designed a TensorFlow ANN with dropout layers for robust loan approval predictions.
- 💬 **LLM Integration**: Leveraged a Language Model (e.g., Qwen via Grok API) for loan approval predictions and customer review analysis.
- 🌐 **Interactive User Interface**:
  - 🖥 A user-friendly Streamlit app with tabs for individual predictions, batch processing, and LLM analysis.
  - 📝 Real-time predictions and customer feedback storage in `reviews.csv`.

---

## 🛠 Technologies and Tools

- 🐍 **Programming Languages**: Python
- 📚 **Libraries**:
  - Data Handling: Pandas, NumPy
  - Machine Learning: Scikit-learn, TensorFlow/Keras, Imbalanced-learn (SMOTEENN)
  - Visualization: Matplotlib, Seaborn
  - MLOps: MLflow
  - Deployment: Streamlit
  - NLP: LangChain, Grok API (Mixtral-8x7B)
- 🌐 **Deployment Platform**: Streamlit (local or cloud-hosted)
- 🧠 **Machine Learning Techniques**:
  - Neural Networks (ANN) with dropout and batch normalization
  - Feature Engineering (e.g., loan-to-income ratio)
  - Data Balancing with SMOTEENN
  - Hyperparameter Tuning
- 📊 **Visualization Tools**: Matplotlib and Seaborn for EDA insights
- 📦 **Model Registry**: MLflow for tracking experiments and models

---

## 📊 Data Overview

The dataset, sourced from [Kaggle Playground Series S4E10](https://www.kaggle.com/competitions/playground-series-s4e10), contains features related to loan applications:
- 👤 **Applicant Information**: Age (`person_age`), income (`person_income`), employment length (`person_emp_length`), credit history length (`cb_person_cred_hist_length`).
- 💰 **Loan Details**: Amount (`loan_amnt`), interest rate (`loan_int_rate`), intent (`loan_intent`), percent of income (`loan_percent_income`), grade (`loan_grade`).
- 📜 **Other Factors**: Homeownership status (`person_home_ownership`), historical default status (`cb_person_default_on_file`).
- 🎯 **Target Variable**: `loan_status` (0: Approved, 1: Denied).

### Key Data Characteristics:
- 🧮 **Shape**: Train dataset has 58,645 rows and 12 columns.
- 🔑 **Notable Features**:
  - Higher incomes correlate with loan approvals.
  - Employment stability (length) significantly impacts decisions.
  - High interest rates increase denial chances.

---

## ⚙️ MLOps Pipeline Workflow

### 1. **Data Ingestion** 📂
- **Script**: `data_injestion.py`
- **Process**:
  - Loaded `train.csv` and `test.csv` from `data/` directory.
  - Validated file existence and logged errors.
- **Output**: Train and test DataFrames with `id` as the index.

### 2. **Exploratory Data Analysis (EDA)** 📊
- **Performed**:
  - Identified trends (e.g., income vs. approval correlation).
  - Highlighted feature relationships using visualizations.
- **Visuals**:
  - Categorical features distribution.
  - Loan status distribution.
  - Loan intent analysis.

### 3. **Data Preprocessing** 🛠
- **Script**: `preprocess.py`
- **Process**:
  - **Categorical Mapping**:
    - `cb_person_default_on_file`: Y → 1, N → 0
    - `loan_grade`: A → 0, B → 1, ..., G → 6
  - **One-Hot Encoding**: Applied to `person_home_ownership` and `loan_intent` with `drop_first=True`.
  - **Log Transformation**: Applied to numerical features (`loan_amnt`, `loan_int_rate`, `person_income`, `person_age`, `person_emp_length`).
  - **Data Balancing**: Used SMOTEENN to handle class imbalance.
  - **Train-Validation Split**: Split data with 90% train, 10% validation (stratified).
  - **Scaling**: Standardized numerical features using `StandardScaler`, saved as `scaler.pkl`.
- **Output**: Saved preprocessed files (`X_train_scaled.csv`, `X_val_scaled.csv`, `y_train.csv`, `y_val.csv`, `test_data_scaled.csv`) in `data/`.

### 4. **Model Building and Training** 🤖
- **Script**: `train.py`
- **Process**:
  - Built a TensorFlow ANN with configurable layers (from `config.py`):
    - 4 dense layers: 128, 64, 32, 16 units with ReLU activation.
    - Dropout layers (0.2) for regularization.
    - Output layer: 1 unit with sigmoid activation.
  - Compiled with Adam optimizer, binary cross-entropy loss, and accuracy metric.
  - Trained for 20 epochs with a batch size of 64.
  - Logged hyperparameters, metrics (e.g., val_loss, val_accuracy), and history to MLflow.
  - Saved the model as `models/loan_approval_model.keras`.
- **Output**: Trained model and MLflow logs.

### 5. **Model Evaluation** 📈
- **Script**: `evaluate.py`
- **Process**:
  - Evaluated on validation data with metrics:
    - Loss, Accuracy, AUC, Precision, Recall.
  - Generated predictions on test data.
  - Logged confusion matrix and test predictions (`test_predictions.csv`) to MLflow.
- **Results**:
  - Achieved ~94% accuracy and F1-score (based on previous metrics).
  - Insights: Stable employment and low loan-to-income ratios improve approval chances.

### 6. **Model Registry** 📦
- **Script**: `register_model.py`
- **Process**:
  - Registered the trained model in MLflow with the name `LoanApprovalModel`.
  - Assigned alias `ReadyForProduction`.
  - Added tags (e.g., dataset: "Loan Dataset", model_type: "TensorFlow").
  - Included a description: "TensorFlow neural network for binary classification to predict loan approval outcomes."
- **Output**: Registered model version in MLflow.

### 7. **Pipeline Orchestration** 🎯
- **Script**: `pipeline.py`
- **Process**:
  - Orchestrated the entire workflow: preprocessing → training → registration → evaluation.
  - Logged each step and exceptions to `logs/app.log`.
- **Output**: Pipeline results with model version and metrics.

### 8. **Deployment and Interaction** 🌐
- **Script**: `loan_predictor.py`
- **Process**:
  - Deployed a Streamlit app with four tabs:
    1. 🏠 **Home**: Project overview and developer info.
    2. 📋 **Get Loan Approval**: Input form for individual predictions using the ANN.
    3. 📤 **Batch Prediction**: Upload CSV for bulk predictions.
    4. 💬 **LLM Analysis**:
       - Loan approval prediction using LLM (Mixtral-8x7B).
       - Customer review sentiment analysis (satisfied/dissatisfied).
  - Stored LLM predictions and reviews in `reviews.csv` with timestamps.
  - Integrated error handling and logging.
- **Output**: Interactive web app for users.

---

## 🖼 App Features
- 📝 **Input Form**: Enter details like age, income, loan amount, etc.
- 🔮 **Dynamic Predictions**: Real-time loan approval predictions using ANN or LLM.
- 📈 **Batch Processing**: Upload CSV files for bulk predictions with downloadable results.
- 💬 **LLM Insights**:
  - Predicts loan approval with explanations.
  - Analyzes customer feedback sentiment.
- 📊 **Sample Data**: Refreshable sample data for reference.

---

## 📈 Data Insights

Explore key insights from the dataset:
- 👷 Applicants with stable employment and lower loan-to-income ratios are more likely to get approved.
- 🚩 A history of defaults significantly reduces approval chances.
- 📉 High interest rates correlate with higher denial rates.

| Feature                                      | Visualization                                                                                       |
|----------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Categorical Features                         | ![Categorical Features](https://github.com/MuhammadUmerKhan/Customer-Loan-Approval-KAGGLE-COMPETITION/blob/main/pics/Categorical.png)   |
| Loan Status Target Variable                  | ![Loan Status](https://github.com/MuhammadUmerKhan/AI-Driven-Loan-Approval-Prediction-System/blob/main/pics/Loan%20Status%20Distribution.png) |
| Customer Information                         | ![Customer Information](https://github.com/MuhammadUmerKhan/AI-Driven-Loan-Approval-Prediction-System/blob/main/pics/Average%20Loan%20Amount%20by%20Loan%20Status.png)   |
| Distribution Analysis                        | ![Distribution Analysis](https://github.com/MuhammadUmerKhan/AI-Driven-Loan-Approval-Prediction-System/blob/main/pics/Loan%20Status%20vs.%20Home%20Ownership.png)   |

For more visuals, check the [notebook](https://github.com/MuhammadUmerKhan/Customer-Loan-Approval-KAGGLE-COMPETITION/blob/main/ipynbs/Loan_Approval.ipynb).

---

## 🔑 Key Results

### Model Performance:
- ✅ **Accuracy**: ~94%
- 📊 **F1-score**: ~94%
- 📈 **AUC, Precision, Recall**: High scores logged in MLflow.

### Insights:
- 💡 Stable employment and low loan-to-income ratios are critical for approval.
- 🚫 Default history negatively impacts approval likelihood.

---

## 🌟 Why This Project?

This project showcases:
- 💡 Mastery of the MLOps pipeline from data ingestion to deployment.
- 🧑‍💻 Proficiency in building and deploying user-centric ML solutions.
- 🤖 Integration of advanced techniques like ANN, SMOTEENN, and LLM for enhanced decision-making.

---

## 🚀 How to Run the Project

### Prerequisites
1. **Install Python 3.x** and required packages:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure `requirements.txt` includes:
   ```
   streamlit
   tensorflow
   pandas
   numpy
   scikit-learn
   imbalanced-learn
   joblib
   langchain-groq
   mlflow
   ```

2. **Set Up Directory Structure**:
   - Place `train.csv` and `test.csv` in `data/`.
   - Create `models/`, `logs/`, `database/`, and `artifacts/` directories.

3. **Run the MLflow UI** (from root folder):
   ```bash
   mlflow ui --backend-store-uri sqlite:///database/mlflow.db
   ```
   Access at `http://localhost:5000` to view experiments and models.

4. **Run the Pipeline** (from `scripts/` folder):
   ```bash
   cd scripts
   python pipeline.py
   ```
   This executes preprocessing, training, registration, and evaluation.

5. **Run the Streamlit App** (from `scripts/` folder):
   ```bash
   streamlit run loan_predictor.py
   ```
   Access at `http://localhost:8501` for the interactive app.

---

## 📝 Logging
- All scripts log to `logs/app.log`.
- Monitor logs in real-time: `tail -f logs/app.log`.

---
## 🐳 **Dockerization & Deployment**

You can easily run this project using Docker and share or deploy it from Docker Hub.

### ✅ **Build the Docker Image**

Make sure your `Dockerfile` is correctly set up. Then run:

```bash
docker build -t muhammadumerkhan/loan-predictor .
```

### 🚀 **Run the Docker Container**

```bash
docker run -p 8501:8501 muhammadumerkhan/loan-predictor
```

> This will launch the Streamlit/ FastAPI interface on `http://localhost:8501` depending on your app entrypoint.

### 📤 **Push to Docker Hub**

First, log in to Docker:

```bash
docker login
```

Then push your image:

```bash
docker push muhammadumerkhan/loan-predictor
```

### 📥 **Pull & Run from Docker Hub**

Anyone can pull and run the app using:

```bash
docker pull muhammadumerkhan/loan-predictor
docker run -p 8501:8501 muhammadumerkhan/loan-predictor
```
---

## 🔧 Configuration
- Edit `scripts/config.py` to adjust paths, model layers, or hyperparameters.
- Set `GROK_API_KEY` in `config.py` for LLM features.

---

## 🌟 Future Enhancements
- **Deploy**: Host the Streamlit app on a cloud platform (e.g., Streamlit Community Cloud).
- **Monitor**: Add model drift detection using MLflow.
- **Secure**: Implement user authentication for the app.

---

## 📧 Contact

For queries or collaboration, reach out:
- 📛 Name: [Muhammad Umer Khan](https://portfolio-sigma-mocha-67.vercel.app)
- 📧 Email: muhammadumerk546@gmail.com
- 🔗 LinkedIn: [Muhammad Umer Khan](https://linkedin.com/in/muhammad-umer-khan-61729b260/)

---

## 📄 Acknowledgments
- **Dataset**: [Kaggle Playground Series S4E10](https://www.kaggle.com/competitions/playground-series-s4e10)
- **LLM**: Powered by Grok API (xAI)

---

## 🔴 Live Demo:
- **[Click Here](https://customer-loan-approval.streamlit.app/)**

---

📝 **License**  
This project is licensed under the MIT License. See the `LICENSE` file for details.
