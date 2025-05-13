# Multi-Disease Prediction and Health Recommendation System 🧑‍⚕️

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-red.svg)

Welcome to the **Multi-Disease Prediction and Health Recommendation System**, a web application that leverages machine learning to predict the likelihood of diseases such as Diabetes, Heart Disease, Parkinson's Disease, and Breast Cancer. Using trained models and user-provided medical data, the system not only predicts disease presence but also generates personalized health reports with actionable advice using the Hugging Face Inference API. Built with Streamlit, this tool is user-friendly and designed for educational purposes.

> **⚠️ Disclaimer**: This application is for educational purposes only. Predictions and reports are not definitive diagnoses. Always consult a medical professional for accurate diagnosis and treatment.

---

## 🌟 Features

- **Multi-Disease Prediction**:
  - Predicts Diabetes, Heart Disease, Parkinson's Disease, and Breast Cancer.
  - Uses pre-trained machine learning models (Logistic Regression, SVM) for accurate predictions.
- **Personalized Health Reports**:
  - Generates detailed health reports using Hugging Face's Mixtral-8x7B model.
  - Provides actionable steps, focus areas, and precautions for disease management, or healthy living tips if no disease is predicted.
- **Interactive Interface**:
  - Built with Streamlit for a seamless user experience.
  - Sidebar navigation for easy access to different prediction pages.
- **Robust Error Handling**:
  - Validates user inputs and handles API errors gracefully.
- **Timestamped Reports**:
  - Health reports include a timestamp for reference (e.g., "Generated on 04:22 PM IST, Tuesday, May 13, 2025").

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- A Hugging Face API token (for generating health reports)

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rohanmistry231/Multi-Disease-Prediction-and-Health-Recommendation-System.git
   cd Multi-Disease-Prediction-and-Health-Recommendation-System
   ```

2. **Install Dependencies**:
   Ensure you have the required packages installed. Use the provided `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` includes:
   - `streamlit`
   - `streamlit-option-menu`
   - `huggingface_hub`
   - `pandas`
   - `numpy`
   - `scikit-learn`

3. **Set Up Hugging Face API Token**:
   - Create a `.streamlit/secrets.toml` file in the project root.
   - Add your Hugging Face API token:
     ```toml
     HF_API_TOKEN = "your_huggingface_api_token"
     ```
   - Get your token from [Hugging Face](https://huggingface.co/settings/tokens).

4. **Download Pre-trained Models**:
   - The `saved_models/` directory should contain the following pre-trained models:
     - `diabetes_model.sav`
     - `heart_disease_model.sav`
     - `parkinsons_model.sav`
     - `breast_cancer_model.sav`
   - If these models are missing, you can train them using the notebooks in `colab_files_to_train_models/`. See the *Training Models* section for details.

### Running the Application

1. **Start the Streamlit App**:
   ```bash
   streamlit run app.py
   ```
   This will launch the app in your default web browser (typically at `http://localhost:8501`).

2. **Navigate the App**:
   - Use the sidebar to select a prediction page (e.g., "Diabetes Prediction").
   - Enter the required medical parameters in the input fields.
   - Click the "Test Result" button to view the prediction and health report.

---

## 📖 Usage

1. **Home Page**:
   - Overview of the application and instructions on how to use it.
   - Includes a disclaimer about the educational nature of the tool.

2. **Prediction Pages**:
   - **Diabetes Prediction**: Input parameters like Pregnancies, Glucose, BMI, etc.
   - **Heart Disease Prediction**: Input parameters like Age, Sex, Cholesterol, etc.
   - **Parkinson's Prediction**: Input vocal and motor-related metrics (e.g., MDVP:Fo(Hz), Jitter).
   - **Breast Cancer Prediction**: Input tumor metrics (e.g., Mean Radius, Worst Texture).
   - Each page provides a prediction (e.g., "The person is diabetic") and a generated health report.

3. **Health Reports**:
   - If a disease is predicted, the report includes:
     - A summary of the diagnosis.
     - **Actionable Steps**: Lifestyle changes or recommendations.
     - **Focus Areas**: Key health aspects to monitor.
     - **Precautions**: Steps to prevent worsening the condition.
   - If no disease is predicted, the report includes:
     - A confirmation of good health.
     - **Healthy Living Tips**: General wellness advice.
     - **Wellness Recommendations**: Practices for long-term health.

---

## 📂 Project Structure

```
multi-disease-prediction/
│
├── .streamlit/                   # Streamlit configuration (e.g., secrets.toml)
├── colab_files_to_train_models/  # Jupyter notebooks for training models
├── dataset/                      # Datasets used for training
├── saved_models/                 # Pre-trained model files
├── app.py                        # Main Streamlit application script
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
└── .gitignore                    # Git ignore file
```

---

## 🛠️ Training Models

The pre-trained models in `saved_models/` were trained using scikit-learn (version 1.0.2). To retrain or create new models:

1. **Access the Notebooks**:
   - Navigate to `colab_files_to_train_models/`.
   - Open the relevant notebook (e.g., `diabetes_model_training.ipynb`).

2. **Prepare the Data**:
   - Datasets are stored in `dataset/`.
   - Ensure you have the required datasets (e.g., `diabetes.csv`, `heart.csv`).

3. **Train the Models**:
   - The notebooks use scikit-learn models like `LogisticRegression` and `SVM`.
   - Example training process:
     ```python
     from sklearn.svm import SVC
     import pandas as pd
     import pickle

     # Load dataset
     data = pd.read_csv('dataset/diabetes.csv')
     X = data.drop('Outcome', axis=1)
     y = data['Outcome']

     # Train model
     model = SVC()
     model.fit(X, y)

     # Save model
     with open('saved_models/diabetes_model.sav', 'wb') as f:
         pickle.dump(model, f)
     ```
   - Repeat for each disease model.

4. **Update the App**:
   - Ensure the new models are placed in `saved_models/` and match the expected filenames.

> **Note**: The app currently uses scikit-learn 1.5.1, which may cause version mismatch warnings with models trained on 1.0.2. Retraining with 1.5.1 is recommended for compatibility.

---

## 📦 Dependencies

See `requirements.txt` for the full list of dependencies. Key packages include:
- `streamlit`: For the web interface.
- `streamlit-option-menu`: For sidebar navigation.
- `huggingface_hub`: For generating health reports via Hugging Face API.
- `pandas` & `numpy`: For data manipulation.
- `scikit-learn`: For machine learning models.

Install them using:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.