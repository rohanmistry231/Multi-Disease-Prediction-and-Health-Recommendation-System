import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from huggingface_hub import InferenceClient
from datetime import datetime
import pandas as pd
import warnings

# Suppress scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Getting the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Loading the saved models
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))
breast_cancer_model = pickle.load(open(f'{working_dir}/saved_models/breast_cancer_model.sav', 'rb'))

# Define feature names for each model
diabetes_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
heart_disease_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
parkinsons_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 
                       'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 
                       'spread1', 'spread2', 'D2', 'PPE']
breast_cancer_features = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness', 'mean_compactness', 'mean_concavity', 
                         'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension', 'radius_error', 'texture_error', 'perimeter_error', 
                         'area_error', 'smoothness_error', 'compactness_error', 'concavity_error', 'concave_points_error', 'symmetry_error', 
                         'fractal_dimension_error', 'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness', 
                         'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension']

# Function to call Hugging Face Inference API using InferenceClient
def generate_report_with_hf(patient_data, predicted_disease, api_token):
    generated_report = "Unable to generate report due to an unexpected error. Please try again later."
    try:
        client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=api_token)
        has_disease = "not" not in predicted_disease.lower()
        
        if has_disease:
            prompt = f"""
            Based on the following patient data: {patient_data}, the predicted condition is: {predicted_disease}.
            Generate a concise health report (100-150 words) in a professional tone. Structure the report as follows:
            - A brief introduction summarizing the diagnosis (1-2 sentences).
            - A section titled 'Actionable Steps' with 3-4 bullet points listing specific lifestyle changes or recommendations.
            - A section titled 'Focus Areas' with 2-3 bullet points highlighting key areas to monitor or prioritize.
            - A section titled 'Precautions' with 3-4 bullet points listing specific preventive measures to avoid worsening the condition.
            Ensure each section is separated by an empty line for clear readability. Use markdown formatting: section titles with **bold** (e.g., **Actionable Steps**), and bullet points with `-` (e.g., - Item). Start each bullet point with an action verb (e.g., Monitor, Avoid, Engage) for clarity. Ensure all sentences are complete, advice is actionable, and the tone is supportive.
            """
        else:
            prompt = f"""
            Based on the following patient data: {patient_data}, the predicted condition is: {predicted_disease}.
            Generate a concise health report (100-150 words) in a professional tone to help the person maintain and improve their overall health. Structure the report as follows:
            - A brief introduction confirming the absence of the condition and emphasizing the importance of maintaining good health (1-2 sentences).
            - A section titled 'Healthy Living Tips' with 3-4 bullet points listing general lifestyle recommendations to enhance overall health.
            - A section titled 'Wellness Recommendations' with 3-4 bullet points suggesting practices to promote long-term well-being.
            Ensure each section is separated by an empty line for clear readability. Use markdown formatting: section titles with **bold** (e.g., **Healthy Living Tips**), and bullet points with `-` (e.g., - Item). Start each bullet point with an action verb (e.g., Maintain, Incorporate, Ensure) for clarity. Ensure all sentences are complete, advice is practical, and the tone is supportive.
            """
        
        response = client.text_generation(prompt, max_new_tokens=250, temperature=0.7)
        generated_report = response.split(prompt)[-1].strip() if prompt in response else response.strip()
        
        words = generated_report.split()
        if len(words) > 150:
            generated_report = " ".join(words[:150]) + "..."

        if generated_report and generated_report[-1] not in ".!?":
            last_punctuation = max(generated_report.rfind("."), generated_report.rfind("!"), generated_report.rfind("?"))
            if last_punctuation != -1:
                generated_report = generated_report[:last_punctuation + 1]

        return generated_report
    
    except Exception as e:
        error_message = "Unable to generate report. "
        if "rate limit" in str(e).lower():
            error_message += "Hugging Face API rate limit exceeded. Please wait and try again."
        elif "token" in str(e).lower() or "authentication" in str(e).lower():
            error_message += "Invalid or expired API token. Please check your HF_API_TOKEN in secrets.toml."
        else:
            error_message += f"An error occurred: {str(e)}. Please try again later."
        return error_message

# Function to parse and display the report
def display_health_report(report):
    lines = [line.strip() for line in report.split("\n") if line.strip()]
    current_section = None
    section_items = []
    introduction = ""

    for line in lines:
        if line.startswith("**") and line.endswith("**"):
            if current_section and section_items:
                st.subheader(current_section)
                for item in section_items:
                    st.write(f"• {item[2:]}")
            current_section = line.replace("**", "").strip()
            section_items = []
        elif line.startswith("- "):
            section_items.append(line)
        else:
            if not current_section:
                introduction = line

    if introduction:
        st.text(introduction)

    if current_section and section_items:
        st.subheader(current_section)
        for item in section_items:
            st.write(f"• {item[2:]}")

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multi-Disease Prediction and Health Recommendation System',
                           ['Home', 'Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Breast Cancer Prediction'],
                           menu_icon='hospital-fill',
                           icons=['house', 'activity', 'heart', 'person', 'bi-virus2'],
                           default_index=0)

# Home Page
if selected == 'Home':
    st.title('Welcome to the Multi-Disease Prediction and Health Recommendation System 🧑‍⚕️')
    st.markdown("""
    ### About This Application
    This application uses machine learning models to predict the likelihood of four medical conditions:
    - **Diabetes**
    - **Heart Disease**
    - **Parkinson's Disease**
    - **Breast Cancer**

    The predictions are based on trained models that analyze input data provided by the user. Each prediction page allows you to input relevant medical parameters, and the system will provide a model-based prediction and a generated report with actionable advice.

    ### How to Use This Application
    1. **Navigate to a Prediction Page**: Use the sidebar to select a disease prediction option.
    2. **Enter Required Data**: Input numerical medical parameters in the provided fields.
    3. **Get Prediction and Report**: Click the "Test Result" button to see the prediction and a detailed report.
    4. **Interpret Results**: The prediction indicates the presence or absence of the condition, and the report provides management tips.

    ### Important Disclaimer
    **This application is for educational purposes only.** Predictions and reports are not definitive diagnoses. Consult a medical professional for accurate diagnosis and treatment. The developers are not responsible for decisions based on this app.

    ### Get Started
    Select a prediction page from the sidebar!
    """)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        try:
            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            user_input = [float(x) for x in user_input]
            input_df = pd.DataFrame([user_input], columns=diabetes_features)
            diab_prediction = diabetes_model.predict(input_df)
            diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
            st.success(diab_diagnosis)

            patient_data = f"Pregnancies: {Pregnancies}, Glucose: {Glucose}, Blood Pressure: {BloodPressure}, " \
                          f"Skin Thickness: {SkinThickness}, Insulin: {Insulin}, BMI: {BMI}, " \
                          f"Diabetes Pedigree: {DiabetesPedigreeFunction}, Age: {Age}"
            
            api_token = st.secrets.get("HF_API_TOKEN", "")
            if api_token:
                with st.spinner("Generating report..."):
                    report = generate_report_with_hf(patient_data, diab_diagnosis, api_token)
                st.subheader("Health Report")
                current_time = datetime.now().strftime("%I:%M %p IST, %A, %B %d, %Y")
                st.caption(f"Generated on {current_time}")
                st.divider()
                display_health_report(report)
            else:
                st.error("Hugging Face API token not found. Add it to secrets.toml.")
        except ValueError:
            st.error("Please enter valid numerical values for all fields.")

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex')
    with col3:
        cp = st.text_input('Chest Pain types')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        try:
            user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            user_input = [float(x) for x in user_input]
            input_df = pd.DataFrame([user_input], columns=heart_disease_features)
            heart_prediction = heart_disease_model.predict(input_df)
            heart_diagnosis = 'The person is having heart disease' if heart_prediction[0] == 1 else 'The person does not have any heart disease'
            st.success(heart_diagnosis)

            patient_data = f"Age: {age}, Sex: {sex}, Chest Pain: {cp}, Resting BP: {trestbps}, Cholesterol: {chol}, " \
                          f"Fasting Blood Sugar: {fbs}, Resting ECG: {restecg}, Max Heart Rate: {thalach}, " \
                          f"Exercise Angina: {exang}, Oldpeak: {oldpeak}, Slope: {slope}, Major Vessels: {ca}, Thal: {thal}"
            
            api_token = st.secrets.get("HF_API_TOKEN", "")
            if api_token:
                with st.spinner("Generating report..."):
                    report = generate_report_with_hf(patient_data, heart_diagnosis, api_token)
                st.subheader("Health Report")
                current_time = datetime.now().strftime("%I:%M %p IST, %A, %B %d, %Y")
                st.caption(f"Generated on {current_time}")
                st.divider()
                display_health_report(report)
            else:
                st.error("Hugging Face API token not found. Add it to secrets.toml.")
        except ValueError:
            st.error("Please enter valid numerical values for all fields.")

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    with col1:
        RAP = st.text_input('MDVP:RAP')
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
    with col3:
        DDP = st.text_input('Jitter:DDP')
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
    with col3:
        APQ = st.text_input('MDVP:APQ')
    with col4:
        DDA = st.text_input('Shimmer:DDA')
    with col5:
        NHR = st.text_input('NHR')
    with col1:
        HNR = st.text_input('HNR')
    with col2:
        RPDE = st.text_input('RPDE')
    with col3:
        DFA = st.text_input('DFA')
    with col4:
        spread1 = st.text_input('spread1')
    with col5:
        spread2 = st.text_input('spread2')
    with col1:
        D2 = st.text_input('D2')
    with col2:
        PPE = st.text_input('PPE')

    parkinsons_diagnosis = ''
    if st.button("Parkinson's Test Result"):
        try:
            user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                          APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
            user_input = [float(x) for x in user_input]
            input_df = pd.DataFrame([user_input], columns=parkinsons_features)
            parkinsons_prediction = parkinsons_model.predict(input_df)
            parkinsons_diagnosis = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"
            st.success(parkinsons_diagnosis)

            patient_data = f"MDVP:Fo(Hz): {fo}, Fhi(Hz): {fhi}, Flo(Hz): {flo}, Jitter(%): {Jitter_percent}, " \
                          f"Jitter(Abs): {Jitter_Abs}, RAP: {RAP}, PPQ: {PPQ}, DDP: {DDP}, Shimmer: {Shimmer}, " \
                          f"Shimmer(dB): {Shimmer_dB}, APQ3: {APQ3}, APQ5: {APQ5}, APQ: {APQ}, DDA: {DDA}, " \
                          f"NHR: {NHR}, HNR: {HNR}, RPDE: {RPDE}, DFA: {DFA}, spread1: {spread1}, " \
                          f"spread2: {spread2}, D2: {D2}, PPE: {PPE}"
            
            api_token = st.secrets.get("HF_API_TOKEN", "")
            if api_token:
                with st.spinner("Generating report..."):
                    report = generate_report_with_hf(patient_data, parkinsons_diagnosis, api_token)
                st.subheader("Health Report")
                current_time = datetime.now().strftime("%I:%M %p IST, %A, %B %d, %Y")
                st.caption(f"Generated on {current_time}")
                st.divider()
                display_health_report(report)
            else:
                st.error("Hugging Face API token not found. Add it to secrets.toml.")
        except ValueError:
            st.error("Please enter valid numerical values for all fields.")

# Breast Cancer Prediction Page
if selected == 'Breast Cancer Prediction':
    st.title('Breast Cancer Prediction using ML')
    col1, col2, col3 = st.columns(3)

    with col1:
        mean_radius = st.text_input('Mean Radius')
    with col2:
        mean_texture = st.text_input('Mean Texture')
    with col3:
        mean_perimeter = st.text_input('Mean Perimeter')
    with col1:
        mean_area = st.text_input('Mean Area')
    with col2:
        mean_smoothness = st.text_input('Mean Smoothness')
    with col3:
        mean_compactness = st.text_input('Mean Compactness')
    with col1:
        mean_concavity = st.text_input('Mean Concavity')
    with col2:
        mean_concave_points = st.text_input('Mean Concave Points')
    with col3:
        mean_symmetry = st.text_input('Mean Symmetry')
    with col1:
        mean_fractal_dimension = st.text_input('Mean Fractal Dimension')
    with col2:
        radius_error = st.text_input('Radius Error')
    with col3:
        texture_error = st.text_input('Texture Error')
    with col1:
        perimeter_error = st.text_input('Perimeter Error')
    with col2:
        area_error = st.text_input('Area Error')
    with col3:
        smoothness_error = st.text_input('Smoothness Error')
    with col1:
        compactness_error = st.text_input('Compactness Error')
    with col2:
        concavity_error = st.text_input('Concavity Error')
    with col3:
        concave_points_error = st.text_input('Concave Points Error')
    with col1:
        symmetry_error = st.text_input('Symmetry Error')
    with col2:
        fractal_dimension_error = st.text_input('Fractal Dimension Error')
    with col3:
        worst_radius = st.text_input('Worst Radius')
    with col1:
        worst_texture = st.text_input('Worst Texture')
    with col2:
        worst_perimeter = st.text_input('Worst Perimeter')
    with col3:
        worst_area = st.text_input('Worst Area')
    with col1:
        worst_smoothness = st.text_input('Worst Smoothness')
    with col2:
        worst_compactness = st.text_input('Worst Compactness')
    with col3:
        worst_concavity = st.text_input('Worst Concavity')
    with col1:
        worst_concave_points = st.text_input('Worst Concave Points')
    with col2:
        worst_symmetry = st.text_input('Worst Symmetry')
    with col3:
        worst_fractal_dimension = st.text_input('Worst Fractal Dimension')

    cancer_diagnosis = ''
    if st.button('Breast Cancer Test Result'):
        try:
            user_input = [
                mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
                mean_compactness, mean_concavity, mean_concave_points, mean_symmetry,
                mean_fractal_dimension, radius_error, texture_error, perimeter_error,
                area_error, smoothness_error, compactness_error, concavity_error,
                concave_points_error, symmetry_error, fractal_dimension_error,
                worst_radius, worst_texture, worst_perimeter, worst_area,
                worst_smoothness, worst_compactness, worst_concavity,
                worst_concave_points, worst_symmetry, worst_fractal_dimension
            ]
            user_input = [float(x) for x in user_input]
            input_df = pd.DataFrame([user_input], columns=breast_cancer_features)
            prediction = breast_cancer_model.predict(input_df)
            cancer_diagnosis = 'The Person has Breast Cancer' if prediction[0] == 1 else 'The Person does not have Breast Cancer'
            st.success(cancer_diagnosis)

            patient_data = f"Mean Radius: {mean_radius}, Mean Texture: {mean_texture}, Mean Perimeter: {mean_perimeter}, " \
                          f"Mean Area: {mean_area}, Mean Smoothness: {mean_smoothness}, Mean Compactness: {mean_compactness}, " \
                          f"Mean Concavity: {mean_concavity}, Mean Concave Points: {mean_concave_points}, " \
                          f"Mean Symmetry: {mean_symmetry}, Mean Fractal Dimension: {mean_fractal_dimension}, " \
                          f"Radius Error: {radius_error}, Texture Error: {texture_error}, Perimeter Error: {perimeter_error}, " \
                          f"Area Error: {area_error}, Smoothness Error: {smoothness_error}, Compactness Error: {compactness_error}, " \
                          f"Concavity Error: {concavity_error}, Concave Points Error: {concave_points_error}, " \
                          f"Symmetry Error: {symmetry_error}, Fractal Dimension Error: {fractal_dimension_error}, " \
                          f"Worst Radius: {worst_radius}, Worst Texture: {worst_texture}, Worst Perimeter: {worst_perimeter}, " \
                          f"Worst Area: {worst_area}, Worst Smoothness: {worst_smoothness}, Worst Compactness: {worst_compactness}, " \
                          f"Worst Concavity: {worst_concavity}, Worst Concave Points: {worst_concave_points}, " \
                          f"Worst Symmetry: {worst_symmetry}, Worst Fractal Dimension: {worst_fractal_dimension}"
            
            api_token = st.secrets.get("HF_API_TOKEN", "")
            if api_token:
                with st.spinner("Generating report..."):
                    report = generate_report_with_hf(patient_data, cancer_diagnosis, api_token)
                st.subheader("Health Report")
                current_time = datetime.now().strftime("%I:%M %p IST, %A, %B %d, %Y")
                st.caption(f"Generated on {current_time}")
                st.divider()
                display_health_report(report)
            else:
                st.error("Hugging Face API token not found. Add it to secrets.toml.")
        except ValueError:
            st.error("Please enter valid numerical values for all fields.")