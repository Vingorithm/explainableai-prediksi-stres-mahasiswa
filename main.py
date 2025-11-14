import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Load Deployment Artifacts ---
# Pastikan file-file ini berada di direktori yang sama dengan app.py Anda
try:
    model = joblib.load('xgboost_model_for_deployment.pkl')
    scaler = joblib.load('minmax_scaler_for_deployment.pkl')
    feature_names = joblib.load('feature_names.pkl')
    st.success("Model dan artefak berhasil dimuat!")
except FileNotFoundError:
    st.error("Error: File deployment (model, scaler, atau feature names) tidak ditemukan. Pastikan semua file .pkl berada di direktori yang sama.")
    st.stop() # Hentikan aplikasi jika file tidak ditemukan

# --- Preprocessing Function ---
def preprocess_input(raw_input_data, loaded_scaler, loaded_feature_names):
    # Convert raw input dictionary to a DataFrame
    input_df = pd.DataFrame([raw_input_data])

    # 1. Normalize column names (matching notebook's initial step)
    input_df.columns = (
        input_df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace("/", "_")
    )

    # Create a DataFrame with all expected features, initialized to 0
    # This handles one-hot encoded columns that might not be present in a single input instance
    processed_df = pd.DataFrame(0, index=[0], columns=loaded_feature_names)

    # Map raw input values to the processed_df
    # Numeric features
    for col in ['age', 'screen_time_hrs_day', 'sleep_duration_hrs', 'physical_activity_hrs_week']:
        if col in input_df.columns and col in processed_df.columns:
            processed_df[col] = input_df[col].iloc[0]

    # Categorical/Binary Encoding
    # Gender (One-Hot Encoded)
    if 'gender' in input_df.columns:
        gender_val = input_df['gender'].iloc[0].lower()
        if f'gender_{gender_val}' in processed_df.columns:
            processed_df[f'gender_{gender_val}'] = 1

    # Anxious Before Exams (Binary)
    processed_df['anxious_before_exams'] = input_df['anxious_before_exams'].map({"Yes": 1, "No": 0}).iloc[0]

    # Academic Performance Change (Ordinal)
    ordinal_mapping = {"Declined": 0, "Same": 1, "Improved": 2}
    processed_df['academic_performance_change'] = input_df['academic_performance_change'].map(ordinal_mapping).iloc[0]

    # Education Level (Simplified then One-Hot Encoded)
    def simplify_education(level):
        if "Class" in level:
            return "high_school"
        elif level in ["BTech", "BA", "BSc"]:
            return "undergraduate"
        else: # MSc, MTech, MA
            return "postgraduate"

    if 'education_level' in input_df.columns:
        edu_simplified_category = simplify_education(input_df['education_level'].iloc[0])
        if f'edu_{edu_simplified_category}' in processed_df.columns:
            processed_df[f'edu_{edu_simplified_category}'] = 1

    # Feature Engineering (as per notebook)
    processed_df["sleep_quality_index"] = processed_df["sleep_duration_hrs"] / (processed_df["screen_time_hrs_day"] + 1e-6)
    processed_df["lifestyle_balance_score"] = processed_df["physical_activity_hrs_week"] / (processed_df["screen_time_hrs_day"] + 1e-6)

    # Age Group (binning and One-Hot Encoded)
    def categorize_age(age):
        if age <= 18:
            return "15-18"
        elif age <= 22:
            return "19-22"
        else:
            return ">22"

    age_group_category = categorize_age(processed_df['age'].iloc[0])
    if f'agegrp_{age_group_category}' in processed_df.columns:
        processed_df[f'agegrp_{age_group_category}'] = 1

    # Exam Anxiety x Education (Interaction)
    # Reconstruct simplified education status from processed_df's one-hot columns
    edu_status_for_interaction = ""
    if processed_df['edu_undergraduate'].iloc[0] == 1:
        edu_status_for_interaction = "UG"
    elif processed_df['edu_high_school'].iloc[0] == 1:
        edu_status_for_interaction = "HS"
    elif processed_df['edu_postgraduate'].iloc[0] == 1:
        edu_status_for_interaction = "PG"

    anxiety_status = "Anxious" if processed_df['anxious_before_exams'].iloc[0] == 1 else "Calm"
    combined_feature_name = f'anxedu_{anxiety_status}_{edu_status_for_interaction}'.lower()

    if combined_feature_name in processed_df.columns:
        processed_df[combined_feature_name] = 1

    processed_df["screen_x_sleep"] = processed_df["screen_time_hrs_day"] * processed_df["sleep_duration_hrs"]

    # Scaling
    ratio_features_to_scale = ["sleep_quality_index", "lifestyle_balance_score"]
    for col in ratio_features_to_scale:
        if col in processed_df.columns:
            processed_df[[col]] = loaded_scaler.transform(processed_df[[col]])

    # Ensure the final DataFrame has columns in the exact order as feature_names
    processed_df = processed_df[loaded_feature_names]

    return processed_df

# --- Streamlit App Layout ---
st.set_page_config(page_title="Student Stress Level Prediction", layout="centered")

st.title("💡 Aplikasi Prediksi Tingkat Stres Mahasiswa")
st.markdown("""
Aplikasi ini membantu memprediksi tingkat stres mahasiswa (Rendah, Sedang, Tinggi) berdasarkan beberapa faktor gaya hidup.
""")

st.subheader("Masukkan Data Mahasiswa:")

# Input widgets organized in columns for better layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Usia", 15, 30, 20)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    screen_time_hrs_day = st.number_input("Waktu Layar (jam/hari)", min_value=0.0, max_value=24.0, value=7.0, step=0.1)
    physical_activity_hrs_week = st.number_input("Aktivitas Fisik (jam/minggu)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)

with col2:
    education_level = st.selectbox(
        "Tingkat Pendidikan",
        ["Class 8", "Class 9", "Class 10", "Class 11", "Class 12", "BA", "BSc", "BTech", "MA", "MSc", "MTech"]
    )
    sleep_duration_hrs = st.number_input("Durasi Tidur (jam/hari)", min_value=0.0, max_value=12.0, value=7.0, step=0.1)
    anxious_before_exams = st.selectbox("Cemas Sebelum Ujian?", ["No", "Yes"]) # Changed order to match Yes=1, No=0 logic better, though it's mapped.
    academic_performance_change = st.selectbox("Perubahan Performa Akademik", ["Same", "Improved", "Declined"]) # Changed order to match ordinal logic better.

# Collect raw inputs into a dictionary
input_data_raw = {
    "Age": age,
    "Gender": gender,
    "Education Level": education_level,
    "Screen Time (hrs/day)": screen_time_hrs_day,
    "Sleep Duration (hrs)": sleep_duration_hrs,
    "Physical Activity (hrs/week)": physical_activity_hrs_week,
    "Anxious Before Exams": anxious_before_exams,
    "Academic Performance Change": academic_performance_change
}

if st.button("Prediksi Tingkat Stres"):
    with st.spinner('Memproses input dan membuat prediksi...'):
        try:
            processed_data = preprocess_input(input_data_raw, scaler, feature_names)
            prediction_proba = model.predict_proba(processed_data)[0]
            prediction_class = np.argmax(prediction_proba)

            stress_labels = {0: "Rendah (Low Stress)", 1: "Sedang (Medium Stress)", 2: "Tinggi (High Stress)"}
            predicted_stress_level = stress_labels[prediction_class]

            st.subheader("✅ Hasil Prediksi Tingkat Stres:")
            st.success(f"Mahasiswa ini diprediksi memiliki **{predicted_stress_level}**.")

            st.markdown("---")
            st.subheader("Probabilitas untuk Setiap Tingkat Stres:")
            col_proba1, col_proba2, col_proba3 = st.columns(3)
            with col_proba1:
                st.metric("Rendah", f"{prediction_proba[0]*100:.2f}%")
            with col_proba2:
                st.metric("Sedang", f"{prediction_proba[1]*100:.2f}%")
            with col_proba3:
                st.metric("Tinggi", f"{prediction_proba[2]*100:.2f}%")

            st.markdown("---")
            st.subheader("💡 Interpretasi Singkat:")
            if prediction_class == 0:
                st.info("Prediksi stres rendah sering dikaitkan dengan kombinasi durasi tidur yang memadai, aktivitas fisik yang teratur, dan penggunaan waktu layar yang terkontrol.")
            elif prediction_class == 1:
                st.info("Tingkat stres sedang mungkin menunjukkan keseimbangan antara beberapa faktor pemicu stres dan faktor pelindung. Mungkin ada area yang bisa ditingkatkan untuk mengurangi stres lebih lanjut.")
            else:
                st.info("Prediksi stres tinggi sering dikaitkan dengan durasi tidur yang kurang, aktivitas fisik yang minim, dan penggunaan waktu layar yang berlebihan, serta faktor-faktor lain seperti kecemasan sebelum ujian.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses input atau membuat prediksi: {e}")
            st.exception(e)

st.markdown("---")
st.caption("Aplikasi dibangun menggunakan model XGBoost yang dilatih pada dataset mental health mahasiswa.")
