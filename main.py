import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Stress Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .content-box {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    h2 {
        color: #667eea;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    h3 {
        color: #764ba2;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Deployment Artifacts ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load('xgboost_model_for_deployment.pkl')
        scaler = joblib.load('minmax_scaler_for_deployment.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, feature_names, None
    except FileNotFoundError as e:
        return None, None, None, str(e)

model, scaler, feature_names, error = load_models()

# --- Title ---
st.markdown("<h1>🧠 Student Stress Level Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size: 1.2rem;'>Analyze stress levels using AI-powered XGBoost model with lifestyle factors</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if error:
    st.error(f"❌ Error loading model files: {error}")
    st.info("📝 Please ensure all .pkl files are in the same directory as this app.")
    st.stop()

# --- Preprocessing Function ---
def preprocess_input(raw_input_data, loaded_scaler, loaded_feature_names):
    input_df = pd.DataFrame([raw_input_data])
    
    # Normalize column names
    input_df.columns = (
        input_df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace("/", "_")
    )
    
    # Create DataFrame with all expected features
    processed_df = pd.DataFrame(0, index=[0], columns=loaded_feature_names)
    
    # Map numeric features
    for col in ['age', 'screen_time_hrs_day', 'sleep_duration_hrs', 'physical_activity_hrs_week']:
        if col in input_df.columns and col in processed_df.columns:
            processed_df[col] = input_df[col].iloc[0]
    
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
            return "High School"
        elif level in ["BTech", "BA", "BSc"]:
            return "Undergraduate"
        else:
            return "Postgraduate"
    
    if 'education_level' in input_df.columns:
        edu_simplified_category = simplify_education(input_df['education_level'].iloc[0])
        if f'edu_{edu_simplified_category}' in processed_df.columns:
            processed_df[f'edu_{edu_simplified_category}'] = 1
    
    # Feature Engineering
    processed_df["sleep_quality_index"] = processed_df["sleep_duration_hrs"] / (processed_df["screen_time_hrs_day"] + 1e-6)
    processed_df["lifestyle_balance_score"] = processed_df["physical_activity_hrs_week"] / (processed_df["screen_time_hrs_day"] + 1e-6)
    
    # Age Group
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
    edu_status_for_interaction = ""
    if 'edu_Undergraduate' in processed_df.columns and processed_df['edu_Undergraduate'].iloc[0] == 1:
        edu_status_for_interaction = "UG"
    elif 'edu_High School' in processed_df.columns and processed_df['edu_High School'].iloc[0] == 1:
        edu_status_for_interaction = "HS"
    elif 'edu_Postgraduate' in processed_df.columns and processed_df['edu_Postgraduate'].iloc[0] == 1:
        edu_status_for_interaction = "PG"
    
    anxiety_status = "Anxious" if processed_df['anxious_before_exams'].iloc[0] == 1 else "Calm"
    combined_feature_name = f'anxEdu_{anxiety_status}_{edu_status_for_interaction}'
    
    if combined_feature_name in processed_df.columns:
        processed_df[combined_feature_name] = 1
    
    processed_df["screen_x_sleep"] = processed_df["screen_time_hrs_day"] * processed_df["sleep_duration_hrs"]
    
    # Scaling
    ratio_features_to_scale = ["sleep_quality_index", "lifestyle_balance_score"]
    for col in ratio_features_to_scale:
        if col in processed_df.columns:
            processed_df[[col]] = loaded_scaler.transform(processed_df[[col]])
    
    processed_df = processed_df[loaded_feature_names]
    
    return processed_df

# --- Sidebar for Input ---
with st.sidebar:
    st.markdown("## 📝 Input Student Data")
    st.markdown("---")
    
    # Personal Information
    st.markdown("### 👤 Personal Information")
    age = st.slider("Age", 15, 30, 20, help="Student's age")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], help="Student's gender")
    
    education_level = st.selectbox(
        "Education Level",
        ["Class 8", "Class 9", "Class 10", "Class 11", "Class 12", 
         "BA", "BSc", "BTech", "MA", "MSc", "MTech"],
        index=5,
        help="Current education level"
    )
    
    st.markdown("---")
    
    # Lifestyle Factors
    st.markdown("### 🏃 Lifestyle Factors")
    screen_time_hrs_day = st.slider(
        "Screen Time (hrs/day)", 
        0.0, 24.0, 7.0, 0.5,
        help="Average daily screen time including phone, computer, TV"
    )
    
    sleep_duration_hrs = st.slider(
        "Sleep Duration (hrs/day)", 
        0.0, 12.0, 7.0, 0.5,
        help="Average sleep hours per night"
    )
    
    physical_activity_hrs_week = st.slider(
        "Physical Activity (hrs/week)", 
        0.0, 20.0, 5.0, 0.5,
        help="Total weekly hours of exercise or sports"
    )
    
    st.markdown("---")
    
    # Academic Factors
    st.markdown("### 📚 Academic Factors")
    anxious_before_exams = st.radio(
        "Anxious Before Exams?", 
        ["No", "Yes"],
        help="Do you feel anxious before exams?"
    )
    
    academic_performance_change = st.selectbox(
        "Academic Performance Change", 
        ["Improved", "Same", "Declined"],
        help="How has your academic performance changed recently?"
    )
    
    st.markdown("---")
    predict_btn = st.button("🔮 Predict Stress Level", type="primary")

# --- Main Content ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='content-box'>", unsafe_allow_html=True)
    st.markdown("## 📊 Input Summary")
    
    # Create summary metrics
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{age}</div>
                <div class='metric-label'>Years Old</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_b:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{screen_time_hrs_day:.1f}</div>
                <div class='metric-label'>Screen Time (hrs)</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_c:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{sleep_duration_hrs:.1f}</div>
                <div class='metric-label'>Sleep (hrs)</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_d:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{physical_activity_hrs_week:.1f}</div>
                <div class='metric-label'>Exercise (hrs/week)</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='content-box'>", unsafe_allow_html=True)
    st.markdown("## 📋 Quick Stats")
    
    # Calculate health scores
    sleep_quality = sleep_duration_hrs / (screen_time_hrs_day + 0.1)
    lifestyle_balance = physical_activity_hrs_week / (screen_time_hrs_day + 0.1)
    
    st.metric("Sleep Quality Index", f"{sleep_quality:.2f}")
    st.metric("Lifestyle Balance Score", f"{lifestyle_balance:.2f}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- Prediction Section ---
if predict_btn:
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
    
    with st.spinner('🔄 Analyzing data and making prediction...'):
        try:
            processed_data = preprocess_input(input_data_raw, scaler, feature_names)
            prediction_proba = model.predict_proba(processed_data)[0]
            prediction_class = np.argmax(prediction_proba)
            
            stress_labels = {
                0: "Low Stress",
                1: "Medium Stress",
                2: "High Stress"
            }
            stress_colors = {
                0: "#10b981",
                1: "#f59e0b",
                2: "#ef4444"
            }
            stress_emojis = {
                0: "😊",
                1: "😐",
                2: "😰"
            }
            
            predicted_stress_level = stress_labels[prediction_class]
            stress_color = stress_colors[prediction_class]
            stress_emoji = stress_emojis[prediction_class]
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Prediction Result
            st.markdown("<div class='content-box'>", unsafe_allow_html=True)
            st.markdown(f"""
                <div style='text-align: center; padding: 2rem;'>
                    <h1 style='color: {stress_color}; font-size: 4rem; margin: 0;'>{stress_emoji}</h1>
                    <h2 style='color: {stress_color}; margin: 1rem 0;'>{predicted_stress_level}</h2>
                    <p style='font-size: 1.2rem; color: #666;'>Confidence: {prediction_proba[prediction_class]*100:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Probability Distribution
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("<div class='content-box'>", unsafe_allow_html=True)
                st.markdown("### 📊 Probability Distribution")
                
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prediction_proba[prediction_class] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"{predicted_stress_level}", 'font': {'size': 24}},
                    delta = {'reference': 50, 'increasing': {'color': stress_color}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': stress_color},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 33], 'color': '#d1fae5'},
                            {'range': [33, 66], 'color': '#fef3c7'},
                            {'range': [66, 100], 'color': '#fee2e2'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#333", 'family': "Arial"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='content-box'>", unsafe_allow_html=True)
                st.markdown("### 📈 All Stress Levels")
                
                # Create bar chart
                fig2 = go.Figure(data=[
                    go.Bar(
                        x=['Low', 'Medium', 'High'],
                        y=prediction_proba * 100,
                        marker_color=['#10b981', '#f59e0b', '#ef4444'],
                        text=[f'{p*100:.1f}%' for p in prediction_proba],
                        textposition='auto',
                    )
                ])
                
                fig2.update_layout(
                    title="Probability for Each Stress Level",
                    xaxis_title="Stress Level",
                    yaxis_title="Probability (%)",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#333"}
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Interpretation
            st.markdown("<div class='content-box'>", unsafe_allow_html=True)
            st.markdown("### 💡 Interpretation & Recommendations")
            
            if prediction_class == 0:
                st.success("""
                **Great news!** Your lifestyle indicators suggest a low stress level. 
                
                **Key Factors Contributing to Low Stress:**
                - ✅ Adequate sleep duration
                - ✅ Balanced screen time
                - ✅ Regular physical activity
                - ✅ Good overall lifestyle balance
                
                **Keep it up by:**
                - Maintaining your current sleep schedule
                - Continue regular exercise routine
                - Keep screen time in check
                """)
            elif prediction_class == 1:
                st.warning("""
                **Notice:** Your lifestyle indicators suggest a moderate stress level.
                
                **Areas to Focus On:**
                - ⚠️ Some lifestyle factors may need improvement
                - ⚠️ Balance between stress triggers and protective factors
                
                **Recommendations:**
                - 😴 Try to improve sleep quality (aim for 7-9 hours)
                - 🏃 Increase physical activity (aim for 5+ hours/week)
                - 📱 Reduce screen time, especially before bed
                - 🧘 Consider stress management techniques (meditation, yoga)
                """)
            else:
                st.error("""
                **Important:** Your lifestyle indicators suggest a high stress level.
                
                **Major Concerns:**
                - ❌ Insufficient sleep duration
                - ❌ Excessive screen time
                - ❌ Limited physical activity
                - ❌ Poor lifestyle balance
                
                **Urgent Recommendations:**
                - 🛌 Prioritize sleep - aim for 8+ hours nightly
                - 📵 Significantly reduce screen time (especially 2 hours before bed)
                - 🏃‍♂️ Start with 30 min daily physical activity
                - 🧘 Practice stress management techniques
                - 👥 Consider talking to a counselor or mental health professional
                - 📚 Improve study-life balance
                """)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Feature Importance Visualization
            st.markdown("<div class='content-box'>", unsafe_allow_html=True)
            st.markdown("### 🎯 Your Key Lifestyle Factors")
            
            user_factors = pd.DataFrame({
                'Factor': ['Sleep Quality', 'Lifestyle Balance', 'Screen Time', 'Physical Activity', 'Sleep Duration'],
                'Score': [
                    sleep_quality * 20,
                    lifestyle_balance * 20,
                    100 - (screen_time_hrs_day / 24 * 100),
                    (physical_activity_hrs_week / 20 * 100),
                    (sleep_duration_hrs / 12 * 100)
                ]
            })
            
            fig3 = px.bar(
                user_factors,
                x='Score',
                y='Factor',
                orientation='h',
                color='Score',
                color_continuous_scale=['#ef4444', '#f59e0b', '#10b981'],
                text='Score'
            )
            
            fig3.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig3.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                xaxis_title="Health Score (%)",
                yaxis_title="",
                font={'color': "#333"}
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {e}")
            st.exception(e)

# --- Footer ---
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white; padding: 1rem;'>
        <p>🤖 Powered by XGBoost Machine Learning Model</p>
        <p style='font-size: 0.9rem; opacity: 0.8;'>
            Built with ❤️ using Streamlit | Model trained on student mental health dataset
        </p>
    </div>
""", unsafe_allow_html=True)