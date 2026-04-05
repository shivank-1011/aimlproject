import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import re
import os
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from groq_analysis import GroqAnalyzer
from diagnosis import StudentDiagnosis
from agent_workflow import StudentAnalysisAgent
from pdf_generator import generate_study_plan_pdf

# config
MODEL_PATH = 'model/model.pkl'
SCALER_PATH = 'model/scaler.pkl'
PAGE_TITLE = "Student Risk Analysis System"

st.set_page_config(page_title=PAGE_TITLE, layout="wide")

def parse_json_column(df, column_name, prefix):
    """Parses a column containing JSON strings into separate columns."""
    def safe_json_loads(x):
        try:
            if isinstance(x, str):
                # Fix unquoted keys in JSON string
                x = re.sub(r'([a-zA-Z0-9_]+):', r'"\1":', x)
                return json.loads(x)
            return {}
        except Exception:
            return {}

    if column_name not in df.columns:
        return df

    expanded_data = df[column_name].apply(safe_json_loads).apply(pd.Series)
    expanded_data = expanded_data.add_prefix(f'{prefix}_')
    return pd.concat([df, expanded_data], axis=1).drop(columns=[column_name])

def extract_section(urn):
    if not isinstance(urn, str):
        return 'Unknown'
    match = re.search(r'2024[-]?([A-Z])', urn)
    if match:
        return match.group(1)
    return 'Unknown'

@st.cache_resource
def load_resources():
    try:
        model = None
        scaler = None
        
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
        else:
            st.error(f"Model file not found at {MODEL_PATH}")

        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
        else:
            st.error(f"Scaler file not found at {SCALER_PATH}")
            
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

def validate_input(df):
    """Checks for required critical columns."""
    required_subjects = ['Maths', 'SESD', 'AIML', 'FSD', 'DVA']
    missing = [col for col in required_subjects if col not in df.columns]
    
    if missing:
         return False, f"Missing critical subject columns: {', '.join(missing)}"
    
    if df.empty:
        return False, "Uploaded file is empty."
        
    return True, ""

def preprocess_input(df, scaler):
    """Preprocesses input dataframe for prediction."""
    processed_df = df.copy()

    # 1. Drop unnecessary columns (if present)
    cols_to_drop = ['id', 'student_name']
    processed_df = processed_df.drop(columns=[c for c in cols_to_drop if c in processed_df.columns])

    # 2. Extract Section from URN
    if 'URN' in processed_df.columns:
        processed_df['Section'] = processed_df['URN'].apply(extract_section)
        processed_df = processed_df.drop(columns=['URN'])
    elif 'Section' not in processed_df.columns:
        processed_df['Section'] = 'Unknown'

    # 3. Parse JSON-like columns
    if 'topic_wise_accuracy' in processed_df.columns:
        processed_df = parse_json_column(processed_df, 'topic_wise_accuracy', 'Acc')
    if 'time_spent_per_topic' in processed_df.columns:
        processed_df = parse_json_column(processed_df, 'time_spent_per_topic', 'Time')

    # 4. Handle invalid values and outliers
    numeric_candidates = [c for c in processed_df.columns if c != 'Section']
    for col in numeric_candidates:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # Replace negative scores with NaN
    subject_cols = ['Maths', 'SESD', 'AIML', 'FSD', 'DVA']
    for col in subject_cols:
        if col in processed_df.columns:
            processed_df.loc[processed_df[col] < 0, col] = np.nan

    # Drop Target if present
    if 'PlacementStatus' in processed_df.columns:
        processed_df = processed_df.drop(columns=['PlacementStatus'])

    # IQR Capping
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = processed_df[col].quantile(0.25)
        Q3 = processed_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        processed_df[col] = processed_df[col].clip(lower=lower, upper=upper)

    # 5. Impute missing values with mean
    # Note: Ideally this should use training set means, but we follow preprocessing.py logic here
    for col in numeric_cols:
        processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
    processed_df = processed_df.fillna(0) 

    # 6. One-hot encode 'Section'
    processed_df = pd.get_dummies(processed_df, columns=['Section'], prefix='Section', drop_first=False)
    
    # Ensure all expected columns are present (based on scaler features)
    if hasattr(scaler, 'feature_names_in_'):
        expected_cols = scaler.feature_names_in_
        for col in expected_cols:
            if col not in processed_df.columns:
                processed_df[col] = 0
        # Reorder and filter columns to match scaler input
        processed_df = processed_df[expected_cols]
    else:
        st.warning("Scaler does not have feature_names_in_ attribute. Columns might be mismatched.")

    # 7. Scale
    scaled_values = scaler.transform(processed_df)
    return pd.DataFrame(scaled_values, columns=processed_df.columns)

def get_recommendations(category):
    # Synced with model/recommendation.py
    recommendations_map = {
        'At-risk': [
            "Focus on fundamental concepts and weak topics.",
            "Increase daily practice to at least 2 hours.",
            "Seek immediate mentorship or peer support."
        ],
        'Average': [
            "Systematically revise identified weak areas.",
            "Take weekly mock tests to improve exam temperament.",
            "Analyze errors in depth to prevent recurrence."
        ],
        'High-performing': [
            "Solve advanced and competitive-level problems.",
            "Practice under strict time constraints.",
            "Optimize solutions for better time/space complexity."
        ]
    }
    return " | ".join(recommendations_map.get(category, ["No specific recommendation."]))

def main():
    st.title(f"{PAGE_TITLE}")
    st.markdown("Upload student data to predict placement risk and generate study recommendations.")

    # Sidebar
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    model, scaler = load_resources()
    if not model or not scaler:
        st.stop()

    # Clear session state if a new file is uploaded
    if uploaded_file is not None:
        if 'uploaded_file_name' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            for key in ['results_df', 'probs']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.uploaded_file_name = uploaded_file.name

        try:
            # Check for empty file
            if uploaded_file.size == 0:
                st.error("Uploaded file is empty.")
                return

            try:
                raw_df = pd.read_csv(uploaded_file)
            except pd.errors.ParserError:
                st.error("Uploaded file is not a valid CSV.")
                return
            
            st.subheader("Data Preview")
            st.dataframe(raw_df.head())

            # Validation
            is_valid, error_msg = validate_input(raw_df)
            if not is_valid:
                st.error(f"Validation Error: {error_msg}")
                return

            # Perform or retrieve analysis
            if st.button("Analyze Students") or 'results_df' in st.session_state:
                if 'results_df' not in st.session_state:
                    with st.spinner("Processing data and generating predictions..."):
                        try:
                            X_processed = preprocess_input(raw_df, scaler)
                            
                            if hasattr(model, "predict_proba"):
                                probs = model.predict_proba(X_processed)[:, 1]
                            else:
                                probs = model.predict(X_processed)
                            
                            categories = []
                            for p in probs:
                                if p < 0.40:
                                    categories.append('At-risk')
                                elif p < 0.80:
                                    categories.append('Average')
                                else:
                                    categories.append('High-performing')
                            
                            # Results DataFrame
                            res_df = raw_df.copy()
                            res_df['Risk Score'] = np.round(probs, 4)
                            res_df['Risk Category'] = categories
                            res_df['Recommendations'] = [get_recommendations(c) for c in categories]
                            
                            # Store in session state
                            st.session_state.results_df = res_df
                            st.session_state.probs = probs
                        except Exception as e:
                            st.error(f"An error occurred during analysis: {str(e)}")
                            st.exception(e)
                            return

                # Get results from state
                results_df = st.session_state.results_df
                
                # DASHBOARD DISPLAY
                st.divider()
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader(" Detailed Analysis")
                    disp_cols = ['Risk Category', 'Risk Score', 'Recommendations']
                    if 'student_name' in results_df.columns:
                        disp_cols.insert(0, 'student_name')
                    st.dataframe(results_df[disp_cols])
                
                with col2:
                    st.subheader("Risk Distribution")
                    count_df = results_df['Risk Category'].value_counts().reset_index()
                    count_df.columns = ['Category', 'Count']
                    fig = px.pie(count_df, values='Count', names='Category', color='Category',
                                    color_discrete_map={'At-risk':'red', 'Average':'orange', 'High-performing':'green'})
                    st.plotly_chart(fig, width='stretch')

                # Download button for CSV
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Full CSV Report",
                    data=csv,
                    file_name="student_risk_report.csv",
                    mime="text/csv",
                )

                # AI ANALYSIS SECTION
                st.divider()
                st.subheader("AI Study Planner (Powered by Groq)")
                
                # Filter for priority students (at-risk or average)
                priority_students = results_df[results_df['Risk Category'].isin(['At-risk', 'Average'])]
                
                if not priority_students.empty:
                    st.info("Generating a detailed AI diagnosis and study plan for students who need more support.")
                    
                    col_a, col_b = st.columns([1, 1])
                    
                    with col_a:
                        student_names = priority_students['student_name'].tolist() if 'student_name' in priority_students.columns else priority_students.index.tolist()
                        selected_student = st.selectbox("Select a student", options=student_names)
                    
                    with col_b:
                        current_goal = st.text_input("Goal", "Improve core subjects and prepare for placement rounds.")
                        
                    # Data-driven identification of weak areas (integrated from diagnosis.py)
                    diagnostician = StudentDiagnosis()
                    if selected_student:
                        if 'student_name' in priority_students.columns:
                            row = priority_students[priority_students['student_name'] == selected_student].iloc[0]
                        else:
                            row = priority_students.loc[selected_student]
                        
                        perf_dict_raw = row.to_dict()
                        diagnosis_data = diagnostician.identify_weak_areas(perf_dict_raw)
                        
                        # Show Critical Weak Areas (Red/Warning)
                        if diagnosis_data["critical"]:
                            st.error(f"**Critical Weak Areas for {selected_student}:**")
                            for w in diagnosis_data["critical"]:
                                st.write(f"- {w['reason']} (Score: {w['score']})")
                        
                        # Show Areas for Improvement (Yellow/Info)
                        if diagnosis_data["improvement"]:
                            st.warning(f"**Areas for Improvement for {selected_student}:**")
                            for w in diagnosis_data["improvement"]:
                                st.write(f"- {w['reason']} (Score: {w['score']})")
                        
                        if not diagnosis_data["critical"] and not diagnosis_data["improvement"]:
                            st.success(f"Excellent performance across all subjects for {selected_student}!")

                    # --- AGENT ANALYTICS (MEMORY PERSISTENCE) ---
                    if "agent_report" not in st.session_state:
                        st.session_state.agent_report = None
                    if "report_for_student" not in st.session_state:
                        st.session_state.report_for_student = None

                    if st.button("Generate AI Diagnosis"):
                        if selected_student:
                            with st.spinner(f"Agent executing workflow for {selected_student}..."):
                                try:
                                    # Run Agent Task
                                    agent = StudentAnalysisAgent()
                                    report = agent.run(perf_dict_raw, current_goal)
                                    
                                    # Store in Memory
                                    st.session_state.agent_report = report
                                    st.session_state.report_for_student = selected_student
                                    
                                except Exception as e:
                                    st.error(f"Agent Workflow Error: {e}")
                        else:
                            st.warning("Please select a student.")

                    # Display the report if it matches the current selection
                    if st.session_state.agent_report and st.session_state.report_for_student == selected_student:
                        report = st.session_state.agent_report
                        st.markdown("---")
                        st.subheader(f"🎓 Personalized Study Plan for {selected_student}")
                        
                        # Show full report
                        st.markdown(report)

                        col1, col2 = st.columns(2)
                        with col1:
                            # DOWNLOAD OPTION 1: MARKDOWN
                            st.download_button(
                                label="📄 Download Agent Report (MD)",
                                data=report,
                                file_name=f"Agent_Report_{selected_student}.md",
                                mime="text/markdown",
                                width="stretch"
                            )
                        
                        with col2:
                            # DOWNLOAD OPTION 2: PDF
                            try:
                                pdf_bytes = generate_study_plan_pdf(selected_student, report)
                                st.download_button(
                                    label="📕 Download Study Plan (PDF)",
                                    data=pdf_bytes,
                                    file_name=f"StudyPlan_{selected_student}.pdf",
                                    mime="application/pdf",
                                    width="stretch"
                                )
                            except Exception as pdf_err:
                                st.warning(f"Note: PDF generation skipped.")

                else:
                    st.success("All students are currently in the 'High-performing' category. Well done!")

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()