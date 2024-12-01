import streamlit as st

def get_demo_recommendation():
    """Return demo recommendation data"""
    return {
        'recommendation': """**Medication Suggestions**:
* **Add Insulin Sensitizer (e.g., Pioglitazone)**: Consider adding pioglitazone to improve glycemic control, especially in cases of poor metabolic control.
* **Intensify Lifestyle Modifications**: Encourage stricter dietary management and regular aerobic exercise.
* **Add Diuretic**: Consider adding a low dose of thiazide diuretic.""",
        
        'rationale': """**Similar Cases and Supporting Evidence**
1. **Similar Case 1 (Similarity Score: 0.87)**:
   * Age: 45 years, Female
   * Diagnoses: T2DM for 7 years, Hypertension for 2 years
   * Treatment Plan: Combined metformin with pioglitazone, added hydrochlorothiazide
   * Outcome: HbA1c decreased by 1.2%, blood pressure within target range

2. **Similar Case 2 (Similarity Score: 0.82)**:
   * Age: 38 years, Female
   * Diagnoses: T2DM for 5 years, Hypertension with obesity
   * Treatment Plan: Lifestyle modifications with metformin and lisinopril
   * Outcome: Near normal blood glucose, BP reduced to 130/85 mmHg""",
        
        'monitoring': """**Monitoring Recommendations**:
* Daily blood glucose monitoring before meals and at bedtime
* Renal function check every three months
* Regular blood pressure monitoring
* Weight and BMI tracking""",
        
        'risks': [
            "Pioglitazone may cause fluid retention",
            "Thiazide diuretics may lead to electrolyte imbalances",
            "Monitor for signs of hypoglycemia",
            "Regular kidney function monitoring required"
        ],
        
        'similar_cases': [
            {
                'similarity': 0.87,
                'demographics': '45 years, Female',
                'diagnoses': ['T2DM', 'Hypertension'],
                'medications': 'metformin, pioglitazone, hydrochlorothiazide'
            },
            {
                'similarity': 0.82,
                'demographics': '38 years, Female',
                'diagnoses': ['T2DM', 'Hypertension', 'Obesity'],
                'medications': 'metformin, lisinopril'
            }
        ]
    }

def main():
    st.set_page_config(
        page_title="Medical Treatment Recommender",
        page_icon="💊",
        layout="wide"
    )
    
    st.title("🤖DrugGPT")
    st.markdown("""
    This system provides medication guidence and personalized treatments using LLaMA 3.2 1B.
    """)
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Patient Information")
        
        with st.form("patient_form"):
            age_group = st.selectbox(
                "Age Group",
                ["18-30", "31-50", "51-70", "71+"],
                index=1  # Default to 31-50
            )
            
            gender = st.selectbox(
                "Gender",
                ["M", "F"],
                index=1  # Default to F
            )
            
            diagnoses = st.text_area(
                "Diagnoses",
                value="Type 2 Diabetes, Hypertension",
                help="e.g., Type 2 Diabetes, Hypertension"
            )
            
            medications = st.text_area(
                "Current Medications",
                value="metformin 1000mg BID, lisinopril 10mg daily",
                help="e.g., metformin 1000mg BID, lisinopril 10mg daily"
            )
            
            history = st.text_area(
                "Clinical History",
                value="Patient with 5-year history of T2DM, recently diagnosed hypertension",
                height=100
            )
            
            submit = st.form_submit_button("Generate Recommendation")
    
    with col2:
        if submit:
            # Display medical disclaimer
            st.warning("⚕️ MEDICAL DISCLAIMER: This system provides recommendations for reference only. All medical decisions should be made under professional medical supervision.")
            
            # Get demo recommendation
            result = get_demo_recommendation()
            
            # Display recommendation sections
            st.subheader("💊 Treatment Recommendation")
            st.markdown(result['recommendation'])
            
            with st.expander("📋 Clinical Rationale"):
                st.markdown(result['rationale'])
            
            with st.expander("🔍 Monitoring Plan"):
                st.markdown(result['monitoring'])
            
            with st.expander("⚠️ Risks and Precautions"):
                for risk in result['risks']:
                    st.markdown(f"• {risk}")
            
            with st.expander("📊 Similar Cases Analysis"):
                for case in result['similar_cases']:
                    st.markdown(f"""
                    **Case** (Similarity: {case['similarity']:.2f})
                    - Demographics: {case['demographics']}
                    - Diagnoses: {', '.join(case['diagnoses'])}
                    - Medications: {case['medications']}
                    ---
                    """)
            
            # Display disclaimer at the bottom
            st.info("⚕️ NOTE: All recommendations must be verified by healthcare professionals. This is not a substitute for professional medical advice.")

if __name__ == "__main__":
    main()