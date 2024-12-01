import streamlit as st
st.set_page_config(
    page_title="DrugGPT:Medical Treatment Recommender",
    page_icon="💊",
    layout="wide"
)

import torch
import sys
from pathlib import Path
import logging
from typing import List, Dict
import pandas as pd

# Add your source code directory to Python path
sys.path.append('../src')  

# Import your existing components
try:
    from Embedding import ClinicalEmbeddingProcessor
    from Retrival import ClinicalRetrievalSystem
    from EnhancedRetrival import EnhancedClinicalRetrieval
    from transformers import AutoTokenizer, AutoModelForCausalLM

except Exception as e:
    st.error(f"Error importing components: {str(e)}")

def initialize_components():
    """Initialize all required components"""
    try:
        # Initialize processor
        processor = ClinicalEmbeddingProcessor(
            processed_path='../data/processed',
            model_name='emilyalsentzer/Bio_ClinicalBERT'
        )
        
        # Initialize base retriever
        retriever = ClinicalRetrievalSystem(
            processed_path='../data/processed',
            index_type='Flat',
            use_gpu=torch.cuda.is_available()
        )
        
        # Initialize enhanced retriever
        enhanced_retriever = EnhancedClinicalRetrieval(
            base_retriever=retriever,
            icd9_map_path='../data/icd9_map.csv'
        )
        
        return processor, enhanced_retriever
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None
    
def initialize_llm():
    """Initialize LLaMA model and tokenizer with proper token settings"""
    try:
        # Initialize tokenizer first
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        
        # Set special tokens
        special_tokens = {
            'pad_token': '[PAD]',
            'eos_token': '</s>',
            'bos_token': '<s>',
            'unk_token': '[UNK]'
        }
        
        # Add special tokens to tokenizer
        num_added_tokens = tokenizer.add_special_tokens(special_tokens)
        
        # Initialize model
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        # Resize embeddings for new tokens
        if num_added_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))
        
        # Set generation config
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error initializing LLaMA model: {str(e)}")
        return None, None

def generate_recommendation(model, tokenizer, prompt: str) -> str:
    """Generate recommendation using LLaMA"""
    try:
        # Debug info
        st.write("Starting generation...")
        
        # Prepare input
        if not prompt.startswith(tokenizer.bos_token):
            prompt = tokenizer.bos_token + prompt
            
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True
        ).to(model.device)
        
        st.write("Input prepared, generating response...")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=1024,
                min_length=100,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.2,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        st.write("Response generated, processing output...")
        
        # Process output
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the generated text
        generated_text = full_text[len(prompt):]
        
        st.write("Generation complete!")
        
        return generated_text
        
    except Exception as e:
        st.error(f"Generation error: {str(e)}")
        st.exception(e)
        return None

def display_recommendation(recommendation: str, similar_cases: List[Dict]):
    """Display recommendation with proper formatting"""
    if not recommendation:
        return
    
    # Display medical disclaimer at the top
    st.warning("⚕️ MEDICAL DISCLAIMER: This system provides recommendations for reference only. All medical decisions should be made under professional medical supervision.")
    
    # Display the recommendation sections
    try:
        sections = recommendation.split('\n\n')
        
        # Display treatment plan
        st.subheader("💊 Treatment Recommendation")
        treatment_section = next((s for s in sections if "Treatment Plan" in s), None)
        if treatment_section:
            st.info(treatment_section.replace("1. Treatment Plan:", "").strip())
        
        # Display clinical rationale
        with st.expander("📋 Clinical Rationale"):
            rationale_section = next((s for s in sections if "Clinical Rationale" in s), None)
            if rationale_section:
                st.write(rationale_section.replace("2. Clinical Rationale:", "").strip())
        
        # Display monitoring plan
        with st.expander("🔍 Monitoring Plan"):
            monitoring_section = next((s for s in sections if "Monitoring Plan" in s), None)
            if monitoring_section:
                st.write(monitoring_section.replace("3. Monitoring Plan:", "").strip())
        
        # Display similar cases analysis
        with st.expander("📊 Similar Cases Analysis"):
            for i, case in enumerate(similar_cases, 1):
                st.markdown(f"""
                **Case {i}** (Similarity: {case['similarity']:.2f})
                - Demographics: {case['demographics']}
                - Diagnoses: {', '.join(case['diagnoses'])}
                - Medications: {case['medications']}
                ---
                """)
        
        # Display disclaimer again at the bottom
        st.info("⚕️ NOTE: All recommendations must be verified by healthcare professionals. This is not a substitute for professional medical advice.")
        
    except Exception as e:
        st.error(f"Error displaying recommendation: {str(e)}")


def construct_prompt(patient_data: Dict, similar_cases: List[Dict]) -> str:
    """Construct prompt using current patient data and retrieved similar cases"""
    
    # Format similar cases information
    similar_cases_text = []
    for i, case in enumerate(similar_cases, 1):
        case_text = (
            f"Case {i} (Similarity: {case['similarity']:.2f}):\n"
            f"- Demographics: {case['demographics']}\n"
            f"- Diagnoses: {', '.join(case['diagnoses'])}\n"
            f"- Medications: {case['medications']}"
        )
        similar_cases_text.append(case_text)
    
    # Construct the complete prompt
    prompt = (
        f"You are an experienced medical professional providing treatment recommendations. "
        f"Analyze the current patient case and similar historical cases to provide evidence-based treatment recommendations.\n\n"
        
        f"Current Patient:\n"
        f"- Age Group: {patient_data.get('age_group', 'Unknown')}\n"
        f"- Gender: {patient_data.get('gender', 'Unknown')}\n"
        f"- Primary Diagnoses: {', '.join(patient_data.get('diagnoses', []))}\n"
        f"- Current Medications: {patient_data.get('medications', 'None')}\n"
        f"- Clinical History: {patient_data.get('history', 'Not available')}\n\n"
        
        f"Similar Historical Cases:\n"
        f"{chr(10).join(similar_cases_text)}\n\n"
        
        "Based on the current patient's profile and the treatment patterns in similar cases, "
        "provide a comprehensive recommendation following this structure:\n\n"
        
        "1. Treatment Plan:\n"
        "- Analyze the effectiveness of treatments used in similar cases\n"
        "- Recommend medication adjustments or new medications based on similar successful cases\n"
        "- Provide specific dosing informed by similar cases\n"
        "- Suggest lifestyle modifications that worked well in similar cases\n\n"
        
        "2. Clinical Rationale:\n"
        "- Explain why specific treatments from similar cases are applicable\n"
        "- Discuss how patient-specific factors influence the recommendation\n"
        "- Reference outcomes from similar cases to support decisions\n\n"
        
        "3. Monitoring Plan:\n"
        "- Specify monitoring parameters based on similar cases' outcomes\n"
        "- Define follow-up frequency\n"
        "- List warning signs to watch for based on similar cases\n\n"
        
        "Keep your response evidence-based, specifically referencing patterns and outcomes from the similar cases provided."
    )
    
    return prompt

def main():
    st.title("🏥DrugGPT")
    st.markdown("""
    This system provides medication guidence and personalized treatments using LLaMA 3.2 1B.
    """)
    
    # Initialize all components if not already initialized
    if 'initialized' not in st.session_state:
        with st.spinner("Initializing components..."):
            processor, enhanced_retriever = initialize_components()
            model, tokenizer = initialize_llm()
            
            if all([processor, enhanced_retriever, model, tokenizer]):
                st.session_state.processor = processor
                st.session_state.enhanced_retriever = enhanced_retriever
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.initialized = True
                st.success("All components initialized successfully!")
            else:
                st.error("Failed to initialize components")
                return
            
    if 'model' in st.session_state:
        st.write("Model device:", next(st.session_state.model.parameters()).device)
        st.write("Model loaded:", type(st.session_state.model).__name__)
        
    # Create two columns
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Patient Information")
        
        with st.form("patient_form"):
            age_group = st.selectbox(
                "Age Group",
                ["18-30", "31-50", "51-70", "71+"]
            )
            
            gender = st.selectbox(
                "Gender",
                ["M", "F"]
            )
            
            diagnoses = st.text_area(
                "Diagnoses",
                placeholder="Enter diagnoses (comma separated)",
                help="e.g., Type 2 Diabetes, Hypertension"
            )
            
            medications = st.text_area(
                "Current Medications",
                placeholder="Enter current medications",
                help="e.g., metformin 1000mg BID, lisinopril 10mg daily"
            )
            
            history = st.text_area(
                "Clinical History",
                placeholder="Enter relevant clinical history",
                height=100
            )
            
            submit = st.form_submit_button("Generate Recommendation")
    
    with col2:
        if submit and st.session_state.initialized:
            
            if not all([age_group, gender, diagnoses, medications]):
                st.error("Please fill in all required fields.")
                return
            
            with st.spinner("Processing..."):
                try:
                    # Get similar cases
                    case_text = f"Patient: {age_group} {gender} [SEP]Diagnoses: {diagnoses} [SEP]Medications: {medications} [SEP]History: {history}"
                    case_embedding = st.session_state.processor.get_case_embedding(case_text)
                    similar_cases = st.session_state.enhanced_retriever.find_similar_cases(
                        case_embedding,
                        k=5,
                        remove_duplicates=True
                    )
                    
                    # Prepare data
                    case_data = {
                        'age_group': age_group,
                        'gender': gender,
                        'diagnoses': [d.strip() for d in diagnoses.split(',')],
                        'medications': medications,
                        'history': history
                    }
                    
                    # Generate recommendation
                    st.write("Preparing to generate recommendation...")
                    prompt = construct_prompt(case_data, similar_cases)
                    recommendation = generate_recommendation(
                        st.session_state.model,
                        st.session_state.tokenizer,
                        prompt
                    )
                    
                    if recommendation:
                        display_recommendation(recommendation, similar_cases)
                    else:
                        st.error("Failed to generate recommendation")
                        
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.exception(e)


if __name__ == "__main__":
    main()