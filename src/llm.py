#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import logging
import pandas as pd
import sys

pd.set_option('display.max_columns', None)

sys.path.append('../src')


# In[2]:


# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")


# In[3]:


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")


# In[4]:


# 1. First recreate your embedding processor
from Embedding import ClinicalEmbeddingProcessor
processor = ClinicalEmbeddingProcessor(
    processed_path='../data/processed',
    model_name='emilyalsentzer/Bio_ClinicalBERT'
)

# 2. Recreate base retrieval system
from Retrival import ClinicalRetrievalSystem
retriever = ClinicalRetrievalSystem(
    processed_path='../data/processed',
    index_type='Flat',
    use_gpu=True
)


# In[32]:


# 3. Recreate enhanced retrieval system
from EnhancedRetrival import EnhancedClinicalRetrieval
enhanced_retriever = EnhancedClinicalRetrieval(
    base_retriever=retriever,
    icd9_map_path='../data/icd9_map.csv'
)


# In[35]:


import faiss
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict
from collections import defaultdict

class EnhancedClinicalRetrieval:
    def __init__(self, base_retriever, icd9_map_path: str = None):
        """
        Enhanced retrieval system with additional features
        
        Args:
            base_retriever: Base retrieval system instance
            icd9_map_path: Path to ICD-9 code mapping file
        """
        self.retriever = base_retriever
        self._load_icd9_map(icd9_map_path)
        self.setup_logging()
    
    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
    
    def _load_icd9_map(self, icd9_map_path):
        """Load ICD-9 code mapping"""
        if icd9_map_path and Path(icd9_map_path).exists():
            self.icd9_map = pd.read_csv(icd9_map_path)
        else:
            self.logger.warning("ICD-9 mapping file not found")
            self.icd9_map = None
    
    def _normalize_similarity(self, similarity: float) -> float:

        #return (similarity + 1) / 2
        # 使用 sigmoid 函数归一化
        return 1 / (1 + np.exp(-similarity/100))

    
    def _convert_icd9_codes(self, codes: List[str]) -> List[str]:
        """
        Convert ICD-9 codes to readable diagnoses using SHORT_TITLE
        
        Args:
            codes: List of ICD-9 codes
            
        Returns:
            List of readable diagnosis strings
        """
        if self.icd9_map is None:
            return codes
        
        readable_codes = []
        for code in codes:
            code = code.strip()
            match = self.icd9_map[self.icd9_map['ICD9_CODE'] == code]
            if not match.empty:
                # Use SHORT_TITLE for brief description
                readable_codes.append(f"{code}: {match.iloc[0]['SHORT_TITLE']}")
            else:
                readable_codes.append(code)
        return readable_codes
    

    
    def find_similar_cases(self, 
                          query_embedding: np.ndarray, 
                          k: int = 5, 
                          remove_duplicates: bool = True,
                          min_similarity: float = 0.5) -> List[Dict]:

        # Get initial results
        initial_results = self.retriever.find_similar_cases(
            query_embedding, 
            k=k*2 if remove_duplicates else k
        )
        
        processed_results = []
        seen_cases = set()
        
        for result in initial_results:
            try:
                # Normalize similarity score
                similarity = self._normalize_similarity(result['similarity'])
                if similarity < min_similarity:
                    continue

 
                # Parse case text
                case_data = self._parse_case_text(result['text'])
                case_id = result['case_id']
                
                # Check for duplicates
                if remove_duplicates and case_id in seen_cases:
                    continue
                seen_cases.add(case_id)
                
                # Convert ICD-9 codes if available
                diagnoses = case_data.get('diagnoses', [])
                if diagnoses:
                    case_data['diagnoses'] = self._convert_icd9_codes(diagnoses)
                
                # Create enhanced result object
                enhanced_result = {
                    'case_id': case_id,
                    'similarity': similarity,
                    'demographics': case_data.get('demographics', ''),
                    'diagnoses': case_data.get('diagnoses', []),
                    'medications': case_data.get('medications', ''),
                    'original_text': result['text']
                }
                
                processed_results.append(enhanced_result)
                
                if len(processed_results) >= k:
                    break
                    
            except Exception as e:
                self.logger.warning(f"Error processing result: {str(e)}")
                continue
        
        return processed_results
 

        
    def generate_summary(self, results: List[Dict]) -> str:
        """Generate concise clinical summary"""
        # Count patterns
        demographics = defaultdict(int)
        diagnoses = defaultdict(int)
        medications = defaultdict(int)
        
        for result in results:
            demographics[result['demographics']] += 1
            for diag in result['diagnoses']:
                diagnoses[diag] += 1
            for med in result['medications'].split(','):
                if med.strip():
                    medications[med.strip()] += 1
        
        # Format summary
        summary = [
            f"Summary of {len(results)} Similar Cases",
            "-" * 30,
            
            "\nDemographics:",
            *[f"- {demo}: {count} cases" 
              for demo, count in sorted(demographics.items(), 
                                      key=lambda x: x[1], reverse=True)],
            
            "\nCommon Diagnoses:",
            *[f"- {diag}: {count} cases" 
              for diag, count in sorted(diagnoses.items(), 
                                      key=lambda x: x[1], reverse=True)[:5]],
            
            "\nPrescribed Medications:",
            *[f"- {med}: {count} cases" 
              for med, count in sorted(medications.items(), 
                                     key=lambda x: x[1], reverse=True)[:5]]
        ]
        
        return "\n".join(summary)

    def _parse_case_text(self, text: str) -> Dict:
        """Parse case text into structured data"""
        sections = text.split('[SEP]')
        case_data = {}
        
        for section in sections:
            section = section.strip()
            if 'Patient:' in section:
                case_data['demographics'] = section.replace('Patient:', '').strip()
            elif 'Diagnoses:' in section:
                codes = section.replace('Diagnoses:', '').strip().split(',')
                case_data['diagnoses'] = [code.strip() for code in codes]
            elif 'Medications:' in section:
                case_data['medications'] = section.replace('Medications:', '').strip()
        
        return case_data



    def analyze_similar_cases(self, results: List[Dict]) -> Dict:
        """Analyze patterns in similar cases"""
        analysis = {
            'demographics': defaultdict(int),
            'diagnoses': defaultdict(int),
            'medications': defaultdict(int),
            'similarity_stats': {
                'mean': np.mean([r['similarity'] for r in results]),
                'std': np.std([r['similarity'] for r in results]),
                'min': min([r['similarity'] for r in results]),
                'max': max([r['similarity'] for r in results])
            }
        }
        
        # Count patterns
        for result in results:
            # Demographics
            analysis['demographics'][result['demographics']] += 1
            
            # Diagnoses
            for diagnosis in result['diagnoses']:
                analysis['diagnoses'][diagnosis] += 1
            
            # Medications
            meds = result['medications'].split(',')
            for med in meds:
                if med.strip():
                    analysis['medications'][med.strip()] += 1
        
        # Convert to regular dict and sort
        for key in ['demographics', 'diagnoses', 'medications']:
            analysis[key] = dict(sorted(
                analysis[key].items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
        
        return analysis

    def generate_summary(self, results: List[Dict]) -> str:
        """Generate summary of similar cases analysis"""
        analysis = self.analyze_similar_cases(results)
        
        summary = [
            "Similar Cases Analysis Summary",
            "=" * 30,
            f"\nFound {len(results)} similar cases",
            f"Average similarity: {analysis['similarity_stats']['mean']:.3f}",
            
            "\nCommon Demographics:",
            "-" * 20
        ]
        
        for demo, count in list(analysis['demographics'].items())[:3]:
            summary.append(f"- {demo}: {count} cases")
        
        summary.extend([
            "\nTop Diagnoses:",
            "-" * 20
        ])
        
        for diag, count in list(analysis['diagnoses'].items())[:5]:
            summary.append(f"- {diag}: {count} cases")
        
        summary.extend([
            "\nCommonly Prescribed Medications:",
            "-" * 20
        ])
        
        for med, count in list(analysis['medications'].items())[:5]:
            summary.append(f"- {med}: {count} cases")
        
        return "\n".join(summary)


# In[ ]:





# In[68]:


from typing import List, Dict, Optional
from transformers import AutoConfig

@dataclass
class RecommendationResult:
    recommendation: str
    similar_cases: List[Dict]
    confidence: float
    evidence: str
    risks: List[str]


class IntegratedMedicalRAG:
    def __init__(
        self,
        llm_model_name: str = 'meta-llama/Llama-3.2-1B',
        device: str = 'cuda',
        retriever: Optional[EnhancedClinicalRetrieval] = None,
        processor: Optional[ClinicalEmbeddingProcessor] = None,
        min_similarity: float = 0.5,
        top_k: int = 5
    ):
        self.device = device
        self.min_similarity = min_similarity
        self.top_k = top_k
        
        # First load the tokenizer and add special tokens
        logging.info(f"Loading {llm_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        
        # Add special tokens if they don't exist
        special_tokens = {
            'pad_token': '[PAD]',
            'eos_token': '</s>',  # Common end of sequence token
            'bos_token': '<s>',   # Beginning of sequence token
        }
        
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens)
        
        # Load the model with ignore_mismatched_sizes
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            ignore_mismatched_sizes=True
        ).to(device)
        
        # Resize embeddings if tokens were added
        if num_added_tokens > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Update model configuration
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        
        # Set generation config
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.bos_token_id = self.tokenizer.bos_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        
        self.retriever = retriever
        self.processor = processor


    def _prepare_case_text(self, case_data: Dict) -> str:
        """Prepare case text for embedding."""
        case_text = f"Patient: {case_data.get('age_group', '')} {case_data.get('gender', '')}"
        if 'diagnoses' in case_data:
            case_text += f" [SEP]Diagnoses: {', '.join(case_data['diagnoses'])}"
        if 'medications' in case_data:
            case_text += f" [SEP]Medications: {case_data['medications']}"
        if 'sections' in case_data and 'history' in case_data['sections']:
            case_text += f" [SEP]History: {case_data['sections']['history']}"
        return case_text

    def _generate_case_summary(self, similar_cases: List[Dict], current_case: Dict) -> str:
        """Generate a summary of similar cases and current case context."""
        summary_parts = []
        
        # Summarize current case
        summary_parts.append("Current Case Overview:")
        summary_parts.append(f"- Patient Demographics: {current_case.get('age_group', 'Unknown')} {current_case.get('gender', 'Unknown')}")
        summary_parts.append(f"- Current Diagnoses: {', '.join(current_case.get('diagnoses', []))}")
        summary_parts.append(f"- Current Medications: {current_case.get('medications', 'None')}")
        
        # Summarize similar cases
        if similar_cases:
            summary_parts.append("\nRelevant Case Patterns:")
            diagnoses_counts = {}
            medication_patterns = {}
            
            for case in similar_cases:
                # Count diagnoses
                for diagnosis in case.get('diagnoses', []):
                    diagnoses_counts[diagnosis] = diagnoses_counts.get(diagnosis, 0) + 1
                
                # Track medication patterns
                if 'medications' in case:
                    meds = case['medications'].split(',')
                    for med in meds:
                        med = med.strip()
                        medication_patterns[med] = medication_patterns.get(med, 0) + 1
            
            # Add common diagnoses
            common_diagnoses = sorted(diagnoses_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            if common_diagnoses:
                summary_parts.append("Common Diagnoses in Similar Cases:")
                for diagnosis, count in common_diagnoses:
                    summary_parts.append(f"- {diagnosis}: {count} cases")
            
            # Add common medications
            common_meds = sorted(medication_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
            if common_meds:
                summary_parts.append("\nCommon Medications in Similar Cases:")
                for med, count in common_meds:
                    summary_parts.append(f"- {med}: {count} cases")
        
        return "\n".join(summary_parts)

    def process_case(self, case_data: Dict) -> RecommendationResult:
        """Process a single case through the RAG pipeline."""
        # Get similar cases
        case_text = self._prepare_case_text(case_data)
        case_embedding = self.processor.get_case_embedding(case_text)
        similar_cases = self.retriever.find_similar_cases(
            case_embedding,
            k=self.top_k,
            remove_duplicates=True
        )
        
        # Filter similar cases by relevance
        filtered_cases = self._filter_relevant_cases(similar_cases, case_data.get('diagnoses', []))
        
        # Generate case summary
        case_summary = self._generate_case_summary(filtered_cases, case_data)
        
        # Generate recommendation
        prompt = self._construct_prompt(case_data, filtered_cases, case_summary)
        recommendation = self._generate_recommendation(prompt)
        
        # Parse and validate response
        result = self._parse_response(recommendation, filtered_cases)
        result = self._validate_recommendation(result, case_data)
        
        return result

    # ... (保持其他方法不变) ...

    def _filter_relevant_cases(self, cases: List[Dict], diagnoses: List[str]) -> List[Dict]:
        """Filter cases to keep only those most relevant to current diagnoses."""
        relevant_cases = []
        for case in cases:
            case_diagnoses = case.get('diagnoses', [])
            if any(diag in case_diagnoses for diag in diagnoses):
                relevant_cases.append(case)
        return relevant_cases[:self.top_k]

    def _validate_recommendation(self, result: RecommendationResult, case_data: Dict) -> RecommendationResult:
        """Validate and enhance recommendation quality."""
        required_sections = ['Treatment Plan', 'Clinical Rationale', 'Monitoring Plan']
        
        if not all(section in result.recommendation for section in required_sections):
            logging.warning("Generated recommendation missing required sections")
            result.confidence *= 0.8
        
        if not any(med in result.recommendation.lower() for med in case_data.get('medications', '').lower().split(',')):
            logging.warning("Generated recommendation doesn't reference current medications")
            result.confidence *= 0.9
            
        return result

            
    def _construct_prompt(self, case_data: Dict, similar_cases: List[Dict], case_summary: str) -> str:
        """Construct a more focused and structured prompt."""
        return f"""You are an experienced medical professional providing treatment recommendations. Review the following patient case:

Patient Profile:
- Age Group: {case_data.get('age_group', 'Unknown')}
- Gender: {case_data.get('gender', 'Unknown')}
- Primary Diagnoses: {', '.join(case_data.get('diagnoses', []))}
- Current Medications: {case_data.get('medications', 'None')}
- Clinical History: {case_data.get('sections', {}).get('history', 'Not available')}

Based on this information, provide a detailed clinical recommendation using the following structure:

1. Treatment Plan:
- Current medication adjustments (if needed)
- New medication recommendations
- Specific dosing instructions
- Lifestyle modifications

2. Clinical Rationale:
- Justification for each recommendation
- Expected benefits
- Treatment goals

3. Monitoring Plan:
- Specific parameters to monitor
- Frequency of monitoring
- Target values
- Warning signs to watch for

Keep your response focused, evidence-based, and clinically precise. Include specific medications, dosages, and monitoring parameters."""
    
    def _generate_recommendation(self, prompt: str) -> str:
        """Generate recommendation with proper padding configuration."""
        # Add BOS token at the start if needed
        if not prompt.startswith(self.tokenizer.bos_token):
            prompt = self.tokenizer.bos_token + " " + prompt
            
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True
        ).to(self.device)
        
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_length=1024,
                    min_length=200,
                    temperature=0.8,
                    top_p=0.92,
                    repetition_penalty=1.3,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    num_return_sequences=1,
                    use_cache=True
                )
                
                return self.tokenizer.decode(
                    outputs[0], 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
            except Exception as e:
                logging.error(f"Error during generation: {str(e)}")
                return f"Error generating response: {str(e)}"
                
    def _parse_response(self, response: str, similar_cases: List[Dict]) -> RecommendationResult:
        """Parse response with improved handling."""
        sections = {
            'treatment': '',
            'evidence': '',
            'risks': []
        }
        
        # More robust section parsing
        current_section = None
        lines = response.split('\n')
        
        for line in lines:
            if '1. Treatment Plan:' in line or 'Treatment Plan:' in line:
                current_section = 'treatment'
                continue
            elif '2. Clinical Rationale:' in line or 'Clinical Rationale:' in line:
                current_section = 'evidence'
                continue
            elif '3. Risks & Monitoring:' in line or 'Risks:' in line:
                current_section = 'risks'
                continue
                
            if current_section == 'treatment' and line.strip():
                sections['treatment'] += line.strip() + '\n'
            elif current_section == 'evidence' and line.strip():
                sections['evidence'] += line.strip() + '\n'
            elif current_section == 'risks' and line.strip():
                if line.startswith('-') or line.startswith('•'):
                    sections['risks'].append(line.strip().lstrip('-•').strip())
                else:
                    sections['risks'].append(line.strip())
        
        # Clean up sections
        sections['treatment'] = sections['treatment'].strip()
        sections['evidence'] = sections['evidence'].strip()
        sections['risks'] = [risk for risk in sections['risks'] if risk]
        
        # If sections are empty, try alternate parsing
        if not any(sections.values()):
            parts = response.split('\n\n')
            if len(parts) >= 3:
                sections['treatment'] = parts[0]
                sections['evidence'] = parts[1]
                sections['risks'] = [r.strip() for r in parts[2].split('\n') if r.strip()]
        
        # Calculate confidence based on content
        confidence = self._calculate_confidence(sections, similar_cases)
        
        return RecommendationResult(
            recommendation=sections['treatment'],
            similar_cases=similar_cases,
            confidence=confidence,
            evidence=sections['evidence'],
            risks=sections['risks']
        )
    
    def _calculate_confidence(self, sections: Dict, similar_cases: List[Dict]) -> float:
        """Calculate confidence score based on response quality and similar cases."""
        base_confidence = 0.5
        
        # Check response completeness
        if sections['treatment'] and sections['evidence'] and sections['risks']:
            base_confidence += 0.2
        
        # Check for specific medical content
        medical_indicators = ['mg', 'dose', 'monitor', 'adjust', 'increase', 'decrease']
        content = sections['treatment'].lower()
        medical_term_count = sum(1 for term in medical_indicators if term in content)
        base_confidence += min(0.2, medical_term_count * 0.03)
        
        # Factor in similar cases similarity scores
        if similar_cases:
            avg_similarity = np.mean([case['similarity'] for case in similar_cases])
            base_confidence *= (0.5 + 0.5 * avg_similarity)  # Weight similarity at 50%
        
        return min(1.0, base_confidence)

