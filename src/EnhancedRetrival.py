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
    """
    def find_similar_cases(self, 
                          query_embedding: np.ndarray, 
                          k: int = 5, 
                          remove_duplicates: bool = True,
                          min_similarity: float = 0.5) -> List[Dict]:

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
    
    
    """   
    

    
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
