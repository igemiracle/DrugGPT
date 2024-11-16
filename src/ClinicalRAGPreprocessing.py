import pandas as pd
from pathlib import Path
import re
import logging
from tqdm.notebook import tqdm
import gc
from typing import Dict, List
import numpy as np

class ClinicalProcessor:
    def __init__(self, data_path, output_path):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def process_all_data(self):
        """Process all data with progress tracking"""
        # Step 1: Process patient data
        self.logger.info("Step 1: Processing patient data...")
        patients_df = self.process_patient_data()
        
        # Step 2: Process diagnoses
        self.logger.info("\nStep 2: Processing diagnoses...")
        diagnoses_df = self.process_diagnoses()
        
        # Step 3: Process notes in chunks with progress bar
        self.logger.info("\nStep 3: Processing clinical notes...")
        total_notes = sum(1 for _ in pd.read_csv(
            self.data_path / "NOTEEVENTS.csv", 
            usecols=['SUBJECT_ID'], 
            chunksize=10000
        ))
        processed_notes = self.process_clinical_notes(total_notes)
        
        # Step 4: Process medications with progress tracking
        self.logger.info("\nStep 4: Processing medications...")
        total_meds = sum(1 for _ in pd.read_csv(
            self.data_path / "PRESCRIPTIONS.csv", 
            usecols=['SUBJECT_ID'], 
            chunksize=50000
        ))
        processed_meds = self.process_medications(total_meds)
        
        # Step 5: Create patient cases
        self.logger.info("\nStep 5: Creating patient cases...")
        self.create_patient_cases(patients_df, diagnoses_df)
        
        self.logger.info("\nAll processing completed!")

    def process_clinical_notes(self, total_chunks):
        """Process clinical notes with progress bar"""
        chunk_size = 10000
        notes_pbar = tqdm(total=total_chunks, desc="Processing Notes")
        
        for chunk_idx, chunk in enumerate(pd.read_csv(
            self.data_path / "NOTEEVENTS.csv",
            usecols=['SUBJECT_ID', 'HADM_ID', 'CATEGORY', 'TEXT'],
            chunksize=chunk_size
        )):
            # Filter relevant notes
            mask = chunk['CATEGORY'].isin(['Discharge Summary', 'Physician '])
            filtered_chunk = chunk[mask].copy()
            
            # Process text with nested progress bar
            tqdm.write(f"Processing chunk {chunk_idx + 1}")
            filtered_chunk['cleaned_text'] = filtered_chunk['TEXT'].progress_apply(self._clean_text)
            filtered_chunk['sections'] = filtered_chunk['cleaned_text'].progress_apply(self._extract_sections)
            
            # Save processed chunk
            filtered_chunk[['SUBJECT_ID', 'HADM_ID', 'cleaned_text', 'sections']].to_pickle(
                self.output_path / f'processed_notes_{chunk_idx}.pkl'
            )
            
            notes_pbar.update(1)
            gc.collect()
        
        notes_pbar.close()

    def process_medications(self, total_chunks):
        """Process medications with progress bar"""
        chunk_size = 50000
        meds_pbar = tqdm(total=total_chunks, desc="Processing Medications")
        
        for chunk_idx, chunk in enumerate(pd.read_csv(
            self.data_path / "PRESCRIPTIONS.csv",
            usecols=['SUBJECT_ID', 'HADM_ID', 'DRUG', 'DRUG_NAME_GENERIC', 
                    'DOSE_VAL_RX', 'DOSE_UNIT_RX', 'ROUTE'],
            chunksize=chunk_size
        )):
            # Create medication context
            tqdm.write(f"Processing medication chunk {chunk_idx + 1}")
            chunk['med_info'] = chunk.progress_apply(self._create_med_context, axis=1)
            
            # Select columns to save
            processed_chunk = chunk[['SUBJECT_ID', 'HADM_ID', 'med_info']].copy()
            processed_chunk.to_pickle(self.output_path / f'processed_meds_{chunk_idx}.pkl')
            
            meds_pbar.update(1)
            gc.collect()
        
        meds_pbar.close()

    def create_patient_cases(self, patients_df, diagnoses_df):
        """Create patient cases with progress tracking"""
        # Get total number of notes files for progress bar
        total_notes_files = len(list(self.output_path.glob('processed_notes_*.pkl')))
        cases_pbar = tqdm(total=total_notes_files, desc="Creating Patient Cases")
        
        cases = []
        batch_size = 1000
        
        for notes_idx, notes_file in enumerate(sorted(self.output_path.glob('processed_notes_*.pkl'))):
            notes_df = pd.read_pickle(notes_file)
            
            # Merge with patient and diagnosis data
            merged_df = notes_df.merge(patients_df, on='SUBJECT_ID', how='left')
            merged_df = merged_df.merge(diagnoses_df, on=['SUBJECT_ID', 'HADM_ID'], how='left')
            
            # Initialize medications for this admission
            merged_df['medications'] = None
            
            # Add medications with progress tracking
            tqdm.write(f"Processing medications for notes batch {notes_idx + 1}")
            for meds_file in tqdm(sorted(self.output_path.glob('processed_meds_*.pkl')), 
                                desc="Merging medications", leave=False):
                meds_df = pd.read_pickle(meds_file)
                
                # Merge medications without creating duplicate columns
                temp_merge = merged_df.merge(
                    meds_df,
                    on=['SUBJECT_ID', 'HADM_ID'],
                    how='left',
                    suffixes=('', '_new')
                )
                
                # Update medications list
                mask = temp_merge['med_info'].notna()
                if mask.any():
                    merged_df.loc[mask, 'medications'] = temp_merge.loc[mask, 'med_info']
                
                del temp_merge
                gc.collect()
            
            # Create cases
            for _, row in merged_df.iterrows():
                case = {
                    'subject_id': row['SUBJECT_ID'],
                    'admission_id': row['HADM_ID'],
                    'age_group': row['AGE_GROUP'],
                    'gender': row['GENDER'],
                    'diagnoses': row['ICD9_CODE'],
                    'clinical_text': row['cleaned_text'],
                    'sections': row['sections'],
                    'medications': row['medications']
                }
                cases.append(case)
            
            if len(cases) >= batch_size:
                self._save_cases_batch(cases)
                cases = []
            
            cases_pbar.update(1)
            gc.collect()
        
        if cases:
            self._save_cases_batch(cases)
        
        cases_pbar.close()

    @staticmethod
    def _create_med_context(row):
        """Create medication context string"""
        drug = row['DRUG_NAME_GENERIC'] if pd.notna(row['DRUG_NAME_GENERIC']) else row['DRUG']
        dose = f"{row['DOSE_VAL_RX']} {row['DOSE_UNIT_RX']}" if pd.notna(row['DOSE_VAL_RX']) else ""
        route = f"via {row['ROUTE']}" if pd.notna(row['ROUTE']) else ""
        
        return f"{drug} {dose} {route}".strip()


    def process_patient_data(self):
        """Process patient demographics with HIPAA compliance"""
        self.logger.info("Processing patient demographics...")
        
        patients_df = pd.read_csv(
            self.data_path / "PATIENTS.csv",
            usecols=['SUBJECT_ID', 'GENDER', 'DOB', 'DOD', 'DOD_HOSP', 'DOD_SSN', 'EXPIRE_FLAG']
        )
        
        # Convert dates
        date_cols = ['DOB', 'DOD', 'DOD_HOSP', 'DOD_SSN']
        for col in date_cols:
            patients_df[col] = pd.to_datetime(patients_df[col])
        
        # Calculate age with compliance handling
        patients_df['AGE'] = patients_df.apply(self._calculate_compliant_age, axis=1)
        
        # Create age groups
        patients_df['AGE_GROUP'] = pd.cut(
            patients_df['AGE'],
            bins=[0, 18, 30, 50, 70, 89, float('inf')],
            labels=['0-18', '19-30', '31-50', '51-70', '71-89', '90+']
        )
        
        # Add mortality status
        patients_df['MORTALITY_STATUS'] = patients_df.apply(self._determine_mortality, axis=1)
        
        # Save processed patient data
        self.logger.info(f"Processed {len(patients_df)} patients")
        patients_df.to_pickle(self.output_path / 'processed_patients.pkl')
        
        return patients_df

    def process_diagnoses(self):
        """Process diagnosis data"""
        self.logger.info("Processing diagnoses...")
        
        diagnoses_df = pd.read_csv(
            self.data_path / "DIAGNOSES_ICD.csv",
            usecols=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']
        )
        
        # Group diagnoses by admission
        grouped_diagnoses = diagnoses_df.groupby(['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].agg(list).reset_index()
        
        # Save processed diagnoses
        grouped_diagnoses.to_pickle(self.output_path / 'processed_diagnoses.pkl')
        
        return grouped_diagnoses


    @staticmethod
    def _calculate_compliant_age(row):
        """Calculate age with HIPAA compliance"""
        if pd.isna(row['DOB']):
            return None
            
        end_date = row['DOD'] if pd.notna(row['DOD']) else pd.Timestamp('2200-01-01')
        
        try:
            age = (end_date.year - row['DOB'].year) - (
                (end_date.month, end_date.day) < (row['DOB'].month, row['DOB'].day)
            )
            # Comply with HIPAA by marking ages > 89 as '90+'
            return 90 if age > 89 else age
        except:
            return None

    @staticmethod
    def _determine_mortality(row):
        """Determine mortality status"""
        if row['EXPIRE_FLAG'] == 0:
            return 'ALIVE'
        if pd.notna(row['DOD_HOSP']):
            return 'DIED_IN_HOSPITAL'
        if pd.notna(row['DOD_SSN']):
            return 'DIED_SSN_RECORD'
        if pd.notna(row['DOD']):
            return 'DIED_OTHER'
        return 'UNKNOWN'

    @staticmethod
    def _clean_text(text):
        """Clean clinical text"""
        if not isinstance(text, str):
            return ""
        
        # Remove PHI placeholders
        text = re.sub(r'\[\*\*.*?\*\*\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()

    @staticmethod
    def _extract_sections(text):
        """Extract relevant sections from clinical notes"""
        sections = {
            'medications': '',
            'diagnosis': '',
            'history': '',
            'plan': ''
        }
        
        patterns = {
            'medications': r'(?:medications?|meds?)[:\s]+(.*?)(?=\n\n|$)',
            'diagnosis': r'(?:diagnosis|impression)[:\s]+(.*?)(?=\n\n|$)',
            'history': r'(?:history|past medical history)[:\s]+(.*?)(?=\n\n|$)',
            'plan': r'(?:plan|treatment plan)[:\s]+(.*?)(?=\n\n|$)'
        }
        
        for section, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                sections[section] = match.group(1).strip()
                
        return sections


    def _save_cases_batch(self, cases, batch_idx=None):
        """Save a batch of cases"""
        if batch_idx is None:
            batch_idx = len(list(self.output_path.glob('cases_*.pkl')))
        
        pd.DataFrame(cases).to_pickle(self.output_path / f'cases_{batch_idx}.pkl')