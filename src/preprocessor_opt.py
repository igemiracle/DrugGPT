import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import List, Dict, Union
from datetime import datetime
import logging
from tqdm import tqdm

class MIMICPreprocessor:
    """MIMIC-III data preprocessing class"""
    
    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)
        self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    

    def load_tables(self) -> Dict[str, pd.DataFrame]:
        "Load required MIMIC tables with memory optimization"
        self.logger.info("Loading MIMIC tables with optimization...")
        
        # Define dtypes for optimization
        dtypes = {
            'SUBJECT_ID': 'int32',
            'HADM_ID': 'int32',
            'ICUSTAY_ID': 'float32',
            'EXPIRE_FLAG': 'int8',
            'GENDER': 'category',
            'DOSE_VAL_RX': 'float32',
            'ROUTE': 'category',
            'DRUG': 'category'
        }
        
        required_tables = {
            'NOTEEVENTS': [
                'SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CATEGORY', 'TEXT'
            ],
            'PRESCRIPTIONS': [
                'SUBJECT_ID', 'HADM_ID', 'STARTDATE', 'ENDDATE',
                'DRUG_TYPE', 'DRUG', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC',
                'DOSE_VAL_RX', 'DOSE_UNIT_RX', 'ROUTE'
            ],
            'PATIENTS': [
                'SUBJECT_ID', 'GENDER', 'DOB', 'DOD', 
                'DOD_HOSP', 'DOD_SSN', 'EXPIRE_FLAG'
            ],
            'DIAGNOSES_ICD': [
                'SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'
            ]
        }
        
        tables = {}
        for table_name, columns in required_tables.items():
            file_path = self.data_path / f"{table_name}.csv"
            try:
                # Create dictionary of dtypes for columns that exist in dtypes
                table_dtypes = {col: dtypes[col] for col in columns if col in dtypes}
                
                # Read data in chunks for large tables
                if table_name in ['NOTEEVENTS', 'PRESCRIPTIONS']:
                    chunks = []
                    chunk_size = 100000  # Adjust based on your memory
                    for chunk in pd.read_csv(
                        file_path,
                        usecols=columns,
                        dtype=table_dtypes,
                        chunksize=chunk_size
                    ):
                        chunks.append(chunk)
                    tables[table_name] = pd.concat(chunks, ignore_index=True)
                else:
                    tables[table_name] = pd.read_csv(
                        file_path,
                        usecols=columns,
                        dtype=table_dtypes
                    )
                
                self.logger.info(f"Successfully loaded {table_name}, shape: {tables[table_name].shape}")
                self.logger.info(f"Memory usage: {tables[table_name].memory_usage().sum() / 1024**2:.2f} MB")
                
            except Exception as e:
                self.logger.error(f"Failed to load {table_name}: {str(e)}")
                raise
                
        return tables

    def process_demographics(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        """Process patient demographics with fixed datetime handling"""
        self.logger.info("Processing demographics data...")
        
        # Convert dates to datetime
        date_columns = ['DOB', 'DOD', 'DOD_HOSP', 'DOD_SSN']
        for col in date_columns:
            patients_df[col] = pd.to_datetime(patients_df[col])
        
        def calculate_age(row):
            """Calculate age using datetime objects to avoid Timedelta overflow"""
            if pd.isna(row['DOB']):
                return np.nan
                
            # Convert to datetime.datetime objects
            dob = row['DOB'].to_pydatetime()
            
            if row['EXPIRE_FLAG'] == 1 and pd.notna(row['DOD']):
                end_date = row['DOD'].to_pydatetime()
            else:
                end_date = pd.Timestamp('2200-01-01').to_pydatetime()
            
            try:
                # Calculate the difference in years
                age = (end_date.year - dob.year) - ((end_date.month, end_date.day) < (dob.month, dob.day))
                
                # If age appears to be > 300 (indicating age > 89 in real data)
                if age > 300:
                    return 90  # Mark as 90+
                return age if age >= 0 else np.nan
                
            except Exception as e:
                self.logger.warning(f"Error calculating age for DOB {dob}: {str(e)}")
                return np.nan
        
        # Calculate age
        patients_df['AGE'] = patients_df.apply(calculate_age, axis=1)
        
        # Create age groups
        patients_df['AGE_GROUP'] = pd.cut(
            patients_df['AGE'],
            bins=[0, 18, 30, 50, 70, 89, np.inf],
            labels=['0-18', '19-30', '31-50', '51-70', '71-89', '90+']
        )
        
        # Add flag for age reliability
        patients_df['age_is_accurate'] = patients_df['AGE'] <= 89
        
        def determine_mortality_status(row):
            if row['EXPIRE_FLAG'] == 0:
                return 'ALIVE'
            if pd.notna(row['DOD_HOSP']):
                return 'DIED_IN_HOSPITAL'
            elif pd.notna(row['DOD_SSN']):
                return 'DIED_SSN_RECORD'
            elif pd.notna(row['DOD']):
                return 'DIED_OTHER'
            return 'UNKNOWN'
        
        patients_df['MORTALITY_STATUS'] = patients_df.apply(determine_mortality_status, axis=1)
        
        # Log summary statistics
        self.logger.info("\nDemographic Processing Summary:")
        self.logger.info(f"Total patients: {len(patients_df)}")
        self.logger.info(f"Patients with accurate age (≤89): {patients_df['age_is_accurate'].sum()}")
        self.logger.info(f"Patients with age >89 (marked as 90+): {(~patients_df['age_is_accurate']).sum()}")
        
        return patients_df
    """
    def process_prescriptions(self, prescriptions_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Processing prescription data...")
        
        # Convert dates to datetime
        prescriptions_df['STARTDATE'] = pd.to_datetime(prescriptions_df['STARTDATE'])
        prescriptions_df['ENDDATE'] = pd.to_datetime(prescriptions_df['ENDDATE'])
        
        def calculate_duration(row):
            if pd.isna(row['STARTDATE']) or pd.isna(row['ENDDATE']):
                return np.nan
                
            try:
                start = row['STARTDATE'].to_pydatetime()
                end = row['ENDDATE'].to_pydatetime()
                duration = (end - start).days
                return duration if duration >= 0 else np.nan
            except Exception as e:
                return np.nan
        
        # Calculate duration
        prescriptions_df['DURATION'] = prescriptions_df.apply(calculate_duration, axis=1)
        
        # Process other fields
        prescriptions_df['DRUG'] = prescriptions_df['DRUG'].str.lower() if prescriptions_df['DRUG'].dtype == object else prescriptions_df['DRUG']
        prescriptions_df['DRUG_NAME_GENERIC'] = prescriptions_df['DRUG_NAME_GENERIC'].str.lower() if prescriptions_df['DRUG_NAME_GENERIC'].dtype == object else prescriptions_df['DRUG_NAME_GENERIC']
        
        # Create standardized drug identifier
        def create_drug_id(row):
            generic_name = row['DRUG_NAME_GENERIC'] if pd.notna(row['DRUG_NAME_GENERIC']) else row['DRUG']
            route = row['ROUTE'] if pd.notna(row['ROUTE']) else 'UNKNOWN'
            dose_unit = row['DOSE_UNIT_RX'] if pd.notna(row['DOSE_UNIT_RX']) else 'UNKNOWN'
            return f"{generic_name}_{route}_{dose_unit}"
        
        prescriptions_df['drug_id'] = prescriptions_df.apply(create_drug_id, axis=1)
        
        self.logger.info(f"Processed {len(prescriptions_df)} prescriptions")
        return prescriptions_df
    """

    def analyze_demographics(self, patients_df: pd.DataFrame) -> Dict:
        """Generate HIPAA-compliant demographic analysis"""
        analysis = {
            'patient_counts': {
                'total': len(patients_df),
                'by_gender': patients_df['GENDER'].value_counts().to_dict(),
                'deceased': patients_df['EXPIRE_FLAG'].sum(),
                'living': len(patients_df) - patients_df['EXPIRE_FLAG'].sum()
            },
            'age_distribution': {
                # Only include accurate statistics for ≤89 years
                'accurate_age_stats': patients_df[patients_df['age_is_accurate']]['AGE'].describe().to_dict(),
                'by_group': patients_df['AGE_GROUP'].value_counts().to_dict(),
                'by_gender': patients_df[patients_df['age_is_accurate']].groupby('GENDER')['AGE'].mean().to_dict()
            },
            'mortality': {
                'total_rate': (patients_df['EXPIRE_FLAG'].mean() * 100),
                'by_age_group': patients_df.groupby('AGE_GROUP')['EXPIRE_FLAG'].mean().multiply(100).to_dict(),
                'by_gender': patients_df.groupby('GENDER')['EXPIRE_FLAG'].mean().multiply(100).to_dict(),
                'death_location': patients_df['MORTALITY_STATUS'].value_counts().to_dict()
            },
            'data_quality': {
                'accurate_age_count': patients_df['age_is_accurate'].sum(),
                'over_89_count': (~patients_df['age_is_accurate']).sum(),
                'missing_dob': patients_df['DOB'].isna().sum(),
                'missing_death_dates': (
                    patients_df['EXPIRE_FLAG'] == 1
                ).sum() - patients_df['DOD'].notna().sum()
            }
        }
        return analysis    
    def clean_notes(self, notes_df: pd.DataFrame) -> pd.DataFrame:
        """Clean clinical notes"""
        self.logger.info("Starting clinical notes cleaning...")
        
        def clean_text(text: str) -> str:
            if not isinstance(text, str):
                return ""
            
            # Remove de-identification tags
            text = re.sub(r'\[\*\*.*?\*\*\]', '', text)
            
            # Standardize whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters
            text = re.sub(r'[^\w\s.,?!-]', ' ', text)
            
            return text.strip().lower()
        
        def extract_sections(text: str) -> Dict[str, str]:
            sections = {
                'medications': '',
                'diagnosis': '',
                'lab_results': '',
                'history': ''
            }
            
            section_patterns = {
                'medications': r'(?i)(?:medications?|meds?):.*?(?=\n\n|$)',
                'diagnosis': r'(?i)(?:diagnosis|impression|assessment):.*?(?=\n\n|$)',
                'lab_results': r'(?i)(?:lab(?:oratory)? results?|labs?):.*?(?=\n\n|$)',
                'history': r'(?i)(?:history|past medical history):.*?(?=\n\n|$)'
            }
            
            for section, pattern in section_patterns.items():
                match = re.search(pattern, text)
                if match:
                    sections[section] = match.group(0).split(':', 1)[1].strip()
            
            return sections
        
        # Keep only relevant record types
        notes_df = notes_df[notes_df['CATEGORY'].isin(['Discharge Summary', 'Physician '])]
        
        # Clean text
        tqdm.pandas(desc="Cleaning text")
        notes_df['cleaned_text'] = notes_df['TEXT'].progress_apply(clean_text)
        
        # Extract sections
        tqdm.pandas(desc="Extracting sections")
        notes_df['sections'] = notes_df['cleaned_text'].progress_apply(extract_sections)
        
        return notes_df
    
    def merge_patient_data(self, 
                        notes_df: pd.DataFrame,
                        prescriptions_df: pd.DataFrame,
                        diagnoses_df: pd.DataFrame,
                        patients_df: pd.DataFrame) -> pd.DataFrame:
        """Merge all patient data with corrected columns"""
        self.logger.info("Starting patient data merge...")
        
        # Print available columns before merge
        self.logger.info("Available columns before merge:")
        self.logger.info(f"Notes columns: {notes_df.columns.tolist()}")
        self.logger.info(f"Prescriptions columns: {prescriptions_df.columns.tolist()}")
        self.logger.info(f"Diagnoses columns: {diagnoses_df.columns.tolist()}")
        self.logger.info(f"Patients columns: {patients_df.columns.tolist()}")
        
        # First merge demographics with notes
        merged_df = notes_df.merge(
            patients_df,
            on='SUBJECT_ID',
            how='left'
        )
        
        # Merge prescriptions with selected columns
        prescription_cols = [
            'SUBJECT_ID', 'HADM_ID', 'DRUG', 'drug_id', 
            'DURATION', 'ROUTE', 'DOSE_VAL_RX', 'DOSE_UNIT_RX'  # Removed strength_value and strength_unit
        ]
        # Only include columns that exist
        available_prescription_cols = [col for col in prescription_cols if col in prescriptions_df.columns]
        
        merged_df = merged_df.merge(
            prescriptions_df[available_prescription_cols],
            on=['SUBJECT_ID', 'HADM_ID'],
            how='left'
        )
        
        # Add diagnoses
        merged_df = merged_df.merge(
            diagnoses_df[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']],
            on=['SUBJECT_ID', 'HADM_ID'],
            how='left'
        )
        
        self.logger.info(f"Completed data merge, final dataset shape: {merged_df.shape}")
        
        # Print final columns
        self.logger.info("Final columns in merged dataset:")
        self.logger.info(merged_df.columns.tolist())
        
        return merged_df

    def process_prescriptions(self, prescriptions_df: pd.DataFrame) -> pd.DataFrame:
        """Process prescriptions without strength extraction"""
        self.logger.info("Processing prescription data...")
        
        # Convert dates to datetime
        prescriptions_df['STARTDATE'] = pd.to_datetime(prescriptions_df['STARTDATE'])
        prescriptions_df['ENDDATE'] = pd.to_datetime(prescriptions_df['ENDDATE'])
        
        def calculate_duration(row):
            """Calculate duration in days using datetime objects"""
            if pd.isna(row['STARTDATE']) or pd.isna(row['ENDDATE']):
                return np.nan
                
            try:
                start = row['STARTDATE'].to_pydatetime()
                end = row['ENDDATE'].to_pydatetime()
                duration = (end - start).days
                return duration if duration >= 0 else np.nan
            except Exception as e:
                return np.nan
        
        # Calculate duration
        prescriptions_df['DURATION'] = prescriptions_df.apply(calculate_duration, axis=1)
        
        # Process drug names
        prescriptions_df['DRUG'] = prescriptions_df['DRUG'].str.lower() if prescriptions_df['DRUG'].dtype == object else prescriptions_df['DRUG']
        prescriptions_df['DRUG_NAME_GENERIC'] = prescriptions_df['DRUG_NAME_GENERIC'].str.lower() if 'DRUG_NAME_GENERIC' in prescriptions_df.columns and prescriptions_df['DRUG_NAME_GENERIC'].dtype == object else prescriptions_df['DRUG']
        
        # Create standardized drug identifier
        def create_drug_id(row):
            drug_name = row['DRUG']
            route = str(row['ROUTE']) if pd.notna(row['ROUTE']) else 'UNKNOWN'
            dose_unit = str(row['DOSE_UNIT_RX']) if pd.notna(row['DOSE_UNIT_RX']) else 'UNKNOWN'
            return f"{drug_name}_{route}_{dose_unit}"
        
        prescriptions_df['drug_id'] = prescriptions_df.apply(create_drug_id, axis=1)
        
        self.logger.info(f"Processed {len(prescriptions_df)} prescriptions")
        # Print available columns after processing
        self.logger.info("Available columns after prescription processing:")
        self.logger.info(prescriptions_df.columns.tolist())
        
        return prescriptions_df
    
    def create_train_test_split(self, 
                              df: pd.DataFrame, 
                              test_size: float = 0.03,
                              random_state: int = 42) -> tuple:
        """Create train-test split"""
        self.logger.info("Creating train-test split...")
        
        # Get unique patient IDs
        unique_patients = df['SUBJECT_ID'].unique()
        np.random.seed(random_state)
        
        # Select test set patients
        test_size_n = int(len(unique_patients) * test_size)
        test_patients = np.random.choice(unique_patients, 
                                       size=test_size_n, 
                                       replace=False)
        
        # Split data
        train_df = df[~df['SUBJECT_ID'].isin(test_patients)]
        test_df = df[df['SUBJECT_ID'].isin(test_patients)]
        
        self.logger.info(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")
        return train_df, test_df
    
    def generate_summary(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive data summary"""
        summary = {
            'demographics': {
                'total_patients': df['SUBJECT_ID'].nunique(),
                'gender_distribution': df['GENDER'].value_counts().to_dict(),
                'age_stats': df['AGE'].describe().to_dict(),
                'mortality_status': df['MORTALITY_STATUS'].value_counts().to_dict()
            },
            'medications': {
                'unique_drugs': df['DRUG'].nunique(),
                'prescriptions_per_patient': df.groupby('SUBJECT_ID')['DRUG'].count().describe().to_dict(),
                'top_medications': df['DRUG'].value_counts().head(10).to_dict(),
                'top_routes': df['ROUTE'].value_counts().head(5).to_dict()
            },
            'notes': {
                'total_notes': len(df),
                'avg_note_length': df['cleaned_text'].str.len().mean(),
                'sections_present': {
                    section: df['sections'].apply(lambda x: bool(x[section])).mean() 
                    for section in ['medications', 'diagnosis', 'lab_results', 'history']
                }
            },
            'diagnoses': {
                'unique_diagnoses': df['ICD9_CODE'].nunique(),
                'diagnoses_per_patient': df.groupby('SUBJECT_ID')['ICD9_CODE'].nunique().describe().to_dict()
            }
        }
        return summary

    def save_processed_data(self, 
                          train_df: pd.DataFrame, 
                          test_df: pd.DataFrame,
                          output_dir: Union[str, Path]):
        """Save processed data"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(output_dir / 'train_data.csv', index=False)
        test_df.to_csv(output_dir / 'test_data.csv', index=False)
        self.logger.info(f"Data saved to {output_dir}")