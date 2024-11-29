import pandas as pd
import numpy as np
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.notebook import tqdm
import gc
import json
import logging
from typing import Dict, List, Tuple

class ClinicalEmbeddingProcessor:
    def __init__(self, 
                 processed_path: str,
                 model_name: str = 'emilyalsentzer/Bio_ClinicalBERT',
                 device: str = None):
        self.processed_path = Path(processed_path)
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create necessary directories
        self.train_dir = self.processed_path / 'train'
        self.val_dir = self.processed_path / 'validation'
        self.embed_dir = self.processed_path / 'embeddings'
        
        for dir_path in [self.train_dir, self.val_dir, self.embed_dir]:
            dir_path.mkdir(exist_ok=True)
            
        self._setup_logging()
        self._setup_model()

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _setup_model(self):
        """Initialize ClinicalBERT model and tokenizer"""
        self.logger.info(f"Loading {self.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def create_split(self, validation_size: float = 0.003) -> None:
        """Create training and validation splits"""
        self.logger.info("Creating train-validation split...")
        
        # Load all case files
        all_cases = []
        case_files = list(self.processed_path.glob('cases_*.pkl'))
        
        for file in tqdm(case_files, desc="Loading cases"):
            cases = pd.read_pickle(file)
            all_cases.append(cases)
        
        all_cases_df = pd.concat(all_cases, ignore_index=True)
        
        # Get unique patient IDs
        unique_patients = all_cases_df['subject_id'].unique()
        n_val = int(len(unique_patients) * validation_size)
        
        # Split patients
        np.random.seed(42)
        val_patients = np.random.choice(unique_patients, size=n_val, replace=False)
        val_mask = all_cases_df['subject_id'].isin(val_patients)
        
        # Save splits
        train_df = all_cases_df[~val_mask]
        val_df = all_cases_df[val_mask]
        
        # Save validation set
        val_df.to_pickle(self.val_dir / 'validation_cases.pkl')
        
        # Save training set in chunks
        chunk_size = 1000
        for i in tqdm(range(0, len(train_df), chunk_size), desc="Saving train chunks"):
            chunk = train_df.iloc[i:i+chunk_size]
            chunk.to_pickle(self.train_dir / f'train_cases_{i//chunk_size}.pkl')
        
        # Save split info
        split_info = {
            'total_cases': len(all_cases_df),
            'train_cases': len(train_df),
            'val_cases': len(val_df),
            'train_patients': len(unique_patients) - n_val,
            'val_patients': n_val
        }
        
        with open(self.processed_path / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        
        self.logger.info(f"Split completed: {split_info}")

    def create_embeddings(self, batch_size: int = 32, max_length: int = 512):
        """Create embeddings for training set"""
        self.logger.info("Creating embeddings for training set...")
        
        # Process each training chunk
        train_files = sorted(self.train_dir.glob('train_cases_*.pkl'))
        
        for file_idx, train_file in enumerate(train_files):
            self.logger.info(f"Processing file {file_idx+1}/{len(train_files)}")
            cases = pd.read_pickle(train_file)
            
            # Process in batches
            embeddings = []
            texts = []
            
            for idx in tqdm(range(0, len(cases), batch_size), desc="Creating embeddings"):
                batch = cases.iloc[idx:idx+batch_size]
                batch_texts = [self._prepare_text_for_embedding(case) for _, case in batch.iterrows()]
                texts.extend(batch_texts)
                
                # Create embeddings
                batch_embeddings = self._create_batch_embeddings(batch_texts, max_length)
                embeddings.append(batch_embeddings)
                
                del batch_embeddings
                gc.collect()
            
            # Combine and save embeddings
            combined_embeddings = torch.cat(embeddings, dim=0)
            
            # Save embeddings and texts
            torch.save({
                'embeddings': combined_embeddings,
                'texts': texts,
                'case_ids': cases['subject_id'].tolist()
            }, self.embed_dir / f'embeddings_{file_idx}.pt')
            
            del combined_embeddings, embeddings, texts
            gc.collect()

    def _prepare_text_for_embedding(self, case: pd.Series) -> str:
        """Prepare case text for embedding"""
        sections = []
        
        # Add demographics
        sections.append(f"Patient: {case['age_group']} {case['gender']}")
        
        # Add diagnoses
        if isinstance(case['diagnoses'], list):
            sections.append(f"Diagnoses: {', '.join(case['diagnoses'])}")
        
        # Add medications if available
        if pd.notna(case.get('medications')):
            sections.append(f"Medications: {case['medications']}")
        
        # Add clinical text sections
        if isinstance(case['sections'], dict):
            for section_name, content in case['sections'].items():
                if content:
                    sections.append(f"{section_name}: {content}")
        
        return " [SEP] ".join(sections)

    @torch.no_grad()
    def _create_batch_embeddings(self, texts: List[str], max_length: int) -> torch.Tensor:
        """Create embeddings for a batch of texts"""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model(**inputs)
        
        # Use [CLS] token embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings.cpu()

    def get_case_embedding(self, case_text: str) -> torch.Tensor:
        """Get embedding for a single case"""
        return self._create_batch_embeddings([case_text], 512)[0]