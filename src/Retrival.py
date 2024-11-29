import faiss
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import gc
from tqdm.notebook import tqdm
from typing import List, Dict


class ClinicalRetrievalSystem:
    def __init__(self, 
                 processed_path: str,
                 index_type: str = 'Flat',
                 n_lists: int = 100,
                 dimension: int = 768,
                 use_gpu: bool = True):
        self.processed_path = Path(processed_path)
        self.embed_dir = self.processed_path / 'embeddings'
        self.index_type = index_type
        self.dimension = dimension
        self.n_lists = min(n_lists, dimension)
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        
        self._setup_logging()
        self._setup_gpu()
        self._load_metadata()
        self._build_index()

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _setup_gpu(self):
        """Setup GPU resources"""
        if self.use_gpu:
            self.logger.info("Setting up GPU resources...")
            self.res = faiss.StandardGpuResources()
            self.co = faiss.GpuMultipleClonerOptions()
            self.co.useFloat16 = True  # 使用float16减少GPU内存使用
            self.gpu_config = faiss.GpuIndexFlatConfig()
            self.gpu_config.device = 0
            
            self.logger.info(f"Using GPU {self.gpu_config.device} for FAISS")
        else:
            self.logger.info("Using CPU for FAISS")

    def _force_numpy(self, data):
        """Force convert data to numpy array"""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data, dtype=np.float32)

    def _load_metadata(self):
        """Load embeddings with strict type checking"""
        self.logger.info("Loading embeddings and metadata...")
        
        all_embeddings = []
        self.case_texts = []
        self.case_ids = []
        
        embedding_files = sorted(self.embed_dir.glob('embeddings_*.pt'))
        total_files = len(embedding_files)
        
        for file_idx, file in enumerate(embedding_files, 1):
            try:
                self.logger.info(f"Processing file {file_idx}/{total_files}")
                data = torch.load(file, map_location='cpu')
                
                # Ensure numpy array
                emb = self._force_numpy(data['embeddings'])
                emb = emb.astype(np.float32)
                
                # Verify array
                assert isinstance(emb, np.ndarray), "Not a numpy array"
                assert emb.dtype == np.float32, f"Wrong dtype: {emb.dtype}"
                
                all_embeddings.append(emb)
                self.case_texts.extend(data['texts'])
                self.case_ids.extend(data['case_ids'])
                
                del data, emb
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"Error processing file {file}: {str(e)}")
                raise
        
        try:
            # Combine embeddings
            self.embeddings = np.vstack(all_embeddings).astype(np.float32)
            self.logger.info(f"Combined embeddings shape: {self.embeddings.shape}")
            
            # Verify final array
            assert isinstance(self.embeddings, np.ndarray), "Final embeddings not numpy array"
            assert self.embeddings.dtype == np.float32, f"Final dtype: {self.embeddings.dtype}"
            assert self.embeddings.flags.c_contiguous, "Array not C-contiguous"
            
            del all_embeddings
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Error finalizing embeddings: {str(e)}")
            raise

    def _build_index(self):
        """Build FAISS index with strict type checking"""
        self.logger.info(f"Building {self.index_type} index...")
        
        try:
            # Create base index
            if self.index_type == 'IVFFlat':
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, 
                                         self.n_lists, faiss.METRIC_INNER_PRODUCT)
            else:
                index = faiss.IndexFlatIP(self.dimension)
            
            # Convert to GPU if needed
            if self.use_gpu:
                self.index = faiss.index_cpu_to_gpu(self.res, 0, index)
            else:
                self.index = index
            
            # Ensure embeddings are ready for FAISS
            vectors = np.ascontiguousarray(self.embeddings, dtype=np.float32)
            assert isinstance(vectors, np.ndarray), "Not a numpy array"
            assert vectors.dtype == np.float32, f"Wrong dtype: {vectors.dtype}"
            assert vectors.flags.c_contiguous, "Not C-contiguous"
            
            # Train if needed
            if self.index_type == 'IVFFlat':
                self.logger.info("Training index...")
                self.index.train(vectors)
            
            # Add vectors
            self.logger.info("Adding vectors to index...")
            self.index.add(vectors)
            
            self.logger.info(f"Successfully built index with {self.index.ntotal} vectors")
            
            del vectors
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Error building index: {str(e)}")
            self.logger.error(f"Embeddings info - Type: {type(self.embeddings)}, "
                            f"Shape: {self.embeddings.shape}, "
                            f"Dtype: {self.embeddings.dtype}")
            raise

    def find_similar_cases(self, 
                          query_embedding: np.ndarray, 
                          k: int = 5, 
                          nprobe: int = 10) -> List[Dict]:
        """Find similar cases"""
        try:
            # Prepare query
            query = self._force_numpy(query_embedding)
            query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
            
            if self.index_type == 'IVFFlat':
                self.index.nprobe = min(nprobe, self.n_lists)
            
            distances, indices = self.index.search(query, k)
            
            return [
                {
                    'case_id': self.case_ids[idx],
                    'similarity': float(dist),
                    'text': self.case_texts[idx]
                }
                for dist, idx in zip(distances[0], indices[0])
                if idx != -1
            ]
            
        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            raise

    def save_index(self, save_path: str = None):
        """Save index and metadata"""
        try:
            save_path = Path(save_path) if save_path else self.processed_path / 'retrieval_system'
            save_path.mkdir(exist_ok=True)
            
            # Convert to CPU index for saving
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
            else:
                cpu_index = self.index
                
            faiss.write_index(cpu_index, str(save_path / 'case_index.faiss'))
            
            # Save metadata
            metadata = {
                'case_ids': self.case_ids,
                'case_texts': self.case_texts,
                'dimension': self.dimension,
                'index_type': self.index_type
            }
            
            torch.save(metadata, save_path / 'metadata.pt')
            self.logger.info(f"Saved system to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving index: {str(e)}")
            raise
            
    @classmethod
    def load(cls, load_path: str, use_gpu: bool = True):
        """Load saved index with GPU support"""
        load_path = Path(load_path)
        
        # Load metadata
        metadata = torch.load(load_path / 'metadata.pt')
        
        # Initialize system
        system = cls(
            processed_path=load_path.parent,
            index_type=metadata['index_type'],
            dimension=metadata['dimension'],
            use_gpu=use_gpu
        )
        
        # Load index
        cpu_index = faiss.read_index(str(load_path / 'case_index.faiss'))
        
        if use_gpu:
            if isinstance(cpu_index, faiss.IndexIVFFlat):
                system.index = faiss.GpuIndexIVFFlat(
                    system.res, 
                    cpu_index,
                    system.gpu_config
                )
            else:
                system.index = faiss.GpuIndexFlat(
                    system.res,
                    cpu_index,
                    faiss.METRIC_INNER_PRODUCT
                )
        else:
            system.index = cpu_index
        
        system.case_ids = metadata['case_ids']
        system.case_texts = metadata['case_texts']
        
        return system


    def analyze_case_patterns(self, similar_cases: List[Dict]) -> Dict:
        """Analyze patterns in similar cases"""
        analysis = {
            'medications': [],
            'diagnoses': [],
            'outcomes': []
        }
        
        # Extract medication and diagnosis patterns
        for case in similar_cases:
            text = case['text'].lower()
            
            # Extract medications
            if 'medications:' in text:
                med_section = text.split('medications:')[1].split('[sep]')[0].strip()
                analysis['medications'].extend(med_section.split(','))
            
            # Extract diagnoses
            if 'diagnoses:' in text:
                diag_section = text.split('diagnoses:')[1].split('[sep]')[0].strip()
                analysis['diagnoses'].extend(diag_section.split(','))
        
        # Count frequencies
        for key in analysis:
            if analysis[key]:
                freq = pd.Series(analysis[key]).value_counts()
                analysis[key] = [{'item': item, 'frequency': count} 
                               for item, count in freq.items()]
        
        return analysis

    def generate_case_summary(self, similar_cases: List[Dict], analysis: Dict) -> str:
        """Generate a summary of similar cases"""
        summary = []
        
        # Add overview
        summary.append(f"Found {len(similar_cases)} similar cases")
        
        # Add common diagnoses
        if analysis['diagnoses']:
            summary.append("\nCommon diagnoses:")
            for diag in analysis['diagnoses'][:3]:
                summary.append(f"- {diag['item']}: {diag['frequency']} cases")
        
        # Add common medications
        if analysis['medications']:
            summary.append("\nCommonly prescribed medications:")
            for med in analysis['medications'][:3]:
                summary.append(f"- {med['item']}: {med['frequency']} cases")
        
        # Add similarity scores
        summary.append("\nSimilarity scores:")
        for case in similar_cases:
            summary.append(f"- Case {case['case_id']}: {case['similarity']:.3f}")
        
        return "\n".join(summary)