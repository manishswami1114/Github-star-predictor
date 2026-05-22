# =============================================================================
# CACHE DIRECTORY OVERRIDE - MUST BE FIRST (before any transformers imports)
# =============================================================================
import os
import sys
from pathlib import Path

# Set cache to project-local writable directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Adjust: main/src -> project root
CACHE_DIR = PROJECT_ROOT / ".hf_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set environment variables BEFORE any transformers/sentence_transformers imports
os.environ['HF_HOME'] = str(CACHE_DIR)
os.environ['TRANSFORMERS_CACHE'] = str(CACHE_DIR)
os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(CACHE_DIR)
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Reduce warnings

# Verify cache is writable
try:
    test_file = CACHE_DIR / '.write_test'
    test_file.write_text('test')
    test_file.unlink()
except PermissionError:
    # Fallback to system temp
    import tempfile
    CACHE_DIR = Path(tempfile.gettempdir()) / 'hf_cache_github'
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(CACHE_DIR)
    os.environ['TRANSFORMERS_CACHE'] = str(CACHE_DIR)
    print(f"⚠️  Using fallback cache: {CACHE_DIR}", file=sys.stderr)

print(f"✅ Cache directory: {CACHE_DIR}", file=sys.stderr)
# =============================================================================
# END CACHE OVERRIDE
# =============================================================================

# NOW import other libraries
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
import torch
import logging
from tqdm import tqdm
import pickle
import gc
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("week3_embeddings.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class READMEEmbeddingExtractor:
    """Extract and cache README embeddings for efficient reuse."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.cache_dir = Path("./data/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model(self):
        """Load transformer model with explicit cache directory."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_dir = os.environ.get('HF_HOME', str(CACHE_DIR))
        
        logger.info(f"Loading model '{self.model_name}' on {device}")
        logger.info(f"Cache directory: {cache_dir}")
        
        try:
            self.model = SentenceTransformer(
                self.model_name, 
                device=device,
                cache_folder=cache_dir
            )
        except Exception as e:
            logger.error(f"Failed to load '{self.model_name}': {e}")
            logger.info("Falling back to 'paraphrase-MiniLM-L3-v2' (smaller, more reliable)")
            self.model_name = "paraphrase-MiniLM-L3-v2"
            self.model = SentenceTransformer(
                self.model_name,
                device=device,
                cache_folder=cache_dir
            )
        
        embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"✅ Model loaded: {self.model_name}, dim={embedding_dim}")
        return self
    
    def extract_readme_text(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract README text from dataframe.
        
        Strategy:
        1. Use 'description' field (most repos don't have full README in API)
        2. Fallback to repo name if description missing
        3. Truncate to 512 tokens (model limit)
        4. Handle null/empty values
        """
        logger.info("Extracting text from description/name fields...")
        texts = []
        
        for idx, row in df.iterrows():
            # Priority: description > name > fallback
            text = row.get('description')
            if pd.isna(text) or str(text).strip() == "":
                text = row.get('name', 'No description available')
            
            # Clean and truncate
            text = str(text).strip()
            if len(text) > 2000:  # Conservative truncation
                text = text[:2000] + "..."
            
            texts.append(text)
        
        return pd.Series(texts, index=df.index)
    
    def generate_embeddings(self, texts: pd.Series, cache_key: str = "readme_embeddings") -> np.ndarray:
        """
        Generate embeddings with caching to avoid re-computation.
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Check cache first
        if cache_file.exists():
            logger.info(f"📂 Loading cached embeddings from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Generate new embeddings
        logger.info(f"🚀 Generating embeddings for {len(texts):,} repositories...")
        
        # Filter empty texts
        valid_mask = texts.str.len() > 0
        valid_indices = texts[valid_mask].index.tolist()
        valid_texts = texts[valid_mask].tolist()
        
        # Initialize output array
        embedding_dim = self.model.get_sentence_embedding_dimension()
        embeddings = np.zeros((len(texts), embedding_dim), dtype=np.float32)
        
        # Batch processing with progress bar
        total_batches = (len(valid_texts) + self.batch_size - 1) // self.batch_size
        for i in tqdm(range(0, len(valid_texts), self.batch_size), 
                     total=total_batches, desc="Embedding"):
            batch = valid_texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch, 
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalization for better similarity
            )
            batch_indices = valid_indices[i:i + len(batch)]
            embeddings[batch_indices] = batch_embeddings
            
            # Memory cleanup every 10 batches
            if i % (self.batch_size * 10) == 0:
                gc.collect()
        
        # Handle empty texts (use zero vector)
        if (~valid_mask).any():
            logger.info(f"ℹ️  Using zero vectors for {(~valid_mask).sum()} empty descriptions")
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        logger.info(f"💾 Embeddings cached to {cache_file}")
        
        return embeddings
    
    def add_embeddings_to_df(self, df: pd.DataFrame, prefix: str = "readme_emb") -> pd.DataFrame:
        """Add embedding columns to dataframe."""
        if self.model is None:
            self.load_model()
        
        # Extract texts
        texts = self.extract_readme_text(df)
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Add as separate columns
        embedding_dim = embeddings.shape[1]
        logger.info(f"Adding {embedding_dim} embedding columns with prefix '{prefix}'...")
        
        for i in range(embedding_dim):
            df[f"{prefix}_{i}"] = embeddings[:, i]
        
        logger.info(f"✅ Added {embedding_dim} embedding columns")
        return df
    
    def get_embedding_feature_cols(self, prefix: str = "readme_emb", dim: int = 384) -> list:
        """Generate list of embedding column names for modeling."""
        return [f"{prefix}_{i}" for i in range(dim)]


def run_week3_pipeline(data_dir: str = "/Users/manishswami/developer/Github_star_project/data/source", output_dir: str = "/Users/manishswami/developer/Github_star_project/data/processed"):
    """
    Execute Week 3: Add README embeddings to dataset.
    """
    print("="*60)
    print("WEEK 3: README TEXT EMBEDDINGS")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60)
    
    # Load data
    files = list(Path(data_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    print(f"\n📂 Loading {len(files)} parquet files...")
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    print(f"   ✅ Loaded {len(df):,} repositories")
    
    # Preprocess (same as baseline)
    print("\n🔧 Preprocessing...")
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
    df['ingested_at'] = pd.to_datetime(df['ingested_at'], utc=True)
    if df['ingested_at'].dt.tz is None:
        df['ingested_at'] = df['ingested_at'].dt.tz_localize('UTC')
    df['age_days'] = (df['ingested_at'] - df['created_at']).dt.days
    df['log_stars'] = np.log1p(df['stargazers_count'])
    df['language'] = df['language'].fillna('Unknown').astype(str)
    print(f"   ✅ Preprocessing complete")
    
    # Initialize extractor
    print("\n🤖 Loading embedding model...")
    extractor = READMEEmbeddingExtractor(model_name="all-MiniLM-L6-v2", batch_size=32)
    extractor.load_model()
    
    # Add embeddings
    print("\n📝 Generating README embeddings...")
    df_enriched = extractor.add_embeddings_to_df(df, prefix="readme_emb")
    
    # Save enriched dataset
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving enriched dataset to {output_dir}/...")
    # Save in chunks to avoid memory issues
    chunk_size = 20000
    total_chunks = (len(df_enriched) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(df_enriched), chunk_size):
        chunk = df_enriched.iloc[i:i+chunk_size]
        output_file = output_path / f"repos_enriched_part_{i//chunk_size}.parquet"
        chunk.to_parquet(output_file, compression='snappy', index=False)
        logger.info(f"   Saved chunk {i//chunk_size + 1}/{total_chunks} to {output_file}")
    
    print(f"\n{'='*60}")
    print("WEEK 3 COMPLETE")
    print("="*60)
    print(f"   ✅ Total rows: {len(df_enriched):,}")
    print(f"   ✅ Total columns: {len(df_enriched.columns)}")
    print(f"   ✅ Embedding columns: 384 (readme_emb_0 ... readme_emb_383)")
    print(f"   ✅ Output directory: {output_dir}/")
    print(f"   ✅ Cache directory: {extractor.cache_dir}/")
    
    return df_enriched, extractor


if __name__ == "__main__":
    try:
        df_enriched, extractor = run_week3_pipeline()
        
        # Quick validation
        print("\n" + "="*60)
        print("EMBEDDING VALIDATION")
        print("="*60)
        emb_cols = [c for c in df_enriched.columns if c.startswith("readme_emb_")]
        print(f"   Embedding columns: {len(emb_cols)}")
        print(f"   Sample values (first repo, first 5 dims):")
        print(f"   {df_enriched[emb_cols[:5]].iloc[0].values}")
        print(f"   Null check: {df_enriched[emb_cols].isnull().sum().sum()} nulls")
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("""
🟢 WEEK 3 COMPLETE - READY FOR FINAL MODEL

   Next Actions:
   1. Update eda_and_baseline.py with embedding features:
      
      feature_cols = [
          'age_days', 'forks_count', 'open_issues_count', 
          'size', 'language'
      ] + extractor.get_embedding_feature_cols(prefix="readme_emb", dim=384)
   
   2. Retrain baseline model with new features
   
   3. Expected RMSLE improvement:
      Before: 0.6921
      After:  0.55-0.62 (with embeddings)
   
   4. Continue with Week 4 (Owner Reputation, Topic Flags)
        """)
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.exception("Week 3 pipeline failed")
        raise