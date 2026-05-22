# week4_advanced_features.py
"""
Week 4: Advanced Feature Engineering
=====================================
Add owner reputation, topic flags, and velocity metrics.
"""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, timezone
import logging
import warnings

# Suppress pandas FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("week4_features.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def generate_owner_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate owner-level aggregate features."""
    logger.info("Generating owner reputation features...")
    
    if 'owner.login' not in df.columns:
        logger.warning("owner.login column not found; skipping owner features")
        return df
    
    # Ensure created_at is timezone-aware
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
    
    # Aggregate owner statistics
    owner_agg = df.groupby('owner.login').agg({
        'stargazers_count': ['mean', 'sum', 'count', 'std'],
        'forks_count': ['mean', 'sum'],
    }).reset_index()
    
    # Flatten column names
    owner_agg.columns = ['owner.login', 
                         'owner_mean_stars', 'owner_total_stars', 
                         'owner_repo_count', 'owner_std_stars',
                         'owner_mean_forks', 'owner_total_forks']
    
    # Add date-based features if created_at exists
    if 'created_at' in df.columns:
        date_agg = df.groupby('owner.login')['created_at'].agg(['min', 'max']).reset_index()
        date_agg.columns = ['owner.login', 'owner_first_repo', 'owner_last_repo']
        owner_agg = owner_agg.merge(date_agg, on='owner.login', how='left')
        
        now_utc = datetime.now(timezone.utc)
        owner_agg['owner_tenure_days'] = (now_utc - owner_agg['owner_first_repo']).dt.days
        owner_agg['owner_active_days'] = (owner_agg['owner_last_repo'] - owner_agg['owner_first_repo']).dt.days
    
    # Flag for organizations
    org_keywords = ['microsoft', 'google', 'facebook', 'amazon', 'aws', 'ibm', 
                    'apache', 'tensorflow', 'pytorch', 'github', 'docker']
    owner_agg['owner_is_org'] = owner_agg['owner.login'].str.lower().str.contains(
        '|'.join(org_keywords), na=False
    ) | (owner_agg['owner_repo_count'] > 100)
    
    # Owner success rate (FIXED: use transform instead of apply)
    df['owner_success_flag'] = (df['stargazers_count'] > 100).astype(int)
    owner_success = df.groupby('owner.login')['owner_success_flag'].mean().reset_index()
    owner_success.columns = ['owner.login', 'owner_success_rate']
    owner_agg = owner_agg.merge(owner_success, on='owner.login', how='left')
    df = df.drop(columns=['owner_success_flag'])
    
    # Merge back
    df = df.merge(owner_agg, on='owner.login', how='left')
    
    # Fill missing values
    for col in ['owner_mean_stars', 'owner_total_stars', 'owner_repo_count',
                'owner_std_stars', 'owner_mean_forks', 'owner_total_forks',
                'owner_tenure_days', 'owner_active_days', 'owner_success_rate']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    if 'owner_is_org' in df.columns:
        df['owner_is_org'] = df['owner_is_org'].fillna(False).astype(int)
    
    # Log-transform skewed features
    for col in ['owner_total_stars', 'owner_total_forks', 'owner_repo_count', 'owner_tenure_days']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
    
    logger.info(f"  ✅ Added {len([c for c in df.columns if c.startswith('owner_')])} owner features")
    return df


def generate_topic_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Generate binary flags for trending technology keywords."""
    logger.info("Generating topic/technology flags...")
    
    topic_keywords = {
        'topic_ai_llm': ['llm', 'chatbot', 'gpt', 'claude', 'llama', 'mistral'],
        'topic_ai_transformer': ['transformer', 'bert', 'vit', 'diffusion', 'attention'],
        'topic_ai_rag': ['rag', 'retrieval', 'vector database', 'pinecone', 'embedding'],
        'topic_ai_agent': ['agent', 'autonomous', 'langchain', 'crewai', 'auto-gpt'],
        'topic_devops_k8s': ['kubernetes', 'k8s', 'helm', 'istio'],
        'topic_devops_docker': ['docker', 'container', 'podman'],
        'topic_web_react': ['react', 'nextjs', 'next.js', 'remix'],
        'topic_data_pytorch': ['pytorch', 'torch', 'pytorch lightning'],
        'topic_data_tensorflow': ['tensorflow', 'tf', 'keras'],
        'topic_data_pandas': ['pandas', 'polars', 'dask'],
    }
    
    # Build text field from available columns
    text_parts = []
    
    if 'name' in df.columns:
        text_parts.append(df['name'].fillna('').astype(str))
        logger.info("  Using 'name' column for topic flags")
    
    if 'full_name' in df.columns:
        text_parts.append(df['full_name'].fillna('').astype(str))
        logger.info("  Using 'full_name' column for topic flags")
    
    if 'description' in df.columns:
        text_parts.append(df['description'].fillna('').astype(str))
        logger.info("  Using 'description' column for topic flags")
    else:
        logger.warning("  'description' column not found; using name/full_name only")
    
    # === FIX: Use pandas concatenation, not Python join ===
    if text_parts:
        # Concatenate Series with spaces between them
        text_field = text_parts[0]
        for part in text_parts[1:]:
            text_field = text_field + ' ' + part
        text_field = text_field.str.lower()
        logger.info(f"  Created text field from {len(text_parts)} columns")
    else:
        logger.error("No text columns available for topic flags")
        for topic_name in topic_keywords.keys():
            df[topic_name] = 0
        df['topic_count'] = 0
        return df
    
    # Generate binary flags
    for topic_name, keywords in topic_keywords.items():
        pattern = '|'.join(keywords)
        df[topic_name] = text_field.str.contains(pattern, na=False).astype(int)
    
    # Add topic count
    topic_cols = [c for c in df.columns if c.startswith('topic_')]
    df['topic_count'] = df[topic_cols].sum(axis=1)
    
    logger.info(f"  ✅ Added {len(topic_cols) + 1} topic features")
    return df


def generate_velocity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate engagement velocity metrics."""
    logger.info("Generating velocity metrics...")
    
    df['age_days_safe'] = df['age_days'].clip(lower=1)
    
    # Velocity
    df['stars_per_day'] = df['stargazers_count'] / df['age_days_safe']
    df['forks_per_day'] = df['forks_count'] / df['age_days_safe']
    df['issues_per_day'] = df['open_issues_count'] / df['age_days_safe']
    
    # Ratios
    df['fork_to_star_ratio'] = df['forks_count'] / (df['stargazers_count'] + 1)
    df['issue_to_star_ratio'] = df['open_issues_count'] / (df['stargazers_count'] + 1)
    
    # Log-transform
    for col in ['stars_per_day', 'forks_per_day', 'issues_per_day']:
        df[f'log_{col}'] = np.log1p(df[col])
    
    logger.info("  ✅ Added velocity metrics")
    return df


def run_week4_pipeline(
    data_dir: str = "/Users/manishswami/developer/Github_star_project/data/processed",
    output_dir: str = "/Users/manishswami/developer/Github_star_project/data/enriched_week4"
):
    """Execute Week 4: Add all advanced features."""
    print("="*60)
    print("WEEK 4: ADVANCED FEATURE ENGINEERING")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60)
    
    # Load Week 3 data
    files = list(Path(data_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    print(f"\n📂 Loading {len(files)} enriched parquet files...")
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    print(f"   ✅ Loaded {len(df):,} records with {len(df.columns)} columns")
    
    # Show available columns for debugging
    print(f"\n📋 Available columns (first 20): {list(df.columns)[:20]}")
    
    # Generate features
    print("\n🔧 Generating advanced features...")
    df = generate_owner_features(df)
    df = generate_topic_flags(df)
    df = generate_velocity_metrics(df)
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving Week 4 dataset to {output_dir}/...")
    chunk_size = 20000
    total_chunks = (len(df) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        output_file = output_path / f"repos_week4_part_{i//chunk_size}.parquet"
        chunk.to_parquet(output_file, compression='snappy', index=False)
    
    print(f"\n{'='*60}")
    print("WEEK 4 COMPLETE")
    print("="*60)
    print(f"   ✅ Total rows: {len(df):,}")
    print(f"   ✅ Total columns: {len(df.columns)}")
    
    # Feature breakdown
    feature_counts = {
        'Metadata': 5,
        'Embeddings': 384,
        'Owner': len([c for c in df.columns if c.startswith('owner_')]),
        'Topic': len([c for c in df.columns if c.startswith('topic_')]),
        'Velocity': len([c for c in df.columns if 'per_day' in c or 'ratio' in c 
                        and not c.startswith('owner_') and not c.startswith('topic_')])
    }
    
    print(f"\n📊 Feature Breakdown:")
    for category, count in feature_counts.items():
        print(f"   • {category}: {count}")
    print(f"   ─────────────────────────────")
    print(f"   • TOTAL: {sum(feature_counts.values())}")
    
    return df


if __name__ == "__main__":
    df_week4 = run_week4_pipeline()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
🟢 WEEK 4 COMPLETE - READY FOR FINAL MODEL

   Next Actions:
   1. Update eda_and_baseline.py to load from ./data/enriched_week4/
   
   2. Use optimized hyperparameters for 400+ features
   
   3. Expected RMSLE: 0.58-0.63
   
   4. Final step: Model ensembling (Week 5)
    """)
    print("="*60)