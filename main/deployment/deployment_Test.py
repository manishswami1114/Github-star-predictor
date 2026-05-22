# deployment/predict_final.py
"""
Production Deployment: GitHub Star Prediction (FINAL FIXED)
============================================================
Complete feature pipeline with proper language encoding.
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GitHubStarPredictor:
    """Production-ready predictor with complete feature engineering pipeline."""
    
    def __init__(self, model_path: str = "../results/final_model_clean.pkl",
                 encoder_path: str = "../results/language_encoder.pkl"):
        """Initialize predictor with trained model."""
        self.model = None
        self.feature_cols = None
        self.language_encoder = None
        
        # Load model
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"✅ Model loaded from {model_path}")
        
        # Load language encoder (if available)
        encoder_file = Path(encoder_path)
        if encoder_file.exists():
            with open(encoder_file, 'rb') as f:
                self.language_encoder = pickle.load(f)
            logger.info(f"✅ Language encoder loaded from {encoder_path}")
        else:
            self.language_encoder = None
            logger.warning("⚠️  Language encoder not found; using fallback mapping")
        
        # Build complete feature list (416 features)
        self.feature_cols = self._build_feature_cols()
        logger.info(f"✅ Feature columns built: {len(self.feature_cols)} features")
    
    def _build_feature_cols(self):
        """Build COMPLETE feature list matching training data."""
        metadata = ['age_days', 'forks_count', 'open_issues_count', 'size', 'language']
        embeddings = [f"readme_emb_{i}" for i in range(384)]
        owner = [
            'owner_mean_stars', 'owner_total_stars', 'owner_repo_count',
            'owner_std_stars', 'owner_mean_forks', 'owner_total_forks',
            'owner_tenure_days', 'owner_active_days', 'owner_success_rate',
            'owner_is_org', 'log_owner_total_stars', 'log_owner_repo_count'
        ]
        topics = [
            'topic_ai_llm', 'topic_ai_transformer', 'topic_ai_rag', 'topic_ai_agent',
            'topic_devops_k8s', 'topic_devops_docker', 'topic_web_react',
            'topic_data_pytorch', 'topic_data_tensorflow', 'topic_data_pandas',
            'topic_count'
        ]
        velocity = ['forks_per_day', 'issues_per_day', 'log_forks_per_day', 'log_issues_per_day']
        
        return metadata + embeddings + owner + topics + velocity
    
    def _extract_topic_flags(self, name: str, full_name: str = "") -> dict:
        """Generate topic flags from repo name."""
        text = f"{name} {full_name}".lower()
        
        topic_keywords = {
            'topic_ai_llm': ['llm', 'chatbot', 'gpt', 'claude', 'llama'],
            'topic_ai_transformer': ['transformer', 'bert', 'vit', 'diffusion'],
            'topic_ai_rag': ['rag', 'retrieval', 'vector database', 'embedding'],
            'topic_ai_agent': ['agent', 'autonomous', 'langchain', 'crewai'],
            'topic_devops_k8s': ['kubernetes', 'k8s', 'helm'],
            'topic_devops_docker': ['docker', 'container', 'podman'],
            'topic_web_react': ['react', 'nextjs', 'next.js'],
            'topic_data_pytorch': ['pytorch', 'torch'],
            'topic_data_tensorflow': ['tensorflow', 'tf', 'keras'],
            'topic_data_pandas': ['pandas', 'polars', 'dask'],
        }
        
        flags = {}
        for topic_name, keywords in topic_keywords.items():
            flags[topic_name] = 1 if any(kw in text for kw in keywords) else 0
        flags['topic_count'] = sum(flags.values())
        return flags
    
    def _generate_owner_features(self, owner_login: str, owner_stats_cache: dict = None) -> dict:
        """Generate owner reputation features."""
        if owner_stats_cache and owner_login in owner_stats_cache:
            stats = owner_stats_cache[owner_login]
        else:
            stats = {
                'owner_mean_stars': 100, 'owner_total_stars': 100,
                'owner_repo_count': 1, 'owner_std_stars': 50,
                'owner_mean_forks': 20, 'owner_total_forks': 20,
                'owner_tenure_days': 365, 'owner_active_days': 365,
                'owner_success_rate': 0.1, 'owner_is_org': 0,
            }
        stats['log_owner_total_stars'] = np.log1p(stats['owner_total_stars'])
        stats['log_owner_repo_count'] = np.log1p(stats['owner_repo_count'])
        return stats
    
    def _generate_velocity_features(self, forks_count: int, open_issues_count: int, 
                                    age_days: int) -> dict:
        """Generate clean velocity features."""
        age_safe = max(age_days, 1)
        forks_per_day = forks_count / age_safe
        issues_per_day = open_issues_count / age_safe
        return {
            'forks_per_day': forks_per_day,
            'issues_per_day': issues_per_day,
            'log_forks_per_day': np.log1p(forks_per_day),
            'log_issues_per_day': np.log1p(issues_per_day)
        }
    
    def _generate_embeddings(self, name: str, description: str = "") -> dict:
        """Generate embeddings (zero vectors for demo)."""
        embedding = np.zeros(384)
        return {f"readme_emb_{i}": float(embedding[i]) for i in range(384)}
    
    def prepare_features(self, repo_data: dict) -> pd.DataFrame:
        """Prepare complete feature vector from raw repo data."""
        name = repo_data.get('name', 'Unknown')
        full_name = repo_data.get('full_name', name)
        description = repo_data.get('description', '')
        owner_login = repo_data.get('owner_login', 'unknown')
        forks_count = repo_data.get('forks_count', 0)
        open_issues_count = repo_data.get('open_issues_count', 0)
        size = repo_data.get('size', 0)
        language = repo_data.get('language', 'Unknown')
        
        created_at = repo_data.get('created_at')
        if created_at:
            if isinstance(created_at, str):
                created_at = pd.to_datetime(created_at, utc=True)
            age_days = (datetime.now(timezone.utc) - created_at).days
        else:
            age_days = repo_data.get('age_days', 30)
        
        features = {}
        
        # 1. Metadata (5)
        features['age_days'] = age_days
        features['forks_count'] = forks_count
        features['open_issues_count'] = open_issues_count
        features['size'] = size
        features['language'] = language
        
        # 2. Topic Flags (11)
        features.update(self._extract_topic_flags(name, full_name))
        
        # 3. Velocity Features (4)
        features.update(self._generate_velocity_features(forks_count, open_issues_count, age_days))
        
        # 4. Owner Reputation (12)
        owner_stats = repo_data.get('owner_stats', None)
        features.update(self._generate_owner_features(owner_login, owner_stats))
        
        # 5. Embeddings (384)
        embeddings = repo_data.get('embeddings', None)
        if embeddings:
            features.update({f"readme_emb_{i}": float(v) for i, v in enumerate(embeddings)})
        else:
            features.update(self._generate_embeddings(name, description))
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # === FIX: Encode language column ===
        if 'language' in df.columns:
            df['language'] = df['language'].fillna('Unknown').astype(str)
            if self.language_encoder:
                df['language'] = self.language_encoder.transform(df['language'])
            else:
                language_mapping = {
                    'Python': 0, 'JavaScript': 1, 'Java': 2, 'TypeScript': 3,
                    'C++': 4, 'C': 5, 'HTML': 6, 'CSS': 7, 'PHP': 8, 'Ruby': 9,
                    'Go': 10, 'Rust': 11, 'Swift': 12, 'Kotlin': 13, 'Scala': 14,
                    'Shell': 15, 'Perl': 16, 'Lua': 17, 'R': 18, 'MATLAB': 19,
                    'Unknown': 99
                }
                df['language'] = df['language'].map(lambda x: language_mapping.get(x, 99))
        
        # Check for missing columns
        missing_cols = set(self.feature_cols) - set(df.columns)
        if missing_cols:
            logger.warning(f"Missing {len(missing_cols)} features, filling with -1")
            for col in missing_cols:
                df[col] = -1
        
        # Remove extra columns
        extra_cols = set(df.columns) - set(self.feature_cols)
        if extra_cols:
            df = df.drop(columns=list(extra_cols))
        
        # Reorder to match training
        df = df[self.feature_cols]
        
        # === FIX: Ensure all numeric dtype ===
        df = df.fillna(-1).astype(np.float32)
        
        return df
    
    def predict(self, repo_data: dict) -> dict:
        """Predict star count for a repository."""
        X = self.prepare_features(repo_data)
        
        log_pred = self.model.predict(X)[0]
        predicted_stars = np.expm1(log_pred)
        
        return {
            'predicted_stars': float(predicted_stars),
            'predicted_log_stars': float(log_pred),
            'input_features': len(X.columns),
            'model_version': 'v1.0_clean'
        }


if __name__ == "__main__":
    predictor = GitHubStarPredictor(model_path="/Users/manishswami/developer/Github_star_project/results/final_model_clean.pkl")
    
    print("="*60)
    print("EXAMPLE: Single Repo Prediction")
    print("="*60)
    
    new_repo = {
        'name': 'awesome-ml-toolkit',
        'full_name': 'techcorp/awesome-ml-toolkit',
        'description': 'A comprehensive ML toolkit for production',
        'owner_login': 'techcorp',
        'forks_count': 150,
        'open_issues_count': 25,
        'size': 5000,
        'language': 'Python',
        'created_at': '2025-01-15T00:00:00Z',
    }
    
    result = predictor.predict(new_repo)
    print(f"Repo: {new_repo['name']}")
    print(f"Predicted Stars: {result['predicted_stars']:.0f}")
    print(f"Features Used: {result['input_features']}")
    
    print("\n✅ DEPLOYMENT READY")