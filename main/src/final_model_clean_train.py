# week5_final_model_clean.py
"""
Week 5: Final Model Training (LEAKAGE-FREE)
============================================
Train LightGBM with clean features (no target leakage).
Target RMSLE: 0.55-0.65 (realistic for this task)
"""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_log_error
from datetime import datetime
import logging
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("week5_final_model_clean.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def build_feature_cols_clean():
    """Build COMPLETE feature list EXCLUDING leaked features."""
    # Metadata (5)
    metadata = ['age_days', 'forks_count', 'open_issues_count', 'size', 'language']
    
    # Embeddings (384)
    embeddings = [f"readme_emb_{i}" for i in range(384)]
    
    # Owner Reputation (12)
    owner = [
        'owner_mean_stars', 'owner_total_stars', 'owner_repo_count',
        'owner_std_stars', 'owner_mean_forks', 'owner_total_forks',
        'owner_tenure_days', 'owner_active_days', 'owner_success_rate',
        'owner_is_org', 'log_owner_total_stars', 'log_owner_repo_count'
    ]
    
    # Topic Flags (11)
    topics = [
        'topic_ai_llm', 'topic_ai_transformer', 'topic_ai_rag', 'topic_ai_agent',
        'topic_devops_k8s', 'topic_devops_docker', 'topic_web_react',
        'topic_data_pytorch', 'topic_data_tensorflow', 'topic_data_pandas',
        'topic_count'
    ]
    
    # === VELOCITY FEATURES (CLEAN - NO TARGET LEAKAGE) ===
    # Only use features that DON'T contain stargazers_count
    velocity_clean = [
        'forks_per_day',           # forks_count / age (forks is NOT target)
        'issues_per_day',          # open_issues_count / age (issues is NOT target)
        'log_forks_per_day',       # log(forks_per_day + 1)
        'log_issues_per_day',      # log(issues_per_day + 1)
        # REMOVED: stars_per_day, log_stars_per_day (LEAKAGE)
        # REMOVED: fork_to_star_ratio, issue_to_star_ratio (LEAKAGE)
    ]
    
    feature_cols = metadata + embeddings + owner + topics + velocity_clean
    
    return feature_cols


def train_final_model_clean(
    data_dir: str = "/Users/manishswami/developer/Github_star_project/data/enriched_week4",
    n_splits: int = 3
):
    """Train final LightGBM model with LEAKAGE-FREE features."""
    print("="*60)
    print("WEEK 5: FINAL MODEL TRAINING (LEAKAGE-FREE)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60)
    
    # Load data
    files = list(Path(data_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    print(f"\n📂 Loading {len(files)} Week 4 parquet files...")
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    print(f"   ✅ Loaded {len(df):,} records with {len(df.columns)} columns")
    
    # Preprocessing
    print("\n🔧 Preprocessing...")
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
    df['ingested_at'] = pd.to_datetime(df['ingested_at'], utc=True)
    if df['ingested_at'].dt.tz is None:
        df['ingested_at'] = df['ingested_at'].dt.tz_localize('UTC')
    
    if 'age_days' not in df.columns:
        df['age_days'] = (df['ingested_at'] - df['created_at']).dt.days
    
    if 'log_stars' not in df.columns:
        df['log_stars'] = np.log1p(df['stargazers_count'])
    
    # Encode language
    if 'language' in df.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['language'] = le.fit_transform(df['language'].fillna('Unknown').astype(str))
    
    # Sort by time
    print("\n📅 Sorting by ingested_at for temporal CV...")
    df = df.sort_values(by='ingested_at').reset_index(drop=True)
    
    # Build CLEAN feature columns (no leakage)
    feature_cols = build_feature_cols_clean()
    
    # Filter to existing columns
    existing_cols = [c for c in feature_cols if c in df.columns]
    print(f"\n📋 Feature Count: {len(existing_cols)} (of {len(feature_cols)} defined)")
    
    # === LEAKAGE AUDIT ===
    print("\n🔍 LEAKAGE AUDIT:")
    leaked_features = ['stars_per_day', 'log_stars_per_day', 'fork_to_star_ratio', 
                       'issue_to_star_ratio', 'stargazers_count']
    found_leaked = [c for c in existing_cols if c in leaked_features]
    
    if found_leaked:
        print(f"   ⚠️  WARNING: Found {len(found_leaked)} leaked features:")
        for f in found_leaked:
            print(f"      - {f}")
        print(f"   🗑️  Removing leaked features...")
        existing_cols = [c for c in existing_cols if c not in leaked_features]
        print(f"   ✅ Clean feature count: {len(existing_cols)}")
    else:
        print(f"   ✅ No leaked features detected")
    
    # Remove columns with >50% nulls
    null_pct = df[existing_cols].isnull().sum() / len(df) * 100
    cols_to_drop = null_pct[null_pct > 50].index.tolist()
    if cols_to_drop:
        print(f"⚠️  Dropping {len(cols_to_drop)} columns with >50% nulls")
        existing_cols = [c for c in existing_cols if c not in cols_to_drop]
    
    X = df[existing_cols].copy()
    y = df['log_stars'].copy()
    X = X.fillna(-1)
    
    # Model Configuration
    model_params = dict(
        objective='regression',
        metric='rmse',
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=-1,
        num_leaves=255,
        min_child_samples=100,
        subsample=0.6,
        colsample_bytree=0.4,
        colsample_bynode=0.8,
        reg_alpha=0.3,
        reg_lambda=0.3,
        verbose=-1,
        random_state=42,
        n_jobs=-1
    )
    
    print("\n⚙️  Model Configuration:")
    print(f"   • n_estimators: {model_params['n_estimators']}")
    print(f"   • learning_rate: {model_params['learning_rate']}")
    print(f"   • num_leaves: {model_params['num_leaves']}")
    print(f"   • colsample_bytree: {model_params['colsample_bytree']}")
    
    # TimeSeriesSplit CV
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmsle_scores = []
    best_iterations = []
    feature_importance_list = []
    
    print("\n" + "-"*60)
    print("CROSS-VALIDATION RESULTS")
    print("-"*60)
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**model_params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=100)]
        )
        
        preds = model.predict(X_val)
        score = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(preds)))
        rmsle_scores.append(score)
        best_iterations.append(model.best_iteration_)
        feature_importance_list.append(model.feature_importances_)
        
        print(f"\nFold {fold+1}:")
        print(f"   RMSLE:     {score:.4f}")
        print(f"   Best iter: {model.best_iteration_}")
    
    # Summary
    print("\n" + "="*60)
    print("FINAL PERFORMANCE SUMMARY (LEAKAGE-FREE)")
    print("="*60)
    mean_score = np.mean(rmsle_scores)
    std_score = np.std(rmsle_scores)
    
    print(f"\n📊 Mean CV RMSLE: {mean_score:.4f} (+/- {std_score:.4f})")
    
    print(f"\n📈 Improvement Trajectory:")
    print(f"   Week 2 (Metadata):         0.6921")
    print(f"   Week 3 (+ Embeddings):     0.6639  (-4.1%)")
    print(f"   Week 5 (WITH LEAKAGE):     0.1491  (ARTIFICIAL -78.5%)")
    print(f"   Week 5 (CLEAN):            {mean_score:.4f}  ({(0.6921-mean_score)/0.6921*100:+.1f}%)")
    
    # Realistic target assessment
    if mean_score < 0.58:
        print(f"\n🎉 EXCELLENT: Below target range (0.58-0.65)")
    elif mean_score < 0.65:
        print(f"\n✅ ON TARGET: Within expected range (0.58-0.65)")
    else:
        print(f"\n⚠️  ABOVE TARGET: But this is REALISTIC performance")
    
    # Feature Importance
    print("\n" + "="*60)
    print("TOP 30 FEATURE IMPORTANCE (CLEAN)")
    print("="*60)
    
    avg_importance = np.mean(feature_importance_list, axis=0)
    importance_df = pd.DataFrame({
        'feature': existing_cols,
        'importance': avg_importance
    }).sort_values('importance', ascending=False).head(30)
    
    importance_df['pct'] = (importance_df['importance'] / importance_df['importance'].sum() * 100).round(2)
    
    print("\nTop Features:")
    for i, (_, row) in enumerate(importance_df.iterrows(), 1):
        feat_name = row['feature']
        if feat_name.startswith('readme_emb_'):
            feat_name = f"{feat_name[:12]}..."
        print(f"   {i:2d}. {feat_name:25s}: {row['importance']:8.0f} ({row['pct']:5.2f}%)")
    
    # Feature Family Breakdown
    print("\n" + "="*60)
    print("FEATURE FAMILY IMPORTANCE (CLEAN)")
    print("="*60)
    
    families = {
        'Metadata': ['age_days', 'forks_count', 'open_issues_count', 'size', 'language'],
        'Embeddings': [c for c in existing_cols if c.startswith('readme_emb_')],
        'Owner': [c for c in existing_cols if c.startswith('owner_')],
        'Topic': [c for c in existing_cols if c.startswith('topic_')],
        'Velocity (Clean)': [c for c in existing_cols if 'per_day' in c and 'star' not in c]
    }
    
    family_importance = {}
    for family, cols in families.items():
        family_cols = [c for c in cols if c in existing_cols]
        family_mask = importance_df['feature'].isin(family_cols)
        family_importance[family] = importance_df[family_mask]['importance'].sum()
    
    total_imp = sum(family_importance.values())
    for family, imp in sorted(family_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"   {family:15s}: {imp:8.0f} ({imp/total_imp*100:5.1f}%)")
    
    # Save Results
    results_dir = Path("./results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    importance_df.to_csv(results_dir / "feature_importance_final_clean.csv", index=False)
    print(f"\n✅ Feature importance saved to: {results_dir / 'feature_importance_final_clean.csv'}")
    
    final_model = lgb.LGBMRegressor(**{**model_params, 'n_estimators': int(np.mean(best_iterations))})
    final_model.fit(X, y)
    
    import pickle
    with open(results_dir / "final_model_clean.pkl", 'wb') as f:
        pickle.dump(final_model, f)
    print(f"✅ Final model saved to: {results_dir / 'final_model_clean.pkl'}")
    
    plt.figure(figsize=(12, 10))
    top_20 = importance_df.head(20)
    plt.barh(range(len(top_20)), top_20['importance'].values)
    plt.yticks(range(len(top_20)), top_20['feature'].values)
    plt.gca().invert_yaxis()
    plt.xlabel('Gain Importance', fontsize=12)
    plt.title('Top 20 Feature Importance - Final Model (CLEAN)', fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "feature_importance_final_clean.png", dpi=300)
    plt.close()
    print(f"✅ Feature importance plot saved to: {results_dir / 'feature_importance_final_clean.png'}")
    
    return mean_score, std_score, importance_df


if __name__ == "__main__":
    try:
        rmsle, std, importance = train_final_model_clean()
        
        print("\n" + "="*60)
        print("PROJECT COMPLETE (LEAKAGE-FREE)")
        print("="*60)
        print(f"""
🎉 GITHUB STAR PREDICTION MODEL - FINAL RESULTS (CLEAN)

   Final RMSLE: {rmsle:.4f} (+/- {std:.4f})
   
   Improvement vs Baseline: {(0.6921-rmsle)/0.6921*100:+.1f}%
   
   This is REALISTIC performance without target leakage.
   
   Next Steps:
   1. Document methodology with leakage audit
   2. Compare leaked vs clean models
   3. Consider ensembling for additional 2-4% improvement
   4. Deploy to production or submit to Kaggle
        """)
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.exception("Final model training failed")
        raise