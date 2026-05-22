# temporal_holdout_validation.py
"""
Temporal Holdout Validation: Train on older data, test on newer data
This simulates production deployment and detects subtle leakage.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import pyarrow.parquet as pq
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def temporal_holdout_validation(data_dir: str, split_date: str = "2024-06-01"):
    """
    Train on repos created before split_date, test on repos created after.
    
    Args:
        data_dir: Path to parquet files
        split_date: Date string (YYYY-MM-DD) for train/test split
        
    Returns:
        bool: True if model generalizes well (gap < 20%)
    """
    print("="*60)
    print("TEMPORAL HOLDOUT VALIDATION")
    print(f"Split Date: {split_date}")
    print("="*60)
    
    # Load all data
    files = list(Path(data_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    print(f"\n📂 Loading {len(files)} parquet files...")
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    print(f"   Loaded {len(df):,} repositories")
    
    # === Preprocessing ===
    print("\n🔧 Preprocessing...")
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
    df['ingested_at'] = pd.to_datetime(df['ingested_at'], utc=True)
    if df['ingested_at'].dt.tz is None:
        df['ingested_at'] = df['ingested_at'].dt.tz_localize('UTC')
    df['age_days'] = (df['ingested_at'] - df['created_at']).dt.days
    df['log_stars'] = np.log1p(df['stargazers_count'])
    df['language'] = df['language'].fillna('Unknown').astype(str)
    
    # === Temporal Split (by creation date, NOT ingestion) ===
    train_df = df[df['created_at'] < split_date].copy()
    test_df = df[df['created_at'] >= split_date].copy()
    
    print(f"\n📊 Dataset Split:")
    print(f"   Train: {len(train_df):,} repos (created before {split_date})")
    print(f"   Test:  {len(test_df):,} repos (created on/after {split_date})")
    
    if len(test_df) < 1000:
        print("\n⚠️  Warning: Test set is small. Results may be noisy.")
    
    # === Feature Engineering ===
    feature_cols = ['age_days', 'forks_count', 'open_issues_count', 'size', 'language']
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df['log_stars'].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df['log_stars'].copy()
    
    # === CRITICAL FIX: Label Encode 'language' ===
    print("\n🏷️  Encoding categorical features...")
    le = LabelEncoder()
    
    # Fit on TRAIN only (prevents leakage from test set)
    X_train['language'] = le.fit_transform(X_train['language'])
    # Transform test using train's encoder (handles unseen categories)
    X_test['language'] = X_test['language'].apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
    
    # Handle missing values
    X_train = X_train.fillna(-1)
    X_test = X_test.fillna(-1)
    
    print(f"   Encoded 'language': {len(le.classes_)} unique categories")
    print(f"   Categories: {le.classes_[:10]}{'...' if len(le.classes_) > 10 else ''}")
    
    # Verify dtypes
    print(f"\n🔍 Feature dtype verification:")
    print(f"   X_train dtypes:\n{X_train.dtypes}")
    
    # === Train Model ===
    print("\n🚀 Training LightGBM model...")
    model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        verbose=-1,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    # === Evaluate ===
    preds = model.predict(X_test)
    rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_test), np.expm1(preds)))
    
    # === Compare to CV Score ===
    cv_rmsle = 0.6921  # Your TimeSeriesSplit CV result
    gap = (rmsle - cv_rmsle) / cv_rmsle * 100
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"   CV RMSLE:        {cv_rmsle:.4f}")
    print(f"   Holdout RMSLE:   {rmsle:.4f}")
    print(f"   Best Iteration:  {model.best_iteration_}")
    print(f"\n📈 Generalization Gap: {gap:+.2f}%")
    
    # === Assessment ===
    print("\n" + "="*60)
    print("ASSESSMENT")
    print("="*60)
    
    if gap > 20:
        print("\n⚠️  WARNING: Large generalization gap detected!")
        print("   Possible causes:")
        print("   - Subtle leakage in training data")
        print("   - Significant concept drift (2023 vs 2024+ repos)")
        print("   - Overfitting to training distribution")
        print("\n   RECOMMENDATION: Investigate before proceeding to Week 3.")
        is_valid = False
    elif gap > 10:
        print("\n🟡 MODERATE GAP: Acceptable but monitor drift.")
        print("   Some concept drift is expected in temporal data.")
        print("   RECOMMENDATION: Proceed to Week 3 with caution.")
        is_valid = True
    else:
        print("\n✅ EXCELLENT: Model generalizes well to unseen temporal data.")
        print("   No significant leakage or drift detected.")
        print("   RECOMMENDATION: Proceed to Week 3 feature engineering.")
        is_valid = True
    
    # === Feature Importance ===
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Holdout)")
    print("="*60)
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    importance_df['pct'] = (importance_df['importance'] / importance_df['importance'].sum() * 100).round(2)
    
    for _, row in importance_df.iterrows():
        print(f"   {row['feature']:20s}: {row['importance']:8.0f} ({row['pct']:5.2f}%)")
    
    print("\n" + "="*60)
    
    return is_valid, rmsle, gap


if __name__ == "__main__":
    # Configuration
    DATA_DIR = "/Users/manishswami/developer/Github_star_project/data/source"  # Update to match your actual path
    SPLIT_DATE = "2024-06-01"
    
    print("="*60)
    print("GITHUB STAR PREDICTION - TEMPORAL HOLDOUT VALIDATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60)
    
    try:
        is_valid, rmsle, gap = temporal_holdout_validation(DATA_DIR, SPLIT_DATE)
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        if is_valid:
            print("""
🟢 VALIDATION PASSED

   ✅ Generalization gap is acceptable (< 20%)
   ✅ No significant leakage detected
   ✅ Safe to proceed to Week 3

   Next Actions:
   1. Install embedding dependencies:
      pip install sentence-transformers torch

   2. Generate README embeddings:
      python week3_text_embeddings.py

   3. Retrain baseline with embeddings:
      Update feature_cols in eda_and_baseline.py

   4. Expected improvement:
      RMSLE: 0.69 → 0.55-0.62 (with embeddings)
            """)
        else:
            print("""
🔴 VALIDATION FAILED

   ⚠️  Generalization gap exceeds 20%
   ⚠️  Do NOT proceed to Week 3 yet

   Troubleshooting Steps:
   1. Re-run leakage audit:
      python debug_leakage.py

   2. Check for concept drift:
      - Compare feature distributions (train vs test)
      - Analyze residuals by time period

   3. Consider temporal weighting:
      - Weight recent samples more heavily
      - Use rolling window validation
            """)
        
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nDebugging Tips:")
        print("   1. Verify data path is correct")
        print("   2. Check parquet files are not corrupted")
        print("   3. Ensure all required columns exist")
        raise