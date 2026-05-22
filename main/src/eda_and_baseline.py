import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_log_error
import pyarrow.parquet as pq
from pathlib import Path
import warnings
from datetime import datetime
from datetime import timezone

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

class RepositoryAnalyzer:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.df = None

    def load_data(self, sample_for_eda: float = 0.2):
        """Load parquet files; preprocess FIRST, then sample for EDA."""
        files = list(self.data_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError("No parquet files found.")
        
        # Load full data for modeling
        dfs = [pq.read_table(f).to_pandas() for f in files]
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(self.df):,} records from {len(files)} files.")
        
        # PREPROCESS FIRST - creates log_stars, age_days, etc.
        self._preprocess()
        
        # THEN create sampled dataframe for EDA visualizations
        self.df_eda = self.df.sample(frac=sample_for_eda, random_state=42) if sample_for_eda < 1.0 else self.df
        print(f"Using {len(self.df_eda):,} samples for EDA plots.")

    def _preprocess(self):
        """Feature engineering and cleaning with timezone handling."""
        # Convert dates with explicit timezone handling
        self.df['created_at'] = pd.to_datetime(self.df['created_at'], utc=True)
        self.df['ingested_at'] = pd.to_datetime(self.df['ingested_at'], utc=True)
        
        # If ingested_at is still naive after conversion, localize it to UTC
        if self.df['ingested_at'].dt.tz is None:
            self.df['ingested_at'] = self.df['ingested_at'].dt.tz_localize('UTC')
        
        # Calculate Age at ingestion (Critical Feature)
        # Both columns are now tz-aware, subtraction works correctly
        self.df['age_days'] = (self.df['ingested_at'] - self.df['created_at']).dt.days
        
        # Target: Log-transform stars to handle power law
        self.df['log_stars'] = np.log1p(self.df['stargazers_count'])
        
        # Impute missing language as "Unknown"
        if 'language' in self.df.columns:
            self.df['language'] = self.df['language'].fillna('Unknown')
        
        # Filter invalid ages (negative or extreme outliers)
        self.df = self.df[(self.df['age_days'] >= 0) & (self.df['age_days'] <= 3650)]
        
        # Verify preprocessing
        assert self.df['age_days'].isna().sum() == 0, "Nulls in age_days after preprocessing"
        assert self.df['log_stars'].isna().sum() == 0, "Nulls in log_stars after preprocessing"

    def explore_target(self):
        """Use sampled data for plotting."""
        df_plot = self.df_eda  # Use sampled data
        
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(df_plot['stargazers_count'], bins=100, ax=ax[0], kde=True)
        ax[0].set_title('Raw Star Count Distribution')
        ax[0].set_xlim(0, df_plot['stargazers_count'].quantile(0.99))  # Clip outliers for visibility
        
        sns.histplot(df_plot['log_stars'], bins=100, ax=ax[1], kde=True, color='green')
        ax[1].set_title('Log-Transformed Target')
        
        plt.tight_layout()
        plt.savefig('eda_target_distribution.png', dpi=300)
        plt.show()

    def explore_features(self):
        """Correlation heatmap."""
        numeric_cols = ['age_days', 'forks_count', 'watchers_count', 'open_issues_count', 'size', 'log_stars']
        corr_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Feature Correlation Matrix')
        plt.savefig('eda_correlation.png')
        plt.show()
        
        # Print correlation with target
        print("\nCorrelation with Target (log_stars):")
        print(corr_matrix['log_stars'].sort_values(ascending=False))

    def run_baseline_model(self, use_embeddings: bool = True):
        """
        Train LightGBM baseline with optional embedding features.
        
        Args:
            use_embeddings: If True, include 384 readme_emb_* features
        """
        print("\n" + "="*60)
        print("BASELINE MODEL" + (" + EMBEDDINGS" if use_embeddings else " (METADATA ONLY)"))
        print("="*60)
        
        # Metadata features (always included)
        metadata_features = [
            'age_days', 
            'forks_count', 
            'open_issues_count', 
            'size', 
            'language'
        ]
        
        # Embedding features (optional)
        if use_embeddings:
            embedding_features = [f"readme_emb_{i}" for i in range(384)]
            feature_cols = metadata_features + embedding_features
            print(f"\n📋 Feature Set: {len(metadata_features)} metadata + {len(embedding_features)} embeddings = {len(feature_cols)} total")
        else:
            feature_cols = metadata_features
            print(f"\n📋 Feature Set: {len(metadata_features)} metadata features only")
        
        # Load enriched data if using embeddings
        data_dir = "/Users/manishswami/developer/Github_star_project/data/processed" if use_embeddings else "/Users/manishswami/developer/Github_star_project/data/source"
        files = list(Path(data_dir).glob("*.parquet"))
        
        if not files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")
        
        print(f"\n📂 Loading data from {data_dir}/...")
        self.df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
        print(f"   ✅ Loaded {len(self.df):,} records")
        
        # Preprocess (same as before)
        self._preprocess()
        
        # Sort by ingestion time for temporal split
        self.df = self.df.sort_values(by='ingested_at').reset_index(drop=True)
        
        X = self.df[feature_cols].copy()
        y = self.df['log_stars'].copy()
        
        # Encode categorical features
        from sklearn.preprocessing import LabelEncoder
        categorical_cols = ['language']
        
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].fillna('Unknown').astype(str))
        
        # Handle missing values
        X = X.fillna(-1)
        
        # === Model Configuration ===
        # Adjust hyperparameters for high-dimensional data
        if use_embeddings:
            model_params = dict(
                objective='regression',
                metric='rmse',
                n_estimators=3000,           # More trees for more features
                learning_rate=0.02,          # Lower LR for stability
                max_depth=-1,
                num_leaves=127,              # More leaves for complex patterns
                min_child_samples=50,        # Higher to prevent overfitting
                subsample=0.7,               # More regularization
                colsample_bytree=0.5,        # Critical: sample 50% of 389 features
                reg_alpha=0.2,               # Stronger L1 regularization
                reg_lambda=0.2,              # Stronger L2 regularization
                verbose=-1,
                random_state=42
            )
            print("\n⚙️  Model config: Optimized for high-dimensional embeddings")
        else:
            model_params = dict(
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
            print("\n⚙️  Model config: Standard metadata baseline")
        
        # === TimeSeriesSplit Cross-Validation ===
        tscv = TimeSeriesSplit(n_splits=3)
        rmsle_scores = []
        best_iterations = []
        
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
                callbacks=[lgb.early_stopping(stopping_rounds=50)]
            )
            
            preds = model.predict(X_val)
            score = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(preds)))
            rmsle_scores.append(score)
            best_iterations.append(model.best_iteration_)
            
            print(f"\nFold {fold+1}:")
            print(f"   RMSLE:     {score:.4f}")
            print(f"   Best iter: {model.best_iteration_}")
        
        # === Summary ===
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        mean_score = np.mean(rmsle_scores)
        std_score = np.std(rmsle_scores)
        print(f"Mean CV RMSLE: {mean_score:.4f} (+/- {std_score:.4f})")
        
        # Compare to metadata-only baseline
        if use_embeddings:
            baseline_rmsle = 0.6921
            improvement = (baseline_rmsle - mean_score) / baseline_rmsle * 100
            print(f"\n📈 Improvement vs Metadata-Only Baseline:")
            print(f"   Before: {baseline_rmsle:.4f}")
            print(f"   After:  {mean_score:.4f}")
            print(f"   Change: {improvement:+.2f}%")
            
            if improvement < 5:
                print("   ⚠️  Modest gain: Embeddings may need tuning")
            elif improvement < 15:
                print("   ✅ Good gain: Semantic signal captured")
            else:
                print("   🎉 Excellent gain: Embeddings highly predictive")
        
        # === Feature Importance (Top 20) ===
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE (Top 20)")
        print("="*60)
        
        final_model = lgb.LGBMRegressor(**{**model_params, 'n_estimators': int(np.mean(best_iterations))})
        final_model.fit(X, y)
        
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        importance_df['pct'] = (importance_df['importance'] / importance_df['importance'].sum() * 100).round(2)
        
        print("\nTop Features:")
        for _, row in importance_df.iterrows():
            feat_name = row['feature']
            if feat_name.startswith('readme_emb_'):
                feat_name = f"{feat_name[:12]}..."  # Truncate embedding names
            print(f"   {feat_name:25s}: {row['importance']:8.0f} ({row['pct']:5.2f}%)")
        
        # Check if embeddings appear in top features
        emb_in_top = importance_df['feature'].str.startswith('readme_emb_').any()
        if emb_in_top:
            emb_count = importance_df['feature'].str.startswith('readme_emb_').sum()
            print(f"\n✅ {emb_count}/20 top features are embeddings → Semantic signal detected")
        else:
            print(f"\n⚠️  No embeddings in top 20 → Model relies on metadata")
        
        # Save importance
        importance_df.to_csv(f'feature_importance_{"with_embeddings" if use_embeddings else "metadata_only"}.csv', index=False)
        print(f"\n✅ Feature importance saved to CSV")
        
        return mean_score, std_score, importance_df

if __name__ == "__main__":
    print("="*60)
    print("GITHUB STAR PREDICTION - WEEK 3 EVALUATION")
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    print("="*60)
    
    analyzer = RepositoryAnalyzer("/Users/manishswami/developer/Github_star_project/data/processed")  # Use enriched data
    
    # Run metadata-only baseline (for comparison)
    print("\n" + "="*60)
    print("STEP 1: METADATA-ONLY BASELINE (RE-RUN)")
    print("="*60)
    meta_rmsle, meta_std, _ = analyzer.run_baseline_model(use_embeddings=False)
    
    # Run with embeddings
    print("\n" + "="*60)
    print("STEP 2: BASELINE + EMBEDDINGS")
    print("="*60)
    emb_rmsle, emb_std, emb_importance = analyzer.run_baseline_model(use_embeddings=True)
    
    # Final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    improvement = (meta_rmsle - emb_rmsle) / meta_rmsle * 100