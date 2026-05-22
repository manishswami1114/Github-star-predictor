# debug_leakage.py
"""
Leakage Diagnostic Script for GitHub Star Prediction Project
============================================================
Purpose: Identify data leakage features that artificially inflate model performance
Author: Manish (Senior Data Scientist & ML Researcher)
Date: 2025

This script performs:
1. Data quality assessment (nulls, unique values, distributions)
2. Feature-target correlation analysis (leakage detection)
3. Temporal integrity checks (future information in training)
4. Visualizations for leakage patterns
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set(style="whitegrid")
plt.rcParams['figure.dpi'] = 300

class LeakageAuditor:
    """Comprehensive leakage detection for ML pipelines."""
    
    def __init__(self, data_dir: str, sample_files: int = 10):
        self.data_dir = Path(data_dir)
        self.sample_files = sample_files
        self.df = None
        self.audit_report = []
        
    def load_data(self):
        """Load sample of parquet files for diagnostic."""
        files = list(self.data_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {self.data_dir}")
        
        print(f"Found {len(files)} parquet files. Loading {self.sample_files} for diagnostics...")
        files_to_load = files[:self.sample_files]
        
        dfs = [pq.read_table(f).to_pandas() for f in files_to_load]
        self.df = pd.concat(dfs, ignore_index=True)
        
        print(f"Loaded {len(self.df):,} records from {len(files_to_load)} files.")
        return self
    
    def assess_data_quality(self):
        """Check data quality issues that may indicate problems."""
        print("\n" + "="*60)
        print("SECTION 1: DATA QUALITY ASSESSMENT")
        print("="*60)
        
        # Basic stats
        print(f"\n📊 Dataset Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"\n📋 Column Names:")
        for i, col in enumerate(self.df.columns, 1):
            print(f"   {i:2d}. {col}")
        
        # Null analysis
        print(f"\n🔍 Null Value Analysis:")
        null_summary = self.df.isnull().sum()
        null_pct = (null_summary / len(self.df) * 100).round(2)
        null_df = pd.DataFrame({
            'Null Count': null_summary,
            'Null %': null_pct
        }).sort_values('Null Count', ascending=False)
        
        # Show columns with >0% nulls
        cols_with_nulls = null_df[null_df['Null Count'] > 0]
        if len(cols_with_nulls) > 0:
            print(f"\n   Columns with null values:")
            for col, row in cols_with_nulls.iterrows():
                print(f"   - {col}: {row['Null Count']:,} ({row['Null %']}%)")
        else:
            print(f"   ✅ No null values detected")
        
        # Unique value analysis for categorical features
        print(f"\n🏷️  Categorical Feature Analysis:")
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            n_unique = self.df[col].nunique()
            top_values = self.df[col].value_counts().head(5)
            print(f"\n   {col}:")
            print(f"   - Unique values: {n_unique}")
            print(f"   - Top 5 values:")
            for val, count in top_values.items():
                pct = count / len(self.df) * 100
                print(f"     • {val}: {count:,} ({pct:.2f}%)")
        
        self.audit_report.append({
            'section': 'data_quality',
            'total_rows': len(self.df),
            'total_cols': len(self.df.columns),
            'categorical_cols': len(categorical_cols),
            'null_columns': len(cols_with_nulls)
        })
        
        return self
    
    def detect_feature_leakage(self, target_col='stargazers_count'):
        """Detect features that leak target information."""
        print("\n" + "="*60)
        print("SECTION 2: FEATURE LEAKAGE DETECTION")
        print("="*60)
        
        # Numeric feature correlations
        print(f"\n📈 Correlation Analysis (Target: {target_col}):")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlations = []
        
        for col in numeric_cols:
            if col != target_col and self.df[col].notna().sum() > 0:
                corr = self.df[col].corr(self.df[target_col])
                correlations.append({
                    'feature': col,
                    'correlation': corr,
                    'abs_correlation': abs(corr)
                })
        
        corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
        
        # Flag high-risk features
        print(f"\n   ⚠️  LEAKAGE RISK ASSESSMENT:")
        for _, row in corr_df.iterrows():
            risk_level = "🔴 CRITICAL" if row['abs_correlation'] > 0.9 else \
                         "🟠 HIGH" if row['abs_correlation'] > 0.7 else \
                         "🟡 MODERATE" if row['abs_correlation'] > 0.5 else \
                         "🟢 LOW"
            print(f"   {risk_level}: {row['feature']:25s} (r = {row['correlation']:+.4f})")
        
        # Specific check: watchers_count vs stargazers_count
        print(f"\n🔍 SPECIFIC LEAKAGE CHECK: watchers_count")
        if 'watchers_count' in self.df.columns and 'stargazers_count' in self.df.columns:
            watchers = self.df['watchers_count']
            stars = self.df['stargazers_count']
            
            corr = watchers.corr(stars)
            exact_match = (watchers == stars).sum() / len(self.df) * 100
            diff = (watchers - stars).abs()
            
            print(f"   - Correlation: {corr:.4f}")
            print(f"   - Exact match: {exact_match:.2f}% of rows")
            print(f"   - Max difference: {diff.max():.0f}")
            print(f"   - Mean difference: {diff.mean():.2f}")
            
            if corr > 0.99 or exact_match > 95:
                print(f"\n   ⚠️  ⚠️  ⚠️  CONFIRMED LEAKAGE: watchers_count mirrors stargazers_count!")
                print(f"   RECOMMENDATION: Remove 'watchers_count' from features immediately.")
        
        # Store audit results
        critical_features = corr_df[corr_df['abs_correlation'] > 0.9]['feature'].tolist()
        self.audit_report.append({
            'section': 'feature_leakage',
            'critical_features': critical_features,
            'high_risk_features': corr_df[corr_df['abs_correlation'] > 0.7]['feature'].tolist(),
            'watchers_leakage': corr > 0.99 if 'watchers_count' in self.df.columns else None
        })
        
        return self, corr_df
    
    def check_temporal_integrity(self):
        """Check for temporal leakage (future information in training)."""
        print("\n" + "="*60)
        print("SECTION 3: TEMPORAL INTEGRITY CHECK")
        print("="*60)
        
        date_cols = ['created_at', 'updated_at', 'pushed_at', 'ingested_at']
        available_date_cols = [c for c in date_cols if c in self.df.columns]
        
        if not available_date_cols:
            print(f"\n⚠️  No standard date columns found. Skipping temporal check.")
            return self
        
        # Convert to datetime
        for col in available_date_cols:
            self.df[col] = pd.to_datetime(self.df[col], utc=True, errors='coerce')
        
        print(f"\n📅 Date Column Ranges:")
        for col in available_date_cols:
            if self.df[col].notna().any():
                min_date = self.df[col].min()
                max_date = self.df[col].max()
                print(f"   {col}:")
                print(f"   - Min: {min_date}")
                print(f"   - Max: {max_date}")
        
        # Check for future dates relative to ingestion
        if 'ingested_at' in self.df.columns:
            ingestion_max = self.df['ingested_at'].max()
            print(f"\n🔍 Future Date Check (relative to ingestion_max = {ingestion_max}):")
            
            for col in available_date_cols:
                if col != 'ingested_at' and self.df[col].notna().any():
                    future_dates = (self.df[col] > ingestion_max).sum()
                    if future_dates > 0:
                        print(f"   ⚠️  {col}: {future_dates:,} rows ({future_dates/len(self.df)*100:.2f}%) have dates AFTER ingestion")
                    else:
                        print(f"   ✅ {col}: No future dates detected")
        
        self.audit_report.append({
            'section': 'temporal_integrity',
            'date_columns': available_date_cols,
            'ingestion_max': str(self.df['ingested_at'].max()) if 'ingested_at' in self.df.columns else None
        })
        
        return self
    
    def visualize_leakage_patterns(self, output_dir: str = './diagnostics'):
        """Create visualizations for leakage patterns."""
        print("\n" + "="*60)
        print("SECTION 4: LEAKAGE VISUALIZATIONS")
        print("="*60)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: watchers_count vs stargazers_count
        if 'watchers_count' in self.df.columns and 'stargazers_count' in self.df.columns:
            plt.figure(figsize=(8, 8))
            sample = self.df.sample(min(10000, len(self.df)), random_state=42)
            plt.scatter(sample['watchers_count'], sample['stargazers_count'], 
                       alpha=0.1, s=1, color='steelblue')
            max_val = max(sample['watchers_count'].max(), sample['stargazers_count'].max())
            plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='y=x (perfect correlation)')
            plt.xlabel('watchers_count', fontsize=12)
            plt.ylabel('stargazers_count', fontsize=12)
            plt.title('watchers_count vs stargazers_count\n(Red dashed line: perfect correlation)', fontsize=14)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / 'leakage_watchers_vs_stars.png', dpi=300)
            plt.close()
            print(f"   ✅ Saved: {output_path / 'leakage_watchers_vs_stars.png'}")
        
        # Plot 2: Feature correlation heatmap
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            corr_matrix = self.df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                       cmap='coolwarm', center=0, square=True, 
                       linewidths=0.5, cbar_kws={"shrink": 0.8})
            plt.title('Feature Correlation Heatmap\n(High correlations indicate potential leakage)', fontsize=14)
            plt.tight_layout()
            plt.savefig(output_path / 'feature_correlation_heatmap.png', dpi=300)
            plt.close()
            print(f"   ✅ Saved: {output_path / 'feature_correlation_heatmap.png'}")
        
        # Plot 3: Target distribution
        if 'stargazers_count' in self.df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Raw distribution
            sns.histplot(self.df['stargazers_count'], bins=100, ax=axes[0], kde=True, color='steelblue')
            axes[0].set_title('Raw Star Count Distribution\n(Highly skewed)', fontsize=12)
            axes[0].set_xlabel('Stars')
            axes[0].set_xlim(0, self.df['stargazers_count'].quantile(0.99))
            
            # Log distribution
            log_stars = np.log1p(self.df['stargazers_count'])
            sns.histplot(log_stars, bins=100, ax=axes[1], kde=True, color='green')
            axes[1].set_title('Log-Transformed Star Count\n(Closer to normal)', fontsize=12)
            axes[1].set_xlabel('Log(Stars + 1)')
            
            plt.tight_layout()
            plt.savefig(output_path / 'target_distribution.png', dpi=300)
            plt.close()
            print(f"   ✅ Saved: {output_path / 'target_distribution.png'}")
        
        return self
    
    def generate_safe_feature_list(self, exclude_patterns=None):
        """Generate a list of safe features to use in modeling."""
        print("\n" + "="*60)
        print("SECTION 5: RECOMMENDED SAFE FEATURES")
        print("="*60)
        
        if exclude_patterns is None:
            exclude_patterns = ['watchers', 'star', 'subscribers']
        
        all_features = list(self.df.columns)
        safe_features = []
        excluded_features = []
        
        for col in all_features:
            col_lower = col.lower()
            # Skip target and identifiers
            if col in ['stargazers_count', 'log_stars', 'id', 'ingested_at', 'created_at']:
                continue
            # Check exclusion patterns
            if any(pattern in col_lower for pattern in exclude_patterns):
                excluded_features.append(col)
            else:
                safe_features.append(col)
        
        print(f"\n🟢 SAFE FEATURES (recommended for modeling):")
        for i, col in enumerate(safe_features, 1):
            print(f"   {i:2d}. {col}")
        
        print(f"\n🔴 EXCLUDED FEATURES (potential leakage):")
        for col in excluded_features:
            print(f"   - {col}")
        
        print(f"\n📋 Copy-paste ready feature list for your model:")
        print(f"   feature_cols = {safe_features}")
        
        self.audit_report.append({
            'section': 'safe_features',
            'safe_features': safe_features,
            'excluded_features': excluded_features
        })
        
        return safe_features
    
    def save_audit_report(self, output_path: str = './diagnostics/audit_report.txt'):
        """Save complete audit report to file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("LEAKAGE AUDIT REPORT\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("="*60 + "\n\n")
            
            for section in self.audit_report:
                f.write(f"\n--- {section['section'].upper()} ---\n")
                for key, value in section.items():
                    if key != 'section':
                        f.write(f"{key}: {value}\n")
        
        print(f"\n✅ Audit report saved to: {output_path}")
        return self
    
    def run_full_audit(self):
        """Execute complete leakage audit pipeline."""
        print("\n" + "="*60)
        print("GITHUB STAR PREDICTION - LEAKAGE AUDIT")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("="*60)
        
        self.load_data()
        self.assess_data_quality()
        self.detect_feature_leakage()
        self.check_temporal_integrity()
        self.visualize_leakage_patterns()
        self.generate_safe_feature_list()
        self.save_audit_report()
        
        print("\n" + "="*60)
        print("AUDIT COMPLETE")
        print("="*60)
        print("\n📁 Output files generated in ./diagnostics/:")
        print("   - leakage_watchers_vs_stars.png")
        print("   - feature_correlation_heatmap.png")
        print("   - target_distribution.png")
        print("   - audit_report.txt")
        print("\n⚠️  IMPORTANT: Review CRITICAL and HIGH risk features before modeling!")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Configuration
    DATA_DIR = "/Users/manishswami/developer/Github_star_project/data/source"
    SAMPLE_FILES = 10  # Number of parquet files to sample for diagnostics
    OUTPUT_DIR = "./diagnostics"
    
    # Run audit
    auditor = LeakageAuditor(DATA_DIR, SAMPLE_FILES)
    auditor.run_full_audit()
    
    # Print summary for immediate action
    print("\n" + "="*60)
    print("IMMEDIATE ACTION ITEMS")
    print("="*60)
    print("""
1. Review ./diagnostics/audit_report.txt for full details
2. Check feature_correlation_heatmap.png for high correlations (>0.9)
3. Remove 'watchers_count' from your feature list if correlation > 0.99
4. Update eda_and_baseline.py with safe feature list
5. Re-run baseline model and expect RMSLE in 1.2-1.6 range
    """)
    print("="*60)