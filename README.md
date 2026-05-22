---
title: Github Star Predictor
emoji: ⭐
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# 🌟 GitHub Star Predictor

An interactive, production-ready machine learning application that predicts a GitHub repository's stargazers (popularity) using a leakage-free **LightGBM** model combined with **Sentence Embeddings** (`all-MiniLM-L6-v2`) and rich developer reputation features.

Deployable locally as a premium **Streamlit** dashboard, this project features dynamic on-the-fly feature engineering, a comprehensive leakage audit demonstration, and live API integration with GitHub.

---

## 🚀 Key Features

*   **🔍 Auto-Fetch via GitHub URL:** Paste any public GitHub URL (e.g. `https://github.com/django/django`). The app automatically fetches real-time metadata, queries the owner's profile to compute their global reputation score, downloads the description, and outputs a star prediction.
*   **✍️ Manual Sandbox Mode:** Design your own unpublished repository! Experiment with different README summaries, sizes, programming languages, and active issues to see popularity projections before push.
*   **🤖 On-the-fly NLP Embeddings:** Instantly generates a **384-dimensional sentence embedding** of the repository's description using `SentenceTransformer` to capture the semantic complexity of its documentation.
*   **📈 Structural Decomposition:** Visualizes feature category influence (Owner Reputation, Velocity, Metadata, Embeddings) using high-fidelity **Plotly** indicators.
*   **🔬 Rigorous Leakage Audit:** A educational dashboard that showcases the Week 5 data science victory—resolving the target-leakage bug and establishing a robust, leakage-free CV score.

---

## 🛠️ Installation & Quickstart

To run the Streamlit app locally, follow these simple steps:

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/manishswami1114/Github-star-predictor.git
cd Github-star-predictor

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📊 Machine Learning Pipeline & Architecture

The prediction engine is trained on **416 features** split into 5 core groups:

1.  **Repository Metadata (5):** Age (days since creation), open issues, fork count, size in KB, and primary programming language.
2.  **README NLP Embeddings (384):** Semantic representations computed via the pretrained transformer `all-MiniLM-L6-v2` to evaluate documentation quality and keywords.
3.  **Owner Reputation (12):** Historical success metrics including the owner's total stargazers across other repositories, standard deviation of stars, success rate (repos with >100 stars), and account tenure.
4.  **Topic Flags (11):** Binary flags for viral/trending keywords (e.g., `ai_llm`, `rag`, `k8s`, `react`, `pytorch`).
5.  **Velocity Metrics (4):** Safe, target-leakage free indicators tracking activity rate (e.g., `forks_per_day` and `issues_per_day`).

---

## 🔬 Leakage Audit & Validation Results

During Week 5 model training, our team discovered that inclusion of direct star-related speed features (e.g., `stars_per_day`) caused a **Target Leakage Trap**, resulting in an artificially inflated validation score of **0.1491 RMSLE**. 

By auditing the data leakage and training a clean, leakage-free model:
*   We removed all leaked features.
*   We engineered clean, proxy velocity metrics.
*   We achieved a robust, production-generalizable **Mean CV RMSLE of 0.5824**, proving the model's reliability in predicting yet-to-be-published or emerging repositories.

### Model Iteration History
*   **Metadata Baseline:** `0.6921 RMSLE`
*   **Metadata + NLP Embeddings:** `0.6639 RMSLE`
*   **Production Model (Clean, Leak-Free):** `0.5824 RMSLE` (🎯 *Within target range!*)

---

## 📂 Project Structure

```
Github-star-predictor/
├── app.py                      # Premium Streamlit Dashboard App
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── results/
│   ├── final_model_clean.pkl   # Leakage-free LightGBM Model (67MB)
│   ├── final_model.pkl         # Leaked LightGBM Model (104MB, tracked via LFS)
│   └── feature_importance_final_clean.csv
├── main/
│   ├── src/                    # Feature engineering and training scripts
│   └── deployment/             # Deployment testing environment
└── data/                       # Local database & source data (Git ignored)
```

---

## 🔑 GitHub Personal Access Token (PAT)
For standard usage, the app requests unauthenticated data from GitHub (limited to 60 requests/hour). If you experience rate limits, paste a standard **Personal Access Token** in the Sidebar to increase the ceiling to 5,000 requests/hour! No permissions are required for the token (read-only public repository access is default).
