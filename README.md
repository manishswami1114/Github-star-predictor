---
title: Github Star Predictor
emoji: ⭐
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.35.0
app_file: src/streamlit_app.py
pinned: false
---

# 🌟 GitHub Star Predictor

<p align="center">
  <img src="project_thumbnail.png" alt="GitHub Star Predictor Cover" width="650" style="border-radius: 16px; box-shadow: 0 10px 35px rgba(139, 92, 246, 0.4);">
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/swamimanish/Github-star-predictor" target="_blank">
    <img src="https://img.shields.io/badge/%F0%9F%A5%97%20Hugging%20Face-Live%20Demo-blueviolet?style=for-the-badge&logo=huggingface" alt="Live Demo on Hugging Face Spaces">
  </a>
  <a href="https://github.com/swamimanish/Github-star-predictor" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>
</p>

---

### 🔗 Try the Live App Immediately!
> **No installation or coding required!** You can play with the live machine learning model, fetch real-time GitHub repositories, or design your own hypothetical project right now in your web browser:
> ### 👉 [Click Here to Open the Live App on Hugging Face Spaces!](https://huggingface.co/spaces/swamimanish/Github-star-predictor) 🌟

---

## 💡 What is the GitHub Star Predictor?

Have you ever wondered what makes some software projects on GitHub go viral, while others remain undiscovered? 

The **GitHub Star Predictor** is a smart, interactive tool powered by artificial intelligence. It acts like a crystal ball for developers, product managers, and open-source enthusiasts. By evaluating **416 distinct data points**, the app estimates the total number of stargazers (popularity/likes) a repository is expected to receive based on its structural characteristics!

---

## 🎨 How it Works (Simply Explained)

Instead of just guessing, our AI looks at a project the way a human reviewer would:

1.  **👤 Creator Reputation (Success History):** Has the owner created popular repositories in the past? Do they have a loyal following? A developer with high historical engagement naturally gives a new project a massive headstart.
2.  **📝 Pitch & Documentation Quality (AI NLP):** We use a advanced language model to read the repository's description. The AI analyzes keywords, readability, and structural elements to determine how engaging and clear the documentation is.
3.  **⚡ Project Growth Speed (Velocity):** We check safe growth metrics—like how active the project is relative to its age (e.g., issues and forks opened per day).
4.  **🏷️ Trendy Topics:** Does the project mention high-demand modern keywords like **AI, LLM, RAG, React, or DevOps**?

---

## 🚀 How to Use the App

The dashboard offers two simple, interactive modes:

### 🔍 Mode 1: Auto-Fetch via GitHub URL
*   **What it does:** Paste the link to any public GitHub repository (e.g., `https://github.com/django/django`).
*   **The magic:** The app automatically queries the live GitHub API, downloads the owner's history, analyzes the description, runs it through the AI, and outputs a predicted star rating. 
*   **Comparison:** It compares the AI's estimate to the actual current star count, showing whether the project is overperforming, on-track, or underperforming!

### ✍️ Mode 2: Manual Sandbox Mode (Design Your Own!)
*   **What it does:** Let your imagination run wild! Fill in a hypothetical repository name, choose a primary language, slide its age and size, and write a custom README summary.
*   **The magic:** Watch in real-time how changing a description or selecting a trendy topic affects your project's popularity score *before* you even publish it!

---

## 🛠️ Installation & Local Setup (For Developers)

If you are a developer and want to run the app locally on your machine, follow these steps:

### 1. Clone & Navigate to Folder
```bash
# Clone the repository
git clone https://github.com/swamimanish/Github-star-predictor.git
cd Github-star-predictor
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run src/streamlit_app.py
```

---

## 🔬 Behind the Scenes (The Technical Specs)

For the data scientists and developers, here is the technical breakdown of our ML pipeline:

*   **Tree Ensembles (LightGBM):** Built using an optimized LightGBM model utilizing 416 handcrafted features trained on standard RMSLE loss.
*   **Sentence Transformers (`all-MiniLM-L6-v2`):** Used to compute a **384-dimensional dense vector representation** of README files to capture semantic structure offline/online.
*   **Anti-Leakage Audit:** A core victory in our Week 5 milestone was solving the **Target Leakage Trap**. By removing direct star-based velocity features (e.g., `stars_per_day`), we resolved artificial overfitting (which yielded an unrealistic 0.1491 RMSLE) and achieved a highly generalizable and robust **Mean CV RMSLE of 0.5824**—perfect for predicting emerging projects!

### 📂 File Structure
```
Github-star-predictor/
├── src/
│   └── streamlit_app.py        # Interactive Premium Streamlit App
├── results/
│   ├── final_model_clean.pkl   # Leakage-Free LightGBM Model (67MB)
│   ├── final_model.pkl         # Leaked LightGBM Model (104MB, tracked via LFS)
│   └── feature_importance_final_clean.csv
├── requirements.txt            # Python dependencies
├── README.md                   # Beautiful documentation & Space YAML metadata
├── project_thumbnail.png       # Sleek 3D cover art
├── push.sh                     # Helper script to sync GitHub
└── deploy_hf.sh                # Helper script to deploy directly to Hugging Face
```

---

## 🔑 GitHub Personal Access Token (PAT)
The app uses public GitHub API requests by default (limited to 60/hour). If you hit rate limits, paste your read-only **Personal Access Token** in the Sidebar to increase your ceiling to 5,000 requests/hour instantly.

---
Created with ❤️ by **Manish Swami**. Live at [Hugging Face Spaces](https://huggingface.co/spaces/swamimanish/Github-star-predictor).
