#!/bin/bash
echo "=== Git Push Helper ==="
echo "Initializing Git LFS..."
git lfs install

echo "Staging files..."
git add .

echo "Committing files..."
git commit -m "feat: Add interactive Streamlit app, leakage-free model, and Git LFS config"

echo "Pushing to GitHub..."
git push -u origin main --force

echo "=== Git Push Complete! ==="
