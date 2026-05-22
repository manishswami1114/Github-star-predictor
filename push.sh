#!/bin/bash
echo "=== Git Push Helper ==="
echo "Initializing Git LFS..."
git lfs install

echo "Configuring GitHub Remote..."
git remote set-url origin https://github.com/manishswami1114/Github-star-predictor.git 2>/dev/null || git remote add origin https://github.com/swamimanish/Github-star-predictor.git

echo "Staging files..."
git add .

echo "Committing files..."
git commit -m "feat: Update app for HF Spaces, adjust UI, and configure git remote"

echo "Pushing to GitHub (manishswami1114/Github-star-predictor)..."
git push -u origin main --force

echo "=== Git Push Complete! ==="
