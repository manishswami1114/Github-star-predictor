#!/bin/bash
echo "=== Hugging Face Spaces Deployer ==="
read -p "Enter your Hugging Face Username (default: swamimanish): " hf_user
hf_user=${hf_user:-swamimanish}

read -p "Enter your Hugging Face Space Name (default: github-star-predictor): " hf_space
hf_space=${hf_space:-github-star-predictor}

read -sp "Enter your Hugging Face Write Token (get from https://huggingface.co/settings/tokens): " hf_token
echo ""

if [ -z "$hf_token" ]; then
    echo "❌ Error: Hugging Face Write Token is required."
    exit 1
fi

echo "Configuring Hugging Face remote..."
# Remove existing 'hf' remote if it exists
git remote remove hf 2>/dev/null
# Add the new remote with credentials embedded securely for this transaction
git remote add hf "https://${hf_user}:${hf_token}@huggingface.co/spaces/${hf_user}/${hf_space}"

echo "Staging files..."
git add .

echo "Committing files (if any)..."
git commit -m "deploy: Deploy to Hugging Face Spaces" 2>/dev/null || echo "Nothing new to commit"

echo "Pushing to Hugging Face Spaces..."
git push hf main --force

echo "Cleaning up local credentials from remote..."
git remote set-url hf "https://huggingface.co/spaces/${hf_user}/${hf_space}"

echo "=== Deployment Complete! ==="
echo "Your app should be building and running shortly at:"
echo "👉 https://huggingface.co/spaces/${hf_user}/${hf_space}"
