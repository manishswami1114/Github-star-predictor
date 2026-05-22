import subprocess
import sys

def run_cmd(cmd):
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"✅ Success:\n{res.stdout}")
    else:
        print(f"❌ Error (code {res.returncode}):\n{res.stderr}")
    return res.returncode == 0

def main():
    print("=== GIT HELPER (UNSANDBOXED) ===")
    
    # 1. Update remote URL
    print("\nUpdating remote URL to swamimanish/Github-star-predictor...")
    # Try set-url first, if it fails, add origin
    if not run_cmd(["git", "remote", "set-url", "origin", "https://github.com/swamimanish/Github-star-predictor.git"]):
        run_cmd(["git", "remote", "add", "origin", "https://github.com/swamimanish/Github-star-predictor.git"])
        
    # 2. git status
    if not run_cmd(["git", "status"]):
        sys.exit(1)
        
    # 3. git add
    print("\nStaging files...")
    if not run_cmd(["git", "add", "."]):
        sys.exit(1)
        
    # 4. git commit
    print("\nCommitting files...")
    res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if not res.stdout.strip():
        print("Nothing to commit, working tree clean.")
    else:
        if not run_cmd(["git", "commit", "-m", "feat: Update app for HF Spaces, adjust UI, and configure git remote"]):
            sys.exit(1)
            
    # 5. git push
    print("\nPushing to GitHub remote...")
    run_cmd(["git", "push", "-u", "origin", "main", "--force"])

if __name__ == "__main__":
    main()
