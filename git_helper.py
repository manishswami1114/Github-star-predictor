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
    
    # 1. git status
    if not run_cmd(["git", "status"]):
        sys.exit(1)
        
    # 2. git add
    print("\nStaging files...")
    if not run_cmd(["git", "add", "."]):
        sys.exit(1)
        
    # 3. git commit
    print("\nCommitting files...")
    # Check if there is anything to commit
    res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if not res.stdout.strip():
        print("Nothing to commit, working tree clean.")
    else:
        if not run_cmd(["git", "commit", "-m", "feat: Add interactive Streamlit app, leakage-free model, and Git LFS config"]):
            sys.exit(1)
            
    # 4. git push
    print("\nPushing to GitHub remote...")
    # Attempting to push to remote main branch
    # If the user needs credential prompt, they will see it or ssh-agent will handle it.
    run_cmd(["git", "push", "-u", "origin", "main", "--force"])

if __name__ == "__main__":
    main()
