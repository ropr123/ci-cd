import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def trigger_workflow():
    pat = os.getenv('GITHUB_PAT')
    repo = os.getenv('GITHUB_REPO')
    workflow_id = os.getenv('GITHUB_WF_ID')
    branch = os.getenv('GITHUB_BRANCH', 'master')

    if not all([pat, repo, workflow_id]):
        print("Error: GITHUB_PAT, GITHUB_REPO, or GITHUB_WF_ID not found in environment.")
        print("Please update your .env file.")
        return

    # GitHub API endpoint for triggering a workflow_dispatch event
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_id}/dispatches"

    headers = {
        "Authorization": f"token {pat}",
        "Accept": "application/vnd.github.v3+json"
    }

    data = {
        "ref": branch
    }

    print(f"Triggering workflow '{workflow_id}' in repository '{repo}' on branch '{branch}'...")
    
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 204:
        print("Successfully triggered workflow!")
    else:
        print(f"Failed to trigger workflow. Status code: {response.status_code}")
        print(f"Response: {response.text}")

if __name__ == "__main__":
    trigger_workflow()
