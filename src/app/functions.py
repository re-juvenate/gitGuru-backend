import os
import time
from glob import glob
import requests
import json
import getpass
import base64

timestr = lambda: time.strftime("%Y%m%d-%H%M%S")
DATA_PATH = "data/user_doc"


def get_file_path(file_name):
    file_path = os.path.abspath(file_name)  # Join the directory and file name
    return file_path


def get_flie_path_from_name(file_name):
    files = glob(os.path.join(DATA_PATH, "**", file_name), recursive=True)
    return get_file_path(files[0])



if "GITHUB_API_KEY" not in os.environ:
    os.environ["GITHUB_API_KEY"] = getpass.getpass("Github API Key:")
token = os.environ["GITHUB_API_KEY"]


def get_issue_body(owner, repo, issue_number):
    global token

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return "Failed to retrieve issue body."
    
    body = response.json()["body"]
    return body


def get_issue_comment(owner, repo, issue_number):
    global token

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return ["Failed to retrieve issue comments"]
    
    comment = response.json()
    comment_list = []
    for i in comment:
        comment_list.append(i["body"])
    return comment_list


def get_issue_readme(owner, repo):
    global token

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return "Failed to retrieve README."
    
    readme_enc = response.json()["content"]
    readme = base64.b64decode(readme_enc).decode("utf-8")
    return readme

