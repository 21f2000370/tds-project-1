import json                   # to convert API to json format
from urllib.parse import urlencode

import requests               # to get the webpage

import pandas as pd
import duckdb

import re                     # regular expression operators

from datetime import datetime
import time


def fetch_users(location, min_followers, token):
    url = "https://api.github.com/search/users"
    headers = {"Authorization": f"token {token}"}
    params = {
        "q": f"location:{location} followers:>={min_followers}",
        "per_page": 100  # max allowed per page
    }
    users = []
    page = 1

    while True:
        params["page"] = page
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        if "items" in data:
            users.extend(data["items"])
            if len(data["items"]) < 100:  # Last page
                break
            page += 1
        else:
            break

    return users

def fetch_user_details(username, token):
    url = f"https://api.github.com/users/{username}"
    headers = {"Authorization": f"token {token}"}
    response = requests.get(url, headers=headers)
    return response.json()

def fetch_repo_details(public_repo_url, token):
    headers = {"Authorization": f"token {token}"}
    response = requests.get(public_repo_url, headers=headers)
    return response.json()

# main
token = "<TOKEN>" # replace token with your acce token 
users = fetch_users("Paris", 200, token)

detailed_users = []
detailed_repos = []
for user in users:
    details = fetch_user_details(user["login"], token)
    # specific fields
    user_info = {
        "login": user["login"],
        "name": details.get("name"),
        "company": details.get("company"),
        "location": details.get("location"),
        "email": details.get("email"),
        "hireable": details.get("hireable"),
        "bio": details.get("bio"),
        "public_repos": details.get("public_repos"),
        "followers": details.get("followers"),
        "following": details.get("following"),
        "created_at": details.get("created_at"),
    }
    detailed_users.append(user_info)
    repo_details = fetch_repo_details(user["repos_url"], token)
    for repo in repo_details:
      # specific fields
      detailed_repos.append({
          "login": user["login"],
          "full_name": repo["full_name"],
          "created_at": repo["created_at"],
          "stargazers_count": repo["stargazers_count"],
          "watchers_count": repo["watchers_count"],
          "language": repo["language"],
          "has_projects": repo["has_projects"],
          "has_wiki": repo["has_wiki"],
          "license_name": repo["license"]["key"] if repo["license"] else ""
      })
    time.sleep(1)  # To avoid hitting rate limits

user_data = pd.DataFrame(detailed_users)
user_data.to_csv("users.csv")
repo_data = pd.DataFrame(detailed_repos)
repo_data.to_csv("repositories.csv")
