#!/usr/bin/env python3
import os
import sys
import json
import boto3
import requests
from dotenv import load_dotenv

load_dotenv()

def get_pr_diff():
    """Get the PR diff from GitHub API"""
    repo = os.getenv('GITHUB_REPOSITORY')
    pr_number = os.getenv('PR_NUMBER')
    token = os.getenv('GITHUB_TOKEN')
    
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {'Authorization': f'token {token}'}
    
    response = requests.get(f"{url}.diff", headers=headers)
    return response.text

def review_with_llm(diff_text):
    """Send diff to Amazon Bedrock for review"""
    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    prompt = f"""Review this code diff and provide brief, actionable feedback:

{diff_text}

Focus on: bugs, security issues, code quality. Be concise."""

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}]
    })
    
    response = bedrock.invoke_model(
        body=body,
        modelId="anthropic.claude-3-haiku-20240307-v1:0"
    )
    
    result = json.loads(response['body'].read())
    return result['content'][0]['text']

def post_review(review_text):
    """Post review as PR comment"""
    repo = os.getenv('GITHUB_REPOSITORY')
    pr_number = os.getenv('PR_NUMBER')
    token = os.getenv('GITHUB_TOKEN')
    
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {'Authorization': f'token {token}'}
    
    data = {'body': f"## 🤖 AI Code Review\n\n{review_text}"}
    requests.post(url, headers=headers, json=data)

if __name__ == "__main__":
    diff = get_pr_diff()
    if diff.strip():
        review = review_with_llm(diff)
        post_review(review)
        print("Review posted successfully")
    else:
        print("No diff found")