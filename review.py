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
    bedrock = boto3.client('bedrock-runtime', region_name=os.getenv('AWS_REGION', 'us-east-1'))
    
    # Use inference profile or cross-region inference ID
    # Example: "us.anthropic.claude-3-5-sonnet-20241022-v2:0" for cross-region
    # or "arn:aws:bedrock:us-east-1:123456789:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    
    prompt = f"""You are an expert code reviewer. Analyze this Git diff and provide a thorough code review.

## Code Diff to Review:
{diff_text}

## Review Instructions:
Provide a comprehensive analysis covering:

1. **Code Quality**
   - Design patterns and architecture
   - Code readability and maintainability
   - Adherence to best practices
   - DRY (Don't Repeat Yourself) violations
   - Function/variable naming conventions

2. **Potential Bugs**
   - Logic errors
   - Edge cases not handled
   - Null/undefined reference issues
   - Type mismatches
   - Resource leaks

3. **Security Issues**
   - Input validation problems
   - SQL injection risks
   - XSS vulnerabilities
   - Sensitive data exposure
   - Authentication/authorization issues

4. **Performance Concerns**
   - Inefficient algorithms (O(n²) where O(n) would work)
   - Unnecessary database queries
   - Memory leaks
   - Unoptimized loops

5. **Testing Considerations**
   - Missing test coverage
   - Edge cases that need testing
   - Integration points requiring validation

Format your response as:
- Start with a brief summary (2-3 lines)
- List findings by severity: 🔴 Critical, 🟠 Important, 🟡 Minor, 🟢 Good practices observed
- Provide specific line references where applicable
- Suggest concrete improvements with code examples when helpful
- End with overall recommendation (approve, needs changes, requires discussion)

Be specific and actionable in your feedback."""

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": prompt}]
    })
    
    response = bedrock.invoke_model(
        body=body,
        modelId=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
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
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 201:
        print(f"✅ Review posted successfully to PR #{pr_number}")
        return True
    else:
        print(f"❌ Failed to post review: {response.status_code} - {response.text}")
        return False

if __name__ == "__main__":
    try:
        print(f"Starting code review for PR #{os.getenv('PR_NUMBER')}...")
        
        diff = get_pr_diff()
        if not diff.strip():
            print("No diff found - skipping review")
            sys.exit(0)
        
        print(f"Analyzing {len(diff.splitlines())} lines of diff...")
        review = review_with_llm(diff)
        
        print("Posting review to GitHub...")
        success = post_review(review)
        
        if success:
            print("\n=== Review Summary ===")
            print(review[:500] + "..." if len(review) > 500 else review)
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"❌ Error during review: {str(e)}")
        sys.exit(1)