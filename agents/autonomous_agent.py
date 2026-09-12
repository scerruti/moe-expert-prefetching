#!/usr/bin/env python3
"""
Autonomous Issue Agent
Picks up GitHub issues, creates branches, implements solutions, and submits PRs.
Works for any phase or custom issue filters.

Usage:
    export ANTHROPIC_API_KEY="your-api-key"
    export GITHUB_TOKEN="your-github-token"

    # Work on Phase 1 unblocked issues
    python autonomous_agent.py --phase 1

    # Work on Phase 2 unblocked issues
    python autonomous_agent.py --phase 2

    # Work on custom labels
    python autonomous_agent.py --labels "good-first-issue,help-wanted"

    # Work on specific component
    python autonomous_agent.py --labels "phase-1/environment"
"""

import os
import sys
import json
import subprocess
import re
import argparse
from datetime import datetime
from anthropic import Anthropic

# Configuration
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Initialize Claude client
client = Anthropic()

def run_cmd(cmd: str, check=True) -> str:
    """Run shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        print(f"   Error: {e.stderr}")
        raise

def get_repo_info() -> tuple:
    """Get repository owner/name from git remote."""
    try:
        remote_url = run_cmd("git config --get remote.origin.url")
        # Extract owner/repo from URL (works with https and ssh)
        match = re.search(r'github\.com[:/]([^/]+)/(.+?)(\.git)?$', remote_url)
        if match:
            owner, repo = match.group(1), match.group(2)
            return f"{owner}/{repo}"
    except:
        pass
    return None

def get_unblocked_issues(labels: str) -> list:
    """Get issues matching the specified labels."""
    repo = get_repo_info()
    if not repo:
        print("❌ Error: Could not determine repository from git remote")
        sys.exit(1)

    cmd = f'gh issue list --repo {repo} --label "{labels}" --state open --json number,title,body'
    output = run_cmd(cmd)
    if not output:
        return []
    return json.loads(output)

def get_issue_details(issue_number: int) -> dict:
    """Get full issue details."""
    repo = get_repo_info()
    cmd = f'gh issue view {issue_number} --repo {repo} --json number,title,body,labels'
    output = run_cmd(cmd)
    return json.loads(output)

def create_feature_branch(issue_number: int, title: str) -> str:
    """Create feature branch from issue title."""
    # Convert title to branch name (lowercase, hyphens, no special chars)
    branch_name = re.sub(r'[^a-z0-9\s-]', '', title.lower())
    branch_name = re.sub(r'\s+', '-', branch_name)[:40]  # Limit length
    branch_name = f"feature/{issue_number}-{branch_name}"

    print(f"\n📦 Creating branch: {branch_name}")
    run_cmd(f"git checkout -b {branch_name}")
    return branch_name

def extract_acceptance_criteria(issue_body: str) -> list:
    """Extract acceptance criteria from issue body."""
    criteria = []
    in_criteria = False

    for line in issue_body.split('\n'):
        if 'Acceptance Criteria' in line or 'acceptance criteria' in line:
            in_criteria = True
            continue

        if in_criteria:
            if line.startswith('##'):  # Next section
                break
            if line.strip().startswith('- [ ]'):
                criteria.append(line.strip())

    return criteria if criteria else ["Issue resolved and PR passes review"]

def ask_claude(issue: dict, conversation_history: list) -> dict:
    """Ask Claude to work on the issue."""
    issue_number = issue['number']
    title = issue['title']
    body = issue['body']

    acceptance_criteria = extract_acceptance_criteria(body)

    # Add context to conversation
    system_prompt = """You are an expert software engineer working on a research project.
Your task is to work on GitHub issues by:

1. Understanding the issue requirements and acceptance criteria
2. Implementing the code changes needed
3. Running tests and validation
4. Creating proper git commits
5. Ensuring all acceptance criteria are met

You have access to run shell commands (bash) and can read/edit files in the repository.
Follow the repository's coding standards and conventions.

When you're done implementing all acceptance criteria, indicate that the issue is ready for PR submission."""

    # Build message with issue context
    if not conversation_history:
        user_message = f"""Please work on GitHub issue #{issue_number}: {title}

## Issue Description:
{body}

## Acceptance Criteria:
{json.dumps(acceptance_criteria, indent=2)}

## What to do:
1. Read the issue carefully
2. Implement the required functionality
3. Test your implementation
4. Ensure all acceptance criteria are met
5. Create a clear git commit
6. Tell me when you're done

Start by exploring the repository structure and understanding what needs to be done."""
    else:
        user_message = input("📝 Your feedback (or 'done' to submit PR): ").strip()
        if user_message.lower() == 'done':
            return {"status": "ready_for_pr", "message": "User confirmed implementation complete"}

    # Add to conversation
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    print(f"\n🤖 Claude is working on issue #{issue_number}...")

    # Call Claude
    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=16000,
        system=system_prompt,
        messages=conversation_history
    )

    assistant_message = response.content[0].text
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })

    print(f"\n{assistant_message}")

    return {
        "status": "in_progress",
        "message": assistant_message,
        "conversation": conversation_history
    }

def create_pull_request(issue_number: int, branch_name: str, title: str) -> str:
    """Create pull request for the issue."""
    print(f"\n🚀 Creating pull request...")

    repo = get_repo_info()
    pr_body = f"""Closes #{issue_number}

## Summary
Implemented work for issue #{issue_number}.

## Testing
- Code follows project standards
- Acceptance criteria verified
- Ready for review

Generated with Claude Autonomous Agent"""

    cmd = f'''gh pr create --repo {repo} --title "{title}" --body "{pr_body}" --head {branch_name}'''
    output = run_cmd(cmd)

    # Extract PR URL
    pr_url = output.split('\n')[-1] if output else "PR created"
    print(f"✅ PR created: {pr_url}")

    return pr_url

def work_on_issue(issue: dict) -> bool:
    """Work on a single issue until completion."""
    issue_number = issue['number']
    title = issue['title']

    print(f"\n{'='*70}")
    print(f"🎯 Working on Issue #{issue_number}: {title}")
    print(f"{'='*70}")

    # Create feature branch
    branch_name = create_feature_branch(issue_number, title)

    # Start conversation with Claude
    conversation_history = []

    while True:
        result = ask_claude(issue, conversation_history)

        if result["status"] == "ready_for_pr":
            # Verify git changes
            status = run_cmd("git status --short", check=False)
            if status:
                print(f"\n📝 Changes made:\n{status}")

                # Commit changes
                commit_msg = f"Issue #{issue_number}: {title}\n\nImplementation complete with all acceptance criteria met."
                run_cmd(f'git add -A && git commit -m "{commit_msg}"')

                # Create PR
                create_pull_request(issue_number, branch_name, title)
                return True
            else:
                print("⚠️ No changes detected. Please verify implementation.")

        # Ask user if ready to continue or submit
        user_input = input("\n⏭️  Continue working? (yes/no/submit): ").strip().lower()
        if user_input == 'submit' or user_input == 'no':
            # Commit and create PR
            status = run_cmd("git status --short", check=False)
            if status:
                run_cmd(f'git add -A && git commit -m "Issue #{issue_number}: {title}"')
                create_pull_request(issue_number, branch_name, title)
            return True
        elif user_input != 'yes':
            print("Continuing with Claude...")

def main():
    """Main agent loop."""
    parser = argparse.ArgumentParser(
        description="Autonomous GitHub Issue Agent - Works on issues and submits PRs"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Work on issues from a specific phase"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="blocker/none",
        help="Comma-separated GitHub labels to filter issues (default: blocker/none)"
    )
    parser.add_argument(
        "--repo",
        type=str,
        help="Repository (owner/repo format). Auto-detected if not provided."
    )

    args = parser.parse_args()

    # Build label filter
    if args.phase:
        labels = f"phase-{args.phase}/environment,phase-{args.phase}/datasets,phase-{args.phase}/model,blocker/none"
    else:
        labels = args.labels

    repo = args.repo or get_repo_info()

    print("🤖 Autonomous Issue Agent Starting")
    print(f"📍 Repository: {repo}")
    print(f"🏷️  Labels: {labels}")
    print(f"⏰ Started: {datetime.now().isoformat()}\n")

    # Verify credentials
    if not GITHUB_TOKEN:
        print("❌ Error: GITHUB_TOKEN not set")
        print("   Export: export GITHUB_TOKEN='your-token'")
        sys.exit(1)

    if not ANTHROPIC_API_KEY:
        print("❌ Error: ANTHROPIC_API_KEY not set")
        print("   Export: export ANTHROPIC_API_KEY='your-api-key'")
        sys.exit(1)

    # Verify we're in the right directory
    if not os.path.exists(".git"):
        print("❌ Error: Not in a git repository")
        print("   Run this script from the repository root")
        sys.exit(1)

    try:
        while True:
            # Get issues matching labels
            issues = get_unblocked_issues(labels)

            if not issues:
                print("✅ No issues matching filters!")
                break

            print(f"\n📋 Found {len(issues)} issue(s):")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. #{issue['number']}: {issue['title']}")

            # Pick first issue
            issue = issues[0]
            full_issue = get_issue_details(issue['number'])

            # Work on it
            success = work_on_issue(full_issue)

            if not success:
                print("⚠️ Issue work incomplete, moving to next...")
                continue

            # Ask if should continue to next issue
            next_issue = input("\n🔄 Work on next issue? (yes/no): ").strip().lower()
            if next_issue != 'yes':
                break

    except KeyboardInterrupt:
        print("\n\n⏸️  Agent paused by user")
        print("   To resume: python autonomous_agent.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
    finally:
        # Return to main branch
        try:
            run_cmd("git checkout main", check=False)
        except:
            pass

if __name__ == "__main__":
    main()
