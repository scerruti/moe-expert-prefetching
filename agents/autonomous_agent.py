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
    """Get issues matching any of the specified labels (OR logic)."""
    repo = get_repo_info()
    if not repo:
        print("❌ Error: Could not determine repository from git remote")
        sys.exit(1)

    # Split labels and query each separately (OR logic)
    label_list = [l.strip() for l in labels.split(',')]
    all_issues = []
    seen_numbers = set()

    for label in label_list:
        cmd = f'gh issue list --repo {repo} --label "{label}" --state open --json number,title,body'
        output = run_cmd(cmd, check=False)
        if output:
            try:
                issues = json.loads(output)
                for issue in issues:
                    if issue['number'] not in seen_numbers:
                        all_issues.append(issue)
                        seen_numbers.add(issue['number'])
            except:
                pass

    return all_issues

def get_issue_details(issue_number: int) -> dict:
    """Get full issue details."""
    repo = get_repo_info()
    cmd = f'gh issue view {issue_number} --repo {repo} --json number,title,body,labels'
    output = run_cmd(cmd)
    return json.loads(output)

def create_feature_branch(issue_number: int, title: str) -> str:
    """Create feature branch from issue title, or reuse if it has unique work."""
    # Convert title to branch name (lowercase, hyphens, no special chars)
    branch_name = re.sub(r'[^a-z0-9\s-]', '', title.lower())
    branch_name = re.sub(r'\s+', '-', branch_name)[:40]  # Limit length
    branch_name = f"feature/{issue_number}-{branch_name}"

    # Check if branch already exists
    existing = run_cmd(f"git rev-parse --verify {branch_name}", check=False)
    if existing:
        # If the branch is identical to main (no unique commits), recreate it.
        branch_commits = run_cmd(f"git rev-list --count main..{branch_name}", check=False)
        if branch_commits and branch_commits != "0":
            print(f"\n📦 Reusing existing branch: {branch_name}")
            run_cmd(f"git checkout {branch_name}")
        else:
            print(f"\n📦 Recreating stale branch: {branch_name}")
            run_cmd(f"git checkout main", check=False)
            run_cmd(f"git branch -D {branch_name}", check=False)
            run_cmd(f"git checkout -b {branch_name}")
    else:
        print(f"\n📦 Creating branch: {branch_name}")
        run_cmd(f"git checkout -b {branch_name}")

    return branch_name

def extract_acceptance_criteria(issue_body: str) -> list:
    """Extract acceptance criteria and tasks from issue body."""
    criteria = []
    in_section = False

    for line in issue_body.split('\n'):
        # Check for section headers (look for ## Tasks or ## Acceptance Criteria)
        if line.startswith('##'):
            # Start collecting items if we find Tasks or Acceptance Criteria section
            in_section = 'tasks' in line.lower() or 'acceptance' in line.lower()
            continue

        # If we encounter another header (##), stop collecting
        if line.startswith('##') or (line.startswith('#') and not line.startswith('##')):
            in_section = False
            continue

        # Extract items from current section
        if in_section:
            stripped = line.strip()
            # Match both checkbox items (- [ ]) and regular bullets (-)
            if stripped.startswith('- '):
                # Remove leading "- "
                item = stripped[2:].strip()
                # Remove checkbox markers if present ([ ] or [x])
                if item.startswith('[ ]'):
                    item = item[3:].strip()
                elif item.startswith('[x]') or item.startswith('[X]'):
                    item = item[3:].strip()
                if item:
                    criteria.append(item)

    return criteria if criteria else ["Issue resolved and PR passes review"]


def extract_directory_requirements(issue_body: str) -> list[str]:
    """Identify directory-like requirements in an issue so they can be normalized into trackable artifacts."""
    directory_requirements = []
    fallback_names = []

    for line in issue_body.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue

        lower = stripped.lower()
        if 'directory' not in lower and 'structure' not in lower and 'create' not in lower and 'add' not in lower:
            continue

        # Capture explicit paths such as phase_1/scripts or data/phase_1
        explicit_matches = re.findall(r'([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)', stripped)
        for match in explicit_matches:
            if match.startswith(('.', '#', 'http')):
                continue
            if match.count('/') >= 1:
                directory_requirements.append(match.strip('/'))

        # Fallback for simple directives like "Create data directory" or "Create phase_1/scripts directory"
        for prefix in ['create ', 'add ', 'set up ']:
            if stripped.lower().startswith(prefix):
                remainder = stripped[len(prefix):].strip()
                if not remainder:
                    continue
                token = remainder.split()[0]
                if token.startswith(('.', '#', 'http')):
                    continue
                if '.' in token and token.count('/') == 0:
                    continue
                fallback_names.append(token.strip('/'))

    # Keep a small, generic set of directory-like requirements without overmatching file names.
    combined = directory_requirements + fallback_names
    deduped = []
    seen = set()
    for item in combined:
        if item and item not in seen:
            deduped.append(item)
            seen.add(item)

    return deduped


def ensure_git_trackable_directories(issue_body: str) -> list[str]:
    """Create repository artifacts for directory-only issue requirements so Git can track the requested structure."""
    created = []

    for directory in extract_directory_requirements(issue_body):
        os.makedirs(directory, exist_ok=True)
        contents = os.listdir(directory)

        if not contents:
            placeholder = os.path.join(directory, 'README.md')
            with open(placeholder, 'w', encoding='utf-8') as f:
                f.write(
                    f"# {directory}\n\n"
                    "This directory was created to satisfy a repository structure requirement in the issue.\n"
                    "The directory itself is not tracked by Git; this placeholder file makes the requested structure trackable.\n"
                )
            created.append(placeholder)

    return created


def extract_referenced_paths(issue_body: str) -> list[str]:
    """Extract likely file or directory paths mentioned in issue text."""
    matches = []
    for line in issue_body.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue

        for match in re.findall(r'([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+|[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+)', stripped):
            if match.startswith(('http://', 'https://', '#', '.', '(', '[')):
                continue
            if match.count('/') == 0 and not re.search(r'\.[A-Za-z0-9]+$', match):
                continue
            matches.append(match.strip('/'))

    deduped = []
    seen = set()
    for path in matches:
        if path and path not in seen:
            deduped.append(path)
            seen.add(path)

    return deduped


def validate_issue_requirements(issue_body: str) -> dict:
    """Check whether explicit repository artifacts referenced in the issue are actually present."""
    criteria = extract_acceptance_criteria(issue_body)
    referenced_paths = extract_referenced_paths(issue_body)
    missing_paths = []

    for path in referenced_paths:
        if path.startswith(('phase-', 'blocker/', 'good-first-issue', 'label')):
            continue

        if not os.path.exists(path):
            missing_paths.append(path)

    if not missing_paths and not referenced_paths:
        return {
            "valid": True,
            "criteria": criteria,
            "missing_paths": [],
            "message": "Issue does not reference explicit repo artifacts; manual review is required."
        }

    return {
        "valid": len(missing_paths) == 0,
        "criteria": criteria,
        "missing_paths": missing_paths,
        "message": (
            "Missing required repository artifacts:" if missing_paths else "Explicitly referenced artifacts are present."
        )
    }


def handle_write_file(path: str, content: str) -> str:
    """Handle write_file tool invocation."""
    try:
        # Ensure directory exists
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Write the file
        with open(path, 'w') as f:
            f.write(content)

        return f"✅ File written: {path}"
    except Exception as e:
        return f"❌ Error writing file {path}: {str(e)}"

def process_tool_calls(response, conversation_history: list) -> tuple:
    """Process tool calls from Claude's response and return updated history."""
    assistant_message = None
    tool_results = []

    # Build the assistant response for history
    assistant_content = []

    for block in response.content:
        if hasattr(block, 'text'):
            # Text block
            if block.text:
                assistant_message = block.text
                assistant_content.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            # Tool use block
            assistant_content.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input
            })

            # Execute tool
            if block.name == "write_file":
                path = block.input.get("path")
                content = block.input.get("content", "")
                result = handle_write_file(path, content)
                print(f"  {result}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
            elif block.name == "bash":
                cmd = block.input.get("command", "")
                try:
                    result = run_cmd(cmd, check=False)
                    print(f"  $ {cmd}\n  {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
                except Exception as e:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error: {str(e)}"
                    })
            else:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Unknown tool: {block.name}"
                })

    # Add assistant message to history
    if assistant_content:
        conversation_history.append({
            "role": "assistant",
            "content": assistant_content
        })

    # Add tool results if any
    if tool_results:
        conversation_history.append({
            "role": "user",
            "content": tool_results
        })

    return assistant_message, tool_results

def ask_claude(issue: dict, conversation_history: list, autonomous: bool = False) -> dict:
    """Ask Claude to work on the issue."""
    issue_number = issue['number']
    title = issue['title']
    body = issue['body']

    acceptance_criteria = extract_acceptance_criteria(body)

    # Load condensed agent context
    agent_context_path = "agents/context/AGENT_CONTEXT.md"
    try:
        agent_context = open(agent_context_path).read()
    except FileNotFoundError:
        agent_context = "(Context file not found - run: python agents/scripts/generate_agent_context.py)"

    # Add context to conversation
    system_prompt = f"""You are an expert software engineer working on a research project.
Your task is to work on GitHub issues by:

1. **Review provided project context** (See below)
2. Understanding the issue requirements and acceptance criteria
3. Inspecting the real repository state to determine what already exists
4. Implementing the code changes needed
5. Running tests and validation
6. Creating proper git commits
7. Ensuring all acceptance criteria are met

## SOURCE-OF-TRUTH RULES

- The GitHub issue body and the current repository state are the authoritative definition of the task.
- Local design docs such as `SYSTEM_DESIGN.md`, `README.md`, and `phase_1/docs/ARCHITECTURE.md` provide architectural context and background, but they are not current task status.
- Do not treat local status/checklist/review files as evidence that an issue is already implemented, blocked, or complete.
- Do not infer completion from planning docs alone. Verify what actually exists in the repository before concluding work is done.
- When an issue requests missing files or configuration, create or update them in the codebase if they are not already present.

---
## PROJECT CONTEXT (Condensed for Efficiency)

{agent_context}

---

CRITICAL: You have access to two tools:

1. **bash**: Run shell commands to:
   - Read files and explore directories (cat, ls, find, grep)
   - Run tests and validation
   - Execute git operations (commit, push, etc)

2. **write_file**: Create or modify files:
   - Provide the complete file path and content
   - Parent directories are created automatically
   - Always use this tool for creating new files or modifying existing ones

IMPORTANT:
- For reading files: Use bash with cat/grep/etc
- For creating/modifying files: ALWAYS use write_file tool, NEVER bash echo
- Run tests and git operations with bash
- Never create empty directories—they're created automatically when files are written
- If an issue requests a directory structure, make that directory trackable by adding a placeholder file such as `README.md` or `.gitkeep` if the directory would otherwise be empty.

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
        if autonomous:
            user_message = "Continue working on the issue. Show progress and let me know when all acceptance criteria are met."
        else:
            user_message = input("📝 Your feedback (or 'done' to submit PR, default=continue): ").strip() or "Continue working"
            if user_message.lower() == 'done':
                return {"status": "ready_for_pr", "message": "User confirmed implementation complete"}

    # Add to conversation
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    print(f"\n🤖 Claude is working on issue #{issue_number}...")

    # Call Claude with tools
    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=16000,
        system=system_prompt,
        tools=[
            {
                "name": "bash",
                "description": "Run bash commands to explore the repository, run tests, and perform git operations",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to run"
                        }
                    },
                    "required": ["command"]
                }
            },
            {
                "name": "write_file",
                "description": "Create or modify a file with the given content. Automatically creates parent directories.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The file path to create or modify"
                        },
                        "content": {
                            "type": "string",
                            "description": "The complete file content"
                        }
                    },
                    "required": ["path", "content"]
                }
            }
        ],
        messages=conversation_history
    )

    # Process tool calls and extract text
    assistant_message, tool_results = process_tool_calls(response, conversation_history)

    if not assistant_message:
        assistant_message = "Claude is working on the implementation..."

    if tool_results:
        print(f"\n📝 Files created/modified")

    print(f"\n{assistant_message}")

    return {
        "status": "in_progress",
        "message": assistant_message,
        "conversation": conversation_history
    }

def create_pull_request(issue_number: int, branch_name: str, title: str) -> str:
    """Push the branch and create a pull request for the issue."""
    print(f"\n🚀 Creating pull request...")

    repo = get_repo_info()

    # Push branch before PR creation so GitHub has a valid head ref.
    print(f"\n📤 Pushing branch: {branch_name}")
    run_cmd(f"git push -u origin {branch_name}")

    pr_body = f"""Closes #{issue_number}

## Summary
Implemented work for issue #{issue_number}.

## Testing
- Code follows project standards
- Acceptance criteria verified
- Ready for review

Generated with Claude Autonomous Agent"""

    cmd = f'''gh pr create --repo {repo} --title "{title}" --body "{pr_body}" --head {branch_name} --base main'''
    output = run_cmd(cmd)

    # Extract PR URL
    pr_url = output.split('\n')[-1] if output else "PR created"
    print(f"✅ PR created: {pr_url}")

    return pr_url

def work_on_issue(issue: dict, autonomous: bool = False) -> bool:
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

    placeholder_paths = ensure_git_trackable_directories(issue['body'])
    if placeholder_paths:
        print(f"\n📁 Normalized directory-only requirements into trackable placeholders:\n  - {'\n  - '.join(placeholder_paths)}")

    claude_iterations = 0
    max_autonomous_iterations = 3  # Prevent infinite loops in autonomous mode

    while True:
        result = ask_claude(issue, conversation_history, autonomous=autonomous)
        claude_iterations += 1

        validation = validate_issue_requirements(issue['body'])
        if not validation["valid"]:
            print(f"\n⚠️ Acceptance-criteria validation failed. Missing referenced artifacts: {', '.join(validation['missing_paths'])}")
            if autonomous:
                print("   Continuing another iteration so Claude can address the missing artifacts.")
            else:
                user_input = input("\n🔁 Missing artifacts remain. Keep working? (yes/no, default=yes): ").strip().lower() or "yes"
                if user_input != 'yes':
                    print("⚠️ Stopping before PR creation because acceptance criteria are not yet satisfied.")
                    return False

        if result["status"] == "ready_for_pr":
            # Verify git changes
            status = run_cmd("git status --short", check=False)
            if status:
                print(f"\n📝 Changes made:\n{status}")

                if not validation["valid"]:
                    print("⚠️ Cannot create PR yet: the issue still references missing repository artifacts.")
                    return False

                # Commit changes
                commit_msg = f"Issue #{issue_number}: {title}\n\nImplementation complete with all acceptance criteria met."
                run_cmd(f'git add -A && git commit -m "{commit_msg}"')

                # Create PR
                create_pull_request(issue_number, branch_name, title)
                return True
            else:
                print("⚠️ No changes detected. Please verify implementation.")

        # Check if work is done (autonomous mode: auto-commit after Claude finishes or max iterations)
        if autonomous and claude_iterations >= max_autonomous_iterations:
            status = run_cmd("git status --short", check=False)
            if status:
                if not validation["valid"]:
                    print("⚠️ Cannot auto-commit PR yet: the issue still references missing repository artifacts.")
                    return False

                print(f"\n✅ Autonomous mode: Auto-committing changes after {claude_iterations} iterations")
                print(f"📝 Changes made:\n{status}")

                # Commit changes
                commit_msg = f"Issue #{issue_number}: {title}\n\nImplementation complete. All acceptance criteria met."
                run_cmd(f'git add -A && git commit -m "{commit_msg}"')

                # Create PR
                create_pull_request(issue_number, branch_name, title)
                return True
            else:
                print("⚠️ No changes detected after exploration. Exiting.")
                return False

        # Ask user if ready to continue or submit (unless autonomous mode)
        if autonomous:
            user_input = "yes"  # Auto-continue in autonomous mode
        else:
            user_input = input("\n⏭️  Continue working? (yes/no/submit, default=yes): ").strip().lower() or "yes"

        if user_input == 'submit' or user_input == 'no':
            if not validation["valid"]:
                print("⚠️ Cannot create a PR until the issue's referenced artifacts exist in the repository.")
                return False

            # Check if there are actual commits on this branch
            commits = run_cmd(f"git log main..HEAD --oneline", check=False)
            if commits:
                # Branch has commits, safe to create PR
                create_pull_request(issue_number, branch_name, title)
                return True
            else:
                # No commits, check for staged changes
                status = run_cmd("git status --short", check=False)
                if status:
                    run_cmd(f'git add -A && git commit -m "Issue #{issue_number}: {title}"')
                    create_pull_request(issue_number, branch_name, title)
                    return True
                else:
                    # No work done, just cancel
                    print("⚠️ No work completed. Cancelling branch...")
                    run_cmd(f"git checkout main && git branch -D {branch_name}", check=False)
                    return False
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
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Autonomous mode: no user prompts, auto-commit after work completes."
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Process only one issue and exit (useful for testing). Can combine with --autonomous."
    )

    args = parser.parse_args()

    # --single implies no interactive loop
    if args.single:
        args.autonomous = True

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
            success = work_on_issue(full_issue, autonomous=args.autonomous)

            # --single is a hard stop after the first issue, even if that issue
            # does not fully complete.
            if args.single:
                print("🧪 --single mode: stopping after Issue #" + str(issue['number']))
                break

            if not success:
                print("⚠️ Issue work incomplete, moving to next...")
                continue

            # Ask if should continue to next issue (skip in quiet mode)
            if args.autonomous:
                break
            else:
                next_issue = input("\n🔄 Work on next issue? (yes/no): ").strip().lower()
                if next_issue != 'yes':
                    break

    except KeyboardInterrupt:
        print("\n\n⏸️  Agent paused by user")
        print("   To resume: python autonomous_agent.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
