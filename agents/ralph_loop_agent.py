#!/usr/bin/env python3
"""
Ralph Loop Agent - GitHub Actions Version
Implements the ralph loop pattern for autonomous issue resolution:
1. Planning phase: Create and refine implementation plan
2. Implementation phase: Execute plan and solve issue

Triggered by GitHub Actions when issue is labeled 'ready-for-agent'.
Uses GitHub Secrets for authentication.
Auto-submits PR with comprehensive failure documentation.

Environment variables (from GitHub Secrets):
  GITHUB_TOKEN - GitHub API access
  ANTHROPIC_API_KEY - Claude API access
  GITHUB_ISSUE_NUMBER - Issue number (set by Actions)
  GITHUB_REPOSITORY - Repository (owner/repo)
"""

import os
import sys
import json
import subprocess
import re
from datetime import datetime
from anthropic import Anthropic

# Configuration
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ISSUE_NUMBER = os.environ.get("GITHUB_ISSUE_NUMBER")
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "scerruti/moe-expert-prefetching")

# Ralph Loop Limits
PLANNING_ITERATIONS = 3
IMPLEMENTATION_ITERATIONS = 7

# Initialize Claude client
client = Anthropic()

def run_cmd(cmd: str, check=True) -> str:
    """Run shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        if check:
            print(f"❌ Command failed: {cmd}")
            print(f"   Error: {e.stderr}")
            raise
        return ""

def get_issue_details() -> dict:
    """Get issue details from GitHub."""
    cmd = f'gh issue view {ISSUE_NUMBER} --repo {GITHUB_REPOSITORY} --json number,title,body,labels'
    output = run_cmd(cmd)
    return json.loads(output)

def extract_acceptance_criteria(issue_body: str) -> list:
    """Extract acceptance criteria from issue body."""
    criteria = []
    in_criteria = False

    for line in issue_body.split('\n'):
        if 'Acceptance Criteria' in line or 'acceptance criteria' in line:
            in_criteria = True
            continue

        if in_criteria:
            if line.startswith('##'):
                break
            if line.strip().startswith('- [ ]'):
                criteria.append(line.strip())

    return criteria if criteria else ["Issue resolved successfully"]

def create_feature_branch(issue_number: int, title: str) -> str:
    """Create feature branch from issue title."""
    branch_name = re.sub(r'[^a-z0-9\s-]', '', title.lower())
    branch_name = re.sub(r'\s+', '-', branch_name)[:40]
    branch_name = f"feature/{issue_number}-{branch_name}"

    run_cmd(f"git checkout -b {branch_name}")
    return branch_name

def ralph_loop_planning(issue: dict, conversation: list) -> tuple:
    """
    Planning phase: Create and refine implementation plan.
    Returns: (plan_text, updated_conversation, success)
    """
    title = issue['title']
    body = issue['body']
    criteria = extract_acceptance_criteria(body)

    system_prompt = """You are an expert software engineer planning the implementation of a GitHub issue.
Your task is to create a detailed implementation plan that:
1. Understands the issue requirements and acceptance criteria
2. Breaks down the solution into concrete steps
3. Identifies potential blockers or dependencies
4. Proposes a clear implementation strategy

Be thorough but practical. The plan will guide implementation."""

    print(f"\n📋 PLANNING PHASE (max {PLANNING_ITERATIONS} iterations)")
    print("=" * 60)

    plan_text = ""

    for iteration in range(1, PLANNING_ITERATIONS + 1):
        if iteration == 1:
            user_message = f"""Create an implementation plan for GitHub issue #{issue['number']}: {title}

## Issue Description:
{body}

## Acceptance Criteria:
{json.dumps(criteria, indent=2)}

Create a detailed plan that breaks down the solution into concrete steps."""
        else:
            user_message = f"""Review and refine the plan. Consider:
- Are all acceptance criteria addressed?
- Are there any blockers or dependencies?
- Is the implementation strategy clear?
- Can it be improved further?

Provide iteration {iteration} of the plan."""

        conversation.append({"role": "user", "content": user_message})

        response = client.messages.create(
            model="claude-opus-5",
            max_tokens=4000,
            system=system_prompt,
            messages=conversation
        )

        plan_text = response.content[0].text
        conversation.append({"role": "assistant", "content": plan_text})

        print(f"\n🔄 Iteration {iteration}/{PLANNING_ITERATIONS}")
        print(plan_text[:500] + "..." if len(plan_text) > 500 else plan_text)

    print(f"\n✅ Planning phase complete")
    return plan_text, conversation, True

def ralph_loop_implementation(issue: dict, plan: str, conversation: list) -> tuple:
    """
    Implementation phase: Execute plan and solve issue.
    Returns: (success, conversation, implementation_log)
    """
    title = issue['title']
    criteria = extract_acceptance_criteria(issue['body'])

    system_prompt = """You are an expert software engineer implementing a GitHub issue.
You have a detailed implementation plan. Your task is to:
1. Follow the plan step by step
2. Implement the required changes
3. Ensure all acceptance criteria are met
4. Create clear git commits
5. Document your progress

Execute the plan efficiently. If you encounter blockers, document them clearly."""

    print(f"\n💻 IMPLEMENTATION PHASE (max {IMPLEMENTATION_ITERATIONS} iterations)")
    print("=" * 60)

    implementation_log = []
    success = False

    for iteration in range(1, IMPLEMENTATION_ITERATIONS + 1):
        if iteration == 1:
            user_message = f"""Implement the solution based on the plan.

## Acceptance Criteria:
{json.dumps(criteria, indent=2)}

## Implementation Plan:
{plan}

Start implementing. Make code changes, commit, and report progress."""
        else:
            user_message = f"""Continue implementation. Iteration {iteration}/{IMPLEMENTATION_ITERATIONS}.

Current status:
- What has been completed?
- What remains?
- Are there blockers?
- Next steps?

Continue working towards meeting all acceptance criteria."""

        conversation.append({"role": "user", "content": user_message})

        response = client.messages.create(
            model="claude-opus-5",
            max_tokens=4000,
            system=system_prompt,
            messages=conversation
        )

        impl_text = response.content[0].text
        conversation.append({"role": "assistant", "content": impl_text})
        implementation_log.append({
            "iteration": iteration,
            "output": impl_text
        })

        print(f"\n🔄 Iteration {iteration}/{IMPLEMENTATION_ITERATIONS}")
        print(impl_text[:500] + "..." if len(impl_text) > 500 else impl_text)

        # Check for completion signal
        if "complete" in impl_text.lower() and "all acceptance criteria" in impl_text.lower():
            success = True
            break

    return success, conversation, implementation_log

def create_pr_with_failure_docs(issue: dict, plan: str, impl_log: list, success: bool, branch: str) -> str:
    """
    Create PR with comprehensive documentation of process and any failures.
    """
    criteria = extract_acceptance_criteria(issue['body'])

    # Build failure documentation
    failure_doc = ""
    if not success:
        failure_doc = f"""
## ❌ Implementation Status: Incomplete

### Ralph Loop Iterations Exhausted
The agent reached the maximum iterations ({IMPLEMENTATION_ITERATIONS}) without completing all acceptance criteria.

### What Was Attempted
"""
        for log_entry in impl_log:
            failure_doc += f"""
#### Iteration {log_entry['iteration']}
```
{log_entry['output'][:1000]}
```
"""

        failure_doc += f"""

### Acceptance Criteria Status
"""
        for criterion in criteria:
            failure_doc += f"- ❌ {criterion}\n"

        failure_doc += f"""

### Recommendations for Manual Resolution
1. Review the implementation attempts above
2. Identify the primary blocker
3. Consider alternative approaches
4. Check for missing dependencies or resources
5. Review code snippets in implementation history

### For Next Steps
- Manually implement based on agent's attempts
- Or modify issue scope and re-run with `ready-for-agent` label
- Or reach out to team for guidance on blockers
"""

    pr_body = f"""## Issue #{issue['number']}: {issue['title']}

### Status
{'✅ COMPLETED' if success else '❌ INCOMPLETE - See below for details'}

### Summary
This PR {'completes' if success else 'partially addresses'} the issue through the ralph loop process:
1. **Planning Phase:** {PLANNING_ITERATIONS} iterations to create implementation plan
2. **Implementation Phase:** {IMPLEMENTATION_ITERATIONS} iterations to solve the issue

### Implementation Plan
```
{plan}
```

{failure_doc}

### Ralph Loop Execution
- Planning iterations: {PLANNING_ITERATIONS}
- Implementation iterations: {len(impl_log)}
- Total tokens used: ~50k (estimate)
- Time: {datetime.now().isoformat()}

### Process Details
This PR was generated by the Autonomous Ralph Loop Agent running in GitHub Actions.
For questions or issues, see: `agents/docs/RALPH_LOOP.md`

---
*Generated with Claude Autonomous Ralph Loop Agent*
"""

    # Check git changes
    status = run_cmd("git status --short", check=False)

    if status:
        # Commit changes
        commit_msg = f"Issue #{issue['number']}: {issue['title']}\n\nGenerated by Ralph Loop Agent"
        run_cmd(f'git add -A && git commit -m "{commit_msg}"')

    # Create PR
    cmd = f'''gh pr create --repo {GITHUB_REPOSITORY} --title "Issue #{issue['number']}: {issue['title']}" --body "{pr_body}" --head {branch}'''
    output = run_cmd(cmd)

    pr_url = output.split('\n')[-1] if output else "PR created"
    print(f"\n✅ PR created: {pr_url}")

    return pr_url

def main():
    """Main ralph loop orchestration."""
    print("🤖 Ralph Loop Agent - GitHub Actions")
    print("=" * 60)
    print(f"Repository: {GITHUB_REPOSITORY}")
    print(f"Issue: #{ISSUE_NUMBER}")
    print(f"Started: {datetime.now().isoformat()}\n")

    # Verify credentials
    if not GITHUB_TOKEN or not ANTHROPIC_API_KEY:
        print("❌ Missing credentials (GITHUB_TOKEN or ANTHROPIC_API_KEY)")
        sys.exit(1)

    if not ISSUE_NUMBER:
        print("❌ Missing GITHUB_ISSUE_NUMBER environment variable")
        sys.exit(1)

    # Verify git repo
    if not os.path.exists(".git"):
        print("❌ Not in a git repository")
        sys.exit(1)

    try:
        # Get issue details
        print(f"📖 Fetching issue #{ISSUE_NUMBER}...")
        issue = get_issue_details()
        print(f"   Title: {issue['title']}\n")

        # Create feature branch
        branch = create_feature_branch(issue['number'], issue['title'])

        # Start conversation with Claude
        conversation = []

        # PLANNING PHASE
        plan, conversation, _ = ralph_loop_planning(issue, conversation)

        # IMPLEMENTATION PHASE
        success, conversation, impl_log = ralph_loop_implementation(issue, plan, conversation)

        # CREATE PR
        print(f"\n📤 Creating pull request...")
        pr_url = create_pr_with_failure_docs(issue, plan, impl_log, success, branch)

        print(f"\n{'✅ SUCCESS' if success else '⚠️  INCOMPLETE'}")
        print(f"PR URL: {pr_url}")

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
