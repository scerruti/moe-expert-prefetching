# Autonomous Issue Agent Setup

This guide shows how to run the Claude-powered agent that autonomously works on any GitHub issues (across all phases).

## What the Agent Does

1. **Monitors** your GitHub issues (filters for `blocker/none` label)
2. **Picks up** the first unblocked issue
3. **Creates** a feature branch
4. **Calls Claude** to implement the solution
5. **Iterates** with you until acceptance criteria are met
6. **Commits** changes and **creates a PR** when done
7. **Loops** to the next issue

## Prerequisites

### 1. GitHub Token
Create a personal access token:
- Go to: https://github.com/settings/tokens
- Click "Generate new token (classic)"
- Select scopes: `repo`, `workflow`
- Copy the token

### 2. Claude API Key
Get your API key:
- Go to: https://console.anthropic.com/
- Click "Get API key"
- Copy the key

### 3. Python 3.8+
```bash
python --version  # Should be 3.8+
```

### 4. Dependencies
```bash
pip install anthropic
# Also ensure you have: git, gh (GitHub CLI)
```

## Installation

### Step 1: Set Environment Variables

```bash
export GITHUB_TOKEN="ghp_xxxxxxxxxxxxx"
export ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxxxxx"
```

**Or** create a `.env` file in the repo root:
```bash
cat > /Users/scerruti/moe/.env << 'EOF'
GITHUB_TOKEN="ghp_xxxxxxxxxxxxx"
ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxxxxx"
EOF

# Then load it
source .env
```

### Step 2: Make Script Executable

```bash
chmod +x /Users/scerruti/moe/autonomous_agent.py
```

### Step 3: Test Setup

```bash
cd /Users/scerruti/moe

# Verify you're in the right repo
git remote -v

# Test that GitHub CLI works
gh issue list --label "blocker/none"

# Test that Claude API key works
python -c "from anthropic import Anthropic; print('✅ Claude API OK')"
```

## Usage

### Run the Agent

**Work on Phase 1 unblocked issues:**
```bash
cd /Users/scerruti/moe
python autonomous_agent.py --phase 1
```

**Work on Phase 2 issues:**
```bash
python autonomous_agent.py --phase 2
```

**Work on any custom labels:**
```bash
python autonomous_agent.py --labels "good-first-issue"
```

**Work on specific component:**
```bash
python autonomous_agent.py --labels "phase-1/environment"
```

**Default (all unblocked issues):**
```bash
python autonomous_agent.py
```

### What Happens

1. **Agent starts**: Connects to GitHub and fetches unblocked issues
2. **Picks issue**: Shows available issues, picks the first one
3. **Creates branch**: Makes a feature branch (e.g., `feature/phase-1-1-environment-setup`)
4. **Claude works**: Claude reads the issue and starts implementing
5. **Iterative loop**: 
   - Claude makes changes
   - You review the output
   - You provide feedback or approve
   - Loop continues until done
6. **Submits PR**: When accepted, agent commits and creates a PR
7. **Next issue**: Asks if you want to work on the next issue

### Example Session

```
🤖 Autonomous Issue Agent Starting
📍 Repository: scerruti/moe-expert-prefetching
🏷️  Labels: phase-1/environment,phase-1/datasets,phase-1/model,blocker/none
⏰ Started: 2026-09-12T12:00:00

📋 Found 3 issue(s):
   1. #1: Phase 1: Set up Python environment and dependencies
   2. #9: Phase 1: Add logging and resource tracking
   3. #10: Phase 1: Write documentation and README

🎯 Working on Issue #1: Phase 1: Set up Python environment and dependencies

📦 Creating branch: feature/1-environment-setup

🤖 Claude is working on issue #1...

[Claude's response showing what it will do]

📝 Your feedback (or 'done' to submit PR): [You respond]

[Claude makes changes, commits, creates PR]

✅ PR created: https://github.com/scerruti/moe-expert-prefetching/pull/123

🔄 Work on next issue? (yes/no): yes
```

## Workflow

### For Each Issue:

**You:**
1. Let Claude start working
2. Review Claude's approach
3. Provide feedback (e.g., "add error handling for network failures")
4. Approve with "done" or "submit" when ready

**Claude (Agent):**
1. Reads GitHub issue
2. Understands acceptance criteria
3. Implements code/changes
4. Tests implementation
5. Creates git commits
6. Submits PR automatically

### Stopping Early

- **Pause agent**: `Ctrl+C` (you can resume later)
- **Skip issue**: Type `no` when asked about next steps
- **Abandon branch**: `git checkout main && git branch -D feature/...`

## Cost Estimate

Using Claude Opus 5:
- **Per issue:** $0.02 - $0.15
- **Your budget:** $20.00
- **Estimated capacity:** 100+ issues

Most issues should cost < $0.10 in API tokens.

## Troubleshooting

### "GITHUB_TOKEN not set"
```bash
export GITHUB_TOKEN="your-token-here"
# Verify:
echo $GITHUB_TOKEN
```

### "ANTHROPIC_API_KEY not set"
```bash
export ANTHROPIC_API_KEY="your-key-here"
# Verify:
echo $ANTHROPIC_API_KEY
```

### "Not in a git repository"
```bash
cd /Users/scerruti/moe
git status  # Should work
```

### "gh: command not found"
Install GitHub CLI:
```bash
# On Mac:
brew install gh

# Or download: https://cli.github.com/
```

### Agent gets stuck
- Press `Ctrl+C` to pause
- Check git status: `git status`
- Review Claude's output
- Try running again: `python phase_1_agent.py`

## Advanced: Local Testing

Test the agent without submitting PRs:

```bash
# Test on a branch
git checkout -b test-agent

# Run agent (works on Phase 1 issues)
python autonomous_agent.py --phase 1

# When done, see what was created
git log --oneline -5
git diff main

# Clean up
git checkout main
git branch -D test-agent
```

## Working Across Phases

The agent is phase-agnostic and can work on any phase:

```bash
# Phase 1
python autonomous_agent.py --phase 1

# Phase 2
python autonomous_agent.py --phase 2

# Phase 3
python autonomous_agent.py --phase 3

# Mix and match with custom labels
python autonomous_agent.py --labels "phase-2/model,phase-2/hooks"
```

## Next Steps

1. **Set up environment variables** (see Step 1 above)
2. **Test the setup** (see Step 3 above)
3. **Run the agent**: `python phase_1_agent.py`
4. **Watch it work!** 🚀

---

## FAQ

**Q: Can I run multiple agents simultaneously?**
A: Not recommended. They might conflict on branches/PRs. Run one at a time.

**Q: What if Claude makes a mistake?**
A: You review each step and can provide feedback. The agent will iterate.

**Q: How do I see what Claude did?**
A: Check `git log` and `git diff` to see all changes made.

**Q: Can I stop and resume?**
A: Yes! `Ctrl+C` pauses gracefully. Run the script again to resume.

**Q: What happens if my API token expires?**
A: You'll get an auth error. Update the token and restart.

---

**Ready?** Run one of:
```bash
python autonomous_agent.py --phase 1    # Work on Phase 1 issues
python autonomous_agent.py              # Work on any unblocked issues
```

🤖
