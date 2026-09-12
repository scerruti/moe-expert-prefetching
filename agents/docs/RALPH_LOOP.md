# Ralph Loop Agent - GitHub Actions Workflow

The **Ralph Loop Agent** implements the "ralph loop" pattern for autonomous issue resolution in GitHub Actions.

Named after Ralph Wiggum, a ralph loop is an iterative refinement process where the agent plans, reflects, and iterates to improve outcomes.

## Quick Start

### 1. Set Up GitHub Secrets

Store your credentials as GitHub Secrets:

```bash
gh secret set GITHUB_TOKEN --body "ghp_xxxxxxxxxxxxx"
gh secret set ANTHROPIC_API_KEY --body "sk-ant-xxxxxxxxxxxxx"
```

**GitHub UI Alternative:**
- Go to: Settings → Secrets and variables → Actions
- Create secrets: `GITHUB_TOKEN`, `ANTHROPIC_API_KEY`

### 2. Create an Issue

Create a GitHub issue with clear requirements:

```markdown
# Phase 1: Set up Python environment and dependencies

## Description
Set up the Python environment for Phase 1 data collection...

## Acceptance Criteria
- [ ] Create requirements.txt with all dependencies
- [ ] Test Python environment (3.10+ required)
- [ ] Verify GPU access (CUDA, torch install)
```

### 3. Label It `ready-for-agent`

Add the `ready-for-agent` label to the issue.

### 4. Watch the Agent Work

GitHub Actions automatically triggers:
1. Agent checks out repo
2. **Planning phase** (3 iterations): Creates implementation plan
3. **Implementation phase** (7 iterations): Solves the issue
4. **Auto-submits PR** with full documentation
5. Requires 2 human approvals to merge

---

## The Ralph Loop Pattern

### Phase 1: Planning (3 iterations)

**Goal:** Create a solid implementation plan

```
Iteration 1: Initial Plan
  ↓ (Feedback: Is it clear? Does it address all criteria?)
Iteration 2: Refined Plan
  ↓ (Feedback: Any blockers? Missing dependencies?)
Iteration 3: Final Plan
  ↓
Implementation Ready
```

Each iteration builds on the previous, refining the approach.

### Phase 2: Implementation (7 iterations)

**Goal:** Execute the plan and meet acceptance criteria

```
Iteration 1: Start Implementation
  ↓ (Status: What's done? What remains?)
Iteration 2-6: Continue Implementation
  ↓ (Status: Progress toward criteria)
Iteration 7: Final Push
  ↓
PR Ready (Success or Comprehensive Failure Docs)
```

---

## Workflow Execution

### Automatic Trigger

```yaml
When:
  - Issue is created/edited
  - Label "ready-for-agent" is added

Then:
  - GitHub Actions runs ralph_loop_agent.py
  - Agent creates feature branch
  - Agent executes ralph loop
  - Agent submits PR automatically
```

### What Happens Next

1. **PR Created** with:
   - Implementation plan (from planning phase)
   - All changes (from implementation phase)
   - Failure documentation (if incomplete)

2. **Branch Protection Kicks In**:
   - Requires 2 human approvals
   - Cannot merge without review

3. **Human Review**:
   - You (and team) review the PR
   - Approve if satisfied, or request changes
   - Once 2 approvals: Merge to main

---

## Failure Handling

If the agent can't complete all acceptance criteria, the PR includes:

### Comprehensive Failure Documentation

```markdown
## ❌ Implementation Status: Incomplete

### Ralph Loop Iterations Exhausted
The agent reached the maximum iterations (7) without completing all criteria.

### What Was Attempted
- Iteration 1: Tried approach X
- Iteration 2: Encountered blocker Y
- Iteration 3: Adjusted strategy
- ... (full trace of attempts)

### Acceptance Criteria Status
- ❌ Criteria 1: Reason for failure
- ✅ Criteria 2: Successfully completed
- ⚠️  Criteria 3: Partial implementation

### Recommendations for Manual Resolution
1. Review the implementation attempts
2. Identify the primary blocker
3. Consider alternative approaches
4. Check for missing dependencies
```

### Why Incomplete PRs Are Valuable

Instead of giving up, the agent creates a PR that:
- Shows the current code state
- Documents all attempts
- Identifies blockers
- Provides starting point for humans

This gives the team a **head start** on manual resolution rather than starting from zero.

---

## Configuration

### Iteration Limits

In `ralph_loop_agent.py`:

```python
PLANNING_ITERATIONS = 3      # Planning loops
IMPLEMENTATION_ITERATIONS = 7  # Implementation loops
```

**Industry standards:**
- Planning: 3-5 is typical (more causes analysis paralysis)
- Implementation: 7-10 is typical (balance quality vs. cost)

### Token Budget

With your $20 API budget:
- ~$0.05-0.15 per issue
- ~100+ issues supported
- Each issue: ~15k-50k tokens

---

## Monitoring & Troubleshooting

### Check Agent Progress

Go to: **Actions** tab → **Ralph Loop Agent** → Latest run

You'll see:
- ✅ Planning phase output
- ✅ Implementation phase iterations
- ✅ PR creation result

### Common Issues

**"Workflow didn't trigger"**
- Verify label is exactly `ready-for-agent`
- Check GitHub Secrets are set
- Try removing and re-adding the label

**"Agent times out"**
- GitHub Actions has 6-hour limit per job
- Most issues complete in <30 min
- If timeout occurs, issue is too complex for ralph loop

**"PR says incomplete"**
- This is expected sometimes
- Review the failure documentation
- Consider splitting into smaller issues
- Or provide more context in issue description

### Manual Debug

To test locally:

```bash
cd /Users/scerruti/moe
export GITHUB_TOKEN="..."
export ANTHROPIC_API_KEY="..."
export GITHUB_ISSUE_NUMBER=1
export GITHUB_REPOSITORY="scerruti/moe-expert-prefetching"

python agents/ralph_loop_agent.py
```

---

## Best Practices

### Writing Good Issues for Ralph Loop

✅ **DO:**
- Clear title and description
- Specific acceptance criteria (use checkboxes)
- Include context (why this matters)
- Link related issues

❌ **DON'T:**
- Vague requirements ("fix something")
- No acceptance criteria
- Overly complex issues
- Mix multiple concerns in one issue

### Ideal Issue Scope

- **Too Small:** "Add a comment to line 42" (agent overhead)
- **Too Large:** "Rewrite entire backend" (exceeds iterations)
- **Just Right:** "Implement data collection pipeline" (2-5k tokens)

### Review Quality

Even though PR is auto-submitted, **human review is critical:**
1. Read the implementation plan
2. Check code quality
3. Verify acceptance criteria met
4. Approve if satisfied

---

## Workflow File Reference

See `.github/workflows/ralph-loop-agent.yml` for the full workflow definition.

**Key steps:**
1. Checkout repo
2. Set up Python + dependencies
3. Run `ralph_loop_agent.py`
4. Agent handles everything else

---

## Examples

### Example 1: Successful Issue

```
Issue: "Phase 1: Environment Setup"
  ↓
Agent labels with "ready-for-agent"
  ↓
Ralph Loop:
  - Planning: Creates clear plan in 2 iterations
  - Implementation: Completes in 4 iterations
  ↓
PR: ✅ All acceptance criteria met
  ↓
Team approves → Merge
```

### Example 2: Incomplete Issue

```
Issue: "Complex feature requiring external API"
  ↓
Label with "ready-for-agent"
  ↓
Ralph Loop:
  - Planning: Creates plan, identifies blocker (need API key)
  - Implementation: Reaches 7 iterations, can't proceed
  ↓
PR: ⚠️ Incomplete, but documents:
     - What was attempted
     - Why it failed (missing API key)
     - How to fix it
  ↓
Team reviews, gets head start on fix
  ↓
Manual completion + Merge
```

---

## Next Steps

### To Enable Ralph Loop

1. ✅ Set up GitHub Secrets (GITHUB_TOKEN, ANTHROPIC_API_KEY)
2. ✅ Verify workflow is in `.github/workflows/ralph-loop-agent.yml`
3. ✅ Create an issue with clear acceptance criteria
4. ✅ Add `ready-for-agent` label

### To Use Ralph Loop

1. Create issue with acceptance criteria
2. Label with `ready-for-agent`
3. Watch Actions tab
4. Review PR when ready
5. Approve + Merge

---

## Support

For questions:
- Check `agents/README.md` for agent overview
- Review GitHub Actions logs
- Check issue description for blocker details
- Read failure PR documentation

---

**Last Updated:** 2026-09-12
