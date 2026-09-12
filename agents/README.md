# Autonomous Agents

Claude-powered agents for autonomous GitHub issue resolution across all project phases.

## Available Agents

### 1. Autonomous Agent (Local)
**File:** `autonomous_agent.py`  
**Setup:** `docs/SETUP.md`

**Use case:** Run locally with interactive feedback loop during development.

```bash
export GITHUB_TOKEN="..."
export ANTHROPIC_API_KEY="..."
python agents/autonomous_agent.py --phase 1
```

**Features:**
- Works across all phases (1-5)
- Custom label filtering
- Interactive feedback during implementation
- Manual PR submission

**Workflow:**
```
Pick Issue → Create Branch → Implement → You Review → PR
```

---

### 2. Ralph Loop Agent (GitHub Actions)
**File:** `ralph_loop_agent.py`  
**Setup:** `docs/RALPH_LOOP.md`  
**Workflow:** `.github/workflows/ralph-loop-agent.yml`

**Use case:** Runs automatically when issue labeled `ready-for-agent`.

**Features:**
- Triggered by GitHub issue label
- Planning phase (3 iterations) - creates implementation plan
- Implementation phase (7 iterations) - solves the issue
- Comprehensive failure documentation
- Auto-submits PR (no manual submission needed)

**Workflow:**
```
Issue labeled "ready-for-agent"
  ↓
Ralph Loop (Plan → Implement)
  ├─ Planning: 3 iterations
  ├─ Implementation: 7 iterations
  └─ Document progress/failures
  ↓
Auto-submit PR
  ↓
Await 2 approvals → Merge
```

---

## Quick Start

### Local Agent (Interactive)
```bash
cd /Users/scerruti/moe
source .env  # Load tokens (create .env with your credentials)
python agents/autonomous_agent.py --phase 1
```

See: `agents/docs/SETUP.md`

### Ralph Loop Agent (Automated)
```bash
# 1. Create issue in GitHub
# 2. Add label: "ready-for-agent"
# 3. GitHub Actions automatically triggers
# 4. Check Actions tab for progress
# 5. Review PR when ready
```

See: `agents/docs/RALPH_LOOP.md`

---

## Token Management

Both agents require:
- `GITHUB_TOKEN` - GitHub API access
- `ANTHROPIC_API_KEY` - Claude API access

**Local agent:** Store in `.env` file (add to .gitignore)  
**Ralph Loop agent:** Store as GitHub Secrets via:
```bash
gh secret set GITHUB_TOKEN --body "ghp_..."
gh secret set ANTHROPIC_API_KEY --body "sk-ant-..."
```

---

## Cost & Limits

**Per Issue Cost:** $0.02 - $0.15  
**Your Budget:** $20 (API credits)  
**Estimated Capacity:** 100+ issues

**Ralph Loop Iterations:**
- Planning: max 3 loops
- Implementation: max 7 loops
- Total: ~15k-50k tokens per issue

---

## Architecture

```
agents/
├── README.md                    # This file
├── autonomous_agent.py          # Local interactive agent
├── ralph_loop_agent.py          # GitHub Actions agent
└── docs/
    ├── SETUP.md                 # Local agent setup
    └── RALPH_LOOP.md            # Ralph loop workflow

.github/workflows/
└── ralph-loop-agent.yml         # GitHub Actions trigger
```

---

## Roadmap

### Current
- ✅ Local autonomous agent
- ✅ Ralph loop agent (planned)
- ✅ GitHub Actions workflow

### Planned (Issue #11)
- 🔄 PR feedback monitoring
- 🔄 Iterative feedback response
- 🔄 Context preservation across reviews

### Future
- 📋 Multiple concurrent agents
- 📋 Agent coordinator
- 📋 Custom agent templates

---

## Documentation

- **SETUP.md** - How to set up and run local agent
- **RALPH_LOOP.md** - Ralph loop pattern, workflow, failure modes

---

## Support

For issues or questions:
1. Check the relevant documentation in `docs/`
2. Review GitHub Actions logs (if using Ralph Loop agent)
3. Check agent output for detailed error messages
4. Create an issue in the repo with details

---

**Last Updated:** 2026-09-12
