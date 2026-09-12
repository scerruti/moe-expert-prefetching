# Phase 1 GitHub Issues & Work Tracking

**Last Updated:** 2026-09-12  
**Status:** 10 issues created with dependency tracking

---

## Quick Navigation

All Phase 1 issues are tracked on GitHub with proper labeling and blocking relationships.  
**View all issues:** https://github.com/scerruti/moe-expert-prefetching/issues

---

## Issue Dependency Map

```
No Blockers (Can Start Immediately)
├── #1: Environment Setup ✅
│   │
│   ├── #2: Dataset Loading - GSM8K
│   ├── #3: Dataset Loading - MBPP
│   └── #4: Model Loading
│
├── #9: Logging (parallel track)
└── #10: Documentation (parallel track)

Sequential Path (Critical Path)
  #1 → #4 → #5 → #6 → #7 → #8
       ↑    ↑
       └─#2─┘
       └─#3─┘
```

---

## Issues by Priority & Status

### 🚨 CRITICAL PRIORITY (Blocks Other Work)

#### #1: Environment Setup
- **Status:** Ready to start (no blockers)
- **Assignee:** [Available]
- **Estimated Duration:** 1-2 days
- **Labels:** `phase-1/environment`, `type/implementation`, `priority/critical`, `blocker/none`
- **Description:** Set up Python environment, create requirements.txt, verify GPU access
- **Link:** https://github.com/scerruti/moe-expert-prefetching/issues/1

#### #2: Dataset Loading - GSM8K
- **Status:** Blocked by #1
- **Assignee:** [Available]
- **Estimated Duration:** 2 days
- **Labels:** `phase-1/datasets`, `type/implementation`, `priority/critical`, `blocker/depends-on`
- **Description:** Load and prepare 8.5k GSM8K examples
- **Link:** https://github.com/scerruti/moe-expert-prefetching/issues/2

#### #3: Dataset Loading - MBPP
- **Status:** Blocked by #1
- **Assignee:** [Available]
- **Estimated Duration:** 1-2 days
- **Labels:** `phase-1/datasets`, `type/implementation`, `priority/critical`, `blocker/depends-on`
- **Description:** Load and prepare 974 MBPP examples
- **Link:** https://github.com/scerruti/moe-expert-prefetching/issues/3

#### #4: Model Loading and Setup
- **Status:** Blocked by #1
- **Assignee:** [Available]
- **Estimated Duration:** 1-2 days
- **Labels:** `phase-1/model`, `type/implementation`, `priority/critical`, `blocker/depends-on`
- **Description:** Load Mixtral 8x7B, set up tokenizer, verify determinism
- **Link:** https://github.com/scerruti/moe-expert-prefetching/issues/4

#### #5: Forward Hook Implementation
- **Status:** Blocked by #4
- **Assignee:** [Available]
- **Estimated Duration:** 2-3 days
- **Labels:** `phase-1/hooks`, `type/implementation`, `priority/critical`, `blocker/depends-on`
- **Description:** Register hooks on MoE router layers, capture routing probabilities
- **Link:** https://github.com/scerruti/moe-expert-prefetching/issues/5

#### #6: Data Collection Pipeline
- **Status:** Blocked by #2, #3, #5
- **Assignee:** [Available]
- **Estimated Duration:** 3-4 days
- **Labels:** `phase-1/collection`, `type/implementation`, `priority/critical`, `blocker/depends-on`
- **Description:** Main loop: 5 runs × 2 datasets × 9.5k prompts
- **Link:** https://github.com/scerruti/moe-expert-prefetching/issues/6

#### #7: Parquet Storage & Serialization
- **Status:** Blocked by #6
- **Assignee:** [Available]
- **Estimated Duration:** 2-3 days
- **Labels:** `phase-1/storage`, `type/implementation`, `priority/critical`, `blocker/depends-on`
- **Description:** Implement PyArrow schema, batch writing, compression
- **Link:** https://github.com/scerruti/moe-expert-prefetching/issues/7

#### #8: Validation Testing
- **Status:** Blocked by #7
- **Assignee:** [Available]
- **Estimated Duration:** 2-3 days
- **Labels:** `phase-1/validation`, `type/testing`, `priority/critical`, `blocker/depends-on`
- **Description:** Determinism checks, randomization consistency, data quality
- **Link:** https://github.com/scerruti/moe-expert-prefetching/issues/8

### 📊 HIGH PRIORITY (Support Tasks)

#### #9: Logging and Telemetry
- **Status:** Can start anytime (parallel track)
- **Assignee:** [Available]
- **Estimated Duration:** 1-2 days
- **Labels:** `phase-1/logging`, `type/implementation`, `priority/high`, `blocker/none`
- **Description:** Resource tracking, structured logging, metadata JSON
- **Link:** https://github.com/scerruti/moe-expert-prefetching/issues/9

#### #10: Documentation
- **Status:** Can start anytime (parallel track)
- **Assignee:** [Available]
- **Estimated Duration:** 1-2 days
- **Labels:** `phase-1/docs`, `type/documentation`, `priority/high`, `blocker/none`
- **Description:** README, schema documentation, troubleshooting guide
- **Link:** https://github.com/scerruti/moe-expert-prefetching/issues/10

---

## Labels Reference

### Component Labels (Blue)
- `phase-1/environment` – Environment setup
- `phase-1/datasets` – Dataset loading
- `phase-1/model` – Model and tokenizer
- `phase-1/hooks` – Hook implementation
- `phase-1/collection` – Data collection
- `phase-1/storage` – Parquet storage
- `phase-1/validation` – Validation testing
- `phase-1/logging` – Logging/telemetry
- `phase-1/docs` – Documentation

### Type Labels (Purple)
- `type/implementation` – Code implementation
- `type/testing` – Testing/validation
- `type/documentation` – Documentation

### Priority Labels (Orange/Red)
- `priority/critical` – Blocks other work (red)
- `priority/high` – Should be done soon (orange)
- `priority/medium` – Normal priority (yellow)
- `priority/low` – Nice to have (light yellow)

### Blocker Labels (Green/Red)
- `blocker/none` – No blockers, ready to start (green)
- `blocker/depends-on` – Has dependencies (yellow)
- `blocker/blocked` – Currently blocked (red)

---

## How to Pick Up Work

### Step 1: Find Work
1. Go to GitHub Issues: https://github.com/scerruti/moe-expert-prefetching/issues
2. Filter by `blocker/none` to see work ready to start
3. Pick an unassigned issue

### Step 2: Assign Yourself
```bash
gh issue edit <issue-number> --add-assignee @me
```

### Step 3: Create a Branch
```bash
git checkout -b feature/phase-1-<component-name>
# Example: feature/phase-1-environment-setup
```

### Step 4: Work & Commit
- Check off tasks in the issue as you complete them
- Commit frequently with clear messages
- Link commits to the issue:
  ```bash
  git commit -m "Implement requirement.txt creation - fixes #1"
  ```

### Step 5: Create a PR
When done, create a pull request linking to the issue:
```bash
gh pr create --title "Phase 1: [Component Name]" \
  --body "Closes #<issue-number>" \
  --label "phase-1/<component>"
```

### Step 6: PR Review
- Address any review comments
- Ensure all checks pass
- Merge when approved

---

## Current Status

| Issue | Status | Assignee | % Complete |
|-------|--------|----------|-----------|
| #1 Environment Setup | Not Started | — | 0% |
| #2 Dataset Loading (GSM8K) | Blocked | — | 0% |
| #3 Dataset Loading (MBPP) | Blocked | — | 0% |
| #4 Model Loading | Blocked | — | 0% |
| #5 Hook Implementation | Blocked | — | 0% |
| #6 Data Collection | Blocked | — | 0% |
| #7 Parquet Storage | Blocked | — | 0% |
| #8 Validation Testing | Blocked | — | 0% |
| #9 Logging & Telemetry | Not Started | — | 0% |
| #10 Documentation | Not Started | — | 0% |

**Overall Progress:** 0/10 issues started

---

## Recommended Work Order

For a **single developer**, work on issues in this sequence:
1. **#1** (Environment) – 1-2 days
2. **#4** (Model Loading) – 1-2 days (parallel: start #2 or #3)
3. **#2 & #3** (Datasets) – 3-4 days combined (can parallelize)
4. **#5** (Hooks) – 2-3 days
5. **#6** (Collection) – 3-4 days
6. **#7** (Storage) – 2-3 days
7. **#8** (Validation) – 2-3 days
8. **#9 & #10** (Logging & Docs) – 2-3 days combined

**Total: ~3-4 weeks**

For a **team**, parallelize issues without blockers:
- Person A: #1 → #4 → #5 → #6 → #7 → #8 (critical path)
- Person B: #2 (dataset loading)
- Person C: #3 (dataset loading)
- Person D: #9 & #10 (logging & docs, can run anytime)

---

## FAQ

### Q: An issue says "Blocked by #X" – can I still start?
**A:** No, wait for that issue to complete. However, you can review the implementation or prepare code offline.

### Q: What if I find a sub-issue not listed?
**A:** Add it as a checkbox in the existing issue. If it's significant enough for separate tracking, comment on the issue and we can create a new one.

### Q: How do I know if my work is done?
**A:** Check all the boxes in the "Acceptance Criteria" section. If they're all checked ✅, you're done!

### Q: What if I get blocked?
**A:** Comment on the GitHub issue with the blocker, mention the maintainer, and pick up a different issue if possible.

### Q: Can I work on issues in a different order?
**A:** Only if they don't have blocking dependencies. Check the "Blocked by" section in the issue description.

---

## Support & Questions

- **Issues:** Comment on the GitHub issue
- **Slack/Email:** Tag the project lead
- **Documentation:** See phase_1/docs/ folder

---

**Good luck! 🚀 Pick up an issue and get started!**

