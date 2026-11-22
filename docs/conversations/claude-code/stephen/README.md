# Stephen's Claude Code sessions

How Stephen used Claude Code to do this project's work, one file per session per day. See [`../README.md`](../README.md) for how the files are made and what's redacted.

Sessions before 18 October 2026 aren't here; their transcripts were no longer on Stephen's Mac when this record started, most likely removed by Claude Code's 30-day cleanup.

## Sessions

Summaries are written by Claude from the transcripts and reviewed by Stephen. Each names the main work, then how AI was used, including where it went wrong.

### 2026-10-18: [status and project analysis](2026-10-18_claude-code_54acddb0.md)

Repo status, an in-depth review of the project against its open issues, and unblocking #14. A short check-in session.

### 2026-10-28: [Phase 1 status and the first approvals](2026-10-28_claude-code_54acddb0.md)

Reviewing Waleed's PR and merging #56. Stephen couldn't approve his own PR, so Claude added a note that he was merging without approval, and a review was requested from Dvora, and Claude drafted a backup email.

### 2026-10-29: [stacking the Phase 1 branches, and a timesheet](2026-10-29_claude-code_54acddb0.md)

Building #15 on #14, starting #8, combining the acceptance tests into one regression test (#70), and a status document (#71). Claude estimated hours from commit history for a timesheet; Stephen rejected the estimate ("There is no frickin way I worked that many hours") and switched to a simpler per-day method.

### 2026-10-30: [the first full collection run](2026-10-30_claude-code_54acddb0.md)

The longest session. The Phase 1 notebook failed repeatedly on Colab: class names, dependencies, memory, and an account mix-up over Drive. Stephen pushed back on downgrading dependencies ("Seems like a hack"; "Quality here is more important"), stopped Claude guessing at fixes ("Stop guessing"), and slowed it down when it moved on without recording problems ("Whoa nelly! Too fast. We need to document issues"). A bug that had passed acceptance testing led to a lasting rule: **prove acceptance criteria with tests, don't just state them**. Also: the decision to keep Phase 1 on one model (YAGNI), and early discussion of GPU testing in CI.

### 2026-10-30: [a second session on the same run](2026-10-30_claude-code_088b91b3.md)

Claude started fixing a problem someone else was already fixing; Stephen: "you need to follow the rules." Every issue was given a milestone, and the cleanup was done one item at a time.

### 2026-10-31: [GPU test runs and the cost of mistakes](2026-10-31_claude-code_54acddb0.md)

Fixing #90 and #91, sorting out which tests need a GPU, and making every bug-fix commit traceable to an issue. Confusion over 4-bit loading cost paid Colab runs ("WASTING MY TIME"; "Every run costs me money"), which produced two lasting rules: capture full output in one run with `tee`, and re-run only failing tests. Stephen asked "How did our rules fail us?", and the rules were updated.

### 2026-10-31: [data-quality findings and a step back](2026-10-31_claude-code_088b91b3.md)

A Phase 1 review found serious data-quality problems ("How did those make it through?"), filed as issues with tests. Decisions on fp16 (#103) and on holding out data, where Taylor's view was that it isn't needed because the prediction model is what's being tested. Midway, Stephen stepped back to reflect on how organised the project was and on slow reviews; that conversation led to code walkthrough meetings, the block on Phase 2 until the reorganisation (#112), and the proof-of-acceptance rule. Also the first trial collection and the batch-sensitivity write-up (#114). Two passages are redacted at Stephen's request: one about other team members' personal circumstances, and a personal, off-project use of the session to evaluate career options.

### 2026-11-01: [k-scaling and lookahead experiments](2026-11-01_claude-code_088b91b3.md)

Experiments on how many experts to prefetch, including a square-root fit and the maths behind it. Stephen asked Claude to push back on his proposed change to the write-up. Results posted on #114 and #110, and a meeting email proposing a code walkthrough. Opens with Stephen deleting the previous day's personal material (redacted).

### 2026-11-05: [Waleed's questions](2026-11-05_claude-code_088b91b3.md)

A document answering Waleed's questions and comments from email, a recap of the experiments, and finding a reading list.

### 2026-11-07: [merge order, an email, and admin](2026-11-07_claude-code_088b91b3.md)

Fixes for Waleed's review, working out the merge order (#111 first) and where GPU testing fits, and an email to Waleed and Dvora. Then the RET time log, and emails about the stipend, Colab reimbursement and the pizza seminars, with Stephen steering what to ask ("Don't ask about makeup").

### 2026-11-12: [merges and repair](2026-11-12_claude-code_088b91b3.md)

After Waleed's reviews, #111 and #115 merged; the stacked branches were repaired after a squash merge, and the repo switched to merge commits only. #119 opened. Also reporting hours, and using Claude in Chrome with Colab.

### 2026-11-19: [preparing to discuss #120](2026-11-19_claude-code_088b91b3.md)

Notes for the meeting about Waleed's large #120, including how to give feedback kindly. Stephen checked whether Claude's wording read as a dig at Waleed, and used Claude to understand why #120 might have been written, looking for a root cause that needed a wider fix. Claude's answer, from the evidence, was that #120 built on the earlier CI work, and that the best way to find out was to ask. *(Added 22 November: Stephen noted that in the meeting, Waleed explained he was adopting best practices from another project. See the [22 November record](2026-11-22_claude-code_d70ecd9f.md).)* Also the #58 and #59 problems, and questions from the pizza seminar.

### 2026-11-21: [reviews, splitting #120, and a fork](2026-11-21_claude-code_088b91b3.md)

A GPU run for #58, a fix for #59 silently dropping rows, and splitting #120 into #123, #124 and #125. Stephen found the first draft of #121's guidance for Dvora condescending, and it was rewritten. In the reimbursement request, Claude forwarded receipts Stephen hadn't approved ("I wouldn't have had you forward those emails"), which led to a rule: confirm every outgoing message. Designed the #126 pre-test, then forked the session.

### 2026-11-21: [course correction and the determinism decision](2026-11-21_claude-code_d70ecd9f.md)

The #126 results. Stephen recognised that a 30 October decision had diverged from the system design, and amended it (#127): Phase 1 finishes on Qwen, and Phase 2 makes the model swappable. Claude kept insisting on choosing a second model until Stephen corrected it. The determinism-check decision, with repeated corrections to Claude's writing ("Speak more plainly"; "I see word soup…"), became the decision-document PR, opened just as the session hit its usage limit (it's #128, per the next day's record). Claude kept naming only Waleed as reviewer until Stephen asked why.

### 2026-11-22: [separating two Claude accounts](2026-11-22_claude-code_599b2dbe.md)

Tool setup: keeping Stephen's personal and education Claude accounts apart with a hook, and cleaning up the workspace.

### 2026-11-22: [review responses and this record](2026-11-22_claude-code_d70ecd9f.md)

Fixing Waleed's #59 change request, and generalising his findings into #130 (hook cleanup) and #131 (a failure-behaviour standard). A thank-you email to the team. Revising #128 with Waleed's corrections, where Stephen pushed back twice: "We shouldn't be choosing a number" and "Where does this happen?", which exposed that the window had only been computed by hand. Then setting up this record. Stephen's direction: "I don't want the mess hidden."
