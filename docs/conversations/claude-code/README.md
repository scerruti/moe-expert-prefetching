# Claude Code session records

How team members used Claude Code, an AI coding assistant, to do this project's work. Each person who contributes has their own folder:

- [**Stephen**](stephen/README.md): 18 October 2026 onwards

The records document the process honestly, including the misunderstandings, mistakes, corrections and admin work, not just the results. Contributing is voluntary. Each person decides what of theirs is shared.

## How the records are made

[`tools/ai_log/sanitize.py`](../../../tools/ai_log/sanitize.py) generates each record from a Claude Code transcript. Nobody edits the records by hand.

- **Kept word for word:** the person's prompts and Claude's replies.
- **Shortened:** each action Claude took (running a command, editing a file, sending an email) becomes one line saying what it did. What the action returned is left out.
- **Left out:** pasted text (logs, emails), Claude's internal reasoning, and system messages.
- **Redacted automatically:** email addresses, card digits, phone numbers, secrets, local file paths, dollar amounts on lines about pay, and any names in the person's private list. Other dollar amounts, such as compute costs, are shown.
- **Nothing is removed by topic.** A turn or line is redacted only when the person asks, and the record says so in place, with the reason.
- **Forked sessions:** a fork's shared history appears only in the parent's record. The fork's record starts with a note saying where it came from.

**The records are never reformatted or corrected.** A formatter or linter would change what was said, for example by renumbering lists inside replies (see #125).

## Redaction rules, for everyone

- **Your own mess stays in.** Mistakes, frustration, admin work and false starts are what make the record useful.
- **Other people's private lives stay out.** Never publish a teammate's personal circumstances, such as health, exams or family. Redact them, giving the reason.
- **People outside the team are named only if they're public in the project.** Put anyone else in your private names list.
- **Every redaction is visible,** with its reason.
- **Teammates can ask** for anything about them to be redacted, in review or later.

## Adding your own Claude Code sessions

You need Python 3.9 or later; the script uses only the standard library.

1. **Save your transcripts.** Claude Code keeps them in `~/.claude/projects/<folder named after your checkout>/` and deletes them after 30 days, so copy them somewhere private first:

   ```bash
   mkdir -p ~/moe-ai-log/archive
   cp ~/.claude/projects/*moe*/*.jsonl ~/moe-ai-log/archive/
   ```

2. **Optionally, add private settings** in `~/moe-ai-log/`. These never go in the repo:
   - `private_terms.txt`: names to replace with `[private]`, one per line.
   - `pay_context.txt` and `flag_terms.txt`: extra words that mark a line as being about pay, or flag a turn for you to check.
   - `drops.json`: turns or lines you want redacted, each with the reason to show. The format is in `load_drops` in the script.

3. **Generate your records,** from the repo root, with your name in place of `Name`:

   ```bash
   python tools/ai_log/sanitize.py ~/moe-ai-log/archive/*.jsonl --since 2026-11-01 --until 2026-11-23 \
       --per-session --author Name --out docs/conversations/claude-code/name
   ```

   Each record's review report goes to `~/moe-ai-log/reviews/`, outside the repo, because it names your private terms.

4. **Review everything before you commit.** `python tools/ai_log/checklist.py ~/moe-ai-log/reviews` lists the turns flagged for a redaction check, but the automatic redaction can miss things, so read your records too. To redact something, add it to `drops.json` and run step 3 again.

5. **Add a `README.md` in your folder,** with a short summary of each day: the main work, then how AI was used, including where it went wrong. Then open a PR, with your teammates as reviewers.

## Other AI tools

The script reads only Claude Code transcripts. For ChatGPT, Gemini, Copilot or other tools, add a dated summary to [`docs/conversations/`](../README.md), in the format of the September files: the date, who was involved, the topic, and what was decided and why.
