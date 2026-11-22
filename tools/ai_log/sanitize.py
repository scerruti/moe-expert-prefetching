#!/usr/bin/env python3
"""Turn Claude Code session transcripts into sanitized Markdown records.

The records go in docs/conversations/claude-code/<person>/. They show how AI was
used to do the project's work, including the misunderstandings, mistakes and
corrections, so nothing is removed by topic.

Kept word for word:  the user's prompts and Claude's replies to the user.
Shortened:           each tool call becomes one line saying what it did.
Left out:            tool outputs, Claude's internal reasoning, system messages,
                     pasted content, compaction summaries, subagent chatter.
Redacted:            email addresses, card digits, phone numbers, secrets, local
                     paths, the names in private_terms.txt, and dollar amounts on
                     lines about personal pay. Other amounts are shown.

A turn or line is redacted only when the person asks, through drops.json, and the
record says so in place, with the reason.

Nothing here publishes anything. It writes a record (to --out) and a review
report (to --review-dir, outside the repo, because it names private terms); the
person reads both before anything goes into the repo.

Private settings live in a config folder outside the repo (--config-dir, or
$MOE_AI_LOG_DIR, default ~/moe-ai-log). Every file in it is optional:
  private_terms.txt  names to replace with [private], one per line
  pay_context.txt    extra words that mark a line as being about pay
  flag_terms.txt     extra words that flag a turn for a redaction check
  drops.json         redactions requested by the person (see load_drops)

Usage:
  sanitize.py ~/.claude/projects/<project>/*.jsonl --since 2026-11-16 --until 2026-11-23 \\
      --per-session --author Stephen --out docs/conversations/claude-code/stephen
"""

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PATH_END = r"[^\s)\]`'\"]*"

# Applied to every line of text that reaches a record, in this order.
PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+"), "[email]"),
    (re.compile(r"(?i)\b(mastercard|visa|amex|card)( ending( in)?)? ?\d{4}\b"), r"\1 [card]"),
    (re.compile(r"(?i)\bending( in)? \d{4}\b"), "ending [card]"),
    (re.compile(r"(?:[•*xX]{2,}\s?)+\d{4}\b"), "[card]"),
    (re.compile(r"\(?\b\d{3}\)?[-. ]\d{3}[-.]\d{4}\b"), "[phone]"),
    (re.compile(r"\b(ghp|gho|github_pat|sk|xox[bp])[-_][A-Za-z0-9_-]{10,}"), "[secret]"),
    (re.compile(r"/private/tmp/claude-\d+/" + PATH_END), "[scratchpad]"),
    (re.compile(r"(/Users|/home)/[^/\s]+/Documents/" + PATH_END), "[local document]"),
    (re.compile(r"(/Users|/home)/[^/\s]+/\.claude/" + PATH_END), "[claude config]"),
    (re.compile(r"~/Documents/" + PATH_END), "[local document]"),
]
HOME = re.compile(r"(/Users|/home)/[^/\s]+")

AMOUNT = re.compile(r"\$\s?\d[\d,]*(\.\d+)?")
PAY_WORDS = [r"stipend", r"salary", r"wages?", r"paycheck", r"compensation", r"honorarium",
             r"\bpaid me\b", r"payments? of"]
# Turns mentioning these are kept, but flagged so the person checks the redaction.
FLAG_WORDS = [r"reimburs", r"invoice", r"stipend", r"payment", r"receipt", r"time ?log",
              r"timesheet", r"personal account", r"card statement",
              r"Used Gmail", r"Used Google Sheets", r"Used Google Drive"]


def warn(message):
    print(f"warning: {message}", file=sys.stderr)


# ---------- private settings ----------

def read_lines(path):
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]


def load_drops(path):
    """drops.json is a list of redactions the person asked for. Each entry is either
    {"export": "<record file name without .md>", "turns": [first, last], "reason": "..."}
    or {"line_match": "<regex>", "reason": "..."}, which redacts every matching line.
    The reason is shown in the record."""
    if not path.exists():
        return []
    drops = json.loads(path.read_text())
    for d in drops:
        if "reason" not in d or not (("export" in d and "turns" in d) or "line_match" in d):
            raise ValueError(f"{path}: entry needs a reason and either export+turns or line_match: {d}")
        if "turns" in d and not (len(d["turns"]) == 2 and 1 <= d["turns"][0] <= d["turns"][1]):
            raise ValueError(f"{path}: turns must be [first, last] with first <= last: {d}")
    return drops


class Config:
    def __init__(self, config_dir, repo_paths=()):
        config_dir = Path(config_dir).expanduser()
        # longest first, so "Ann Smith" is replaced before "Ann"
        self.terms = sorted(read_lines(config_dir / "private_terms.txt"), key=len, reverse=True)
        self.pay_line = re.compile("(?i)" + "|".join(PAY_WORDS + [re.escape(w) for w in
                                   read_lines(config_dir / "pay_context.txt")]))
        self.flag = re.compile("(?i)" + "|".join(FLAG_WORDS + [re.escape(w) for w in
                               read_lines(config_dir / "flag_terms.txt")]))
        self.drops = load_drops(config_dir / "drops.json")
        self.line_drops = [(re.compile(d["line_match"]), d["reason"]) for d in self.drops if "line_match" in d]
        # paths inside the repo are shown relative to it
        self.repo_paths = sorted((str(Path(p).expanduser()).rstrip("/") + "/" for p in repo_paths),
                                 key=len, reverse=True)


# ---------- redaction ----------

class Redactor:
    def __init__(self, config):
        self.config = config
        self.counts = {}

    def count(self, what, n=1):
        if n:
            self.counts[what] = self.counts.get(what, 0) + n

    def __call__(self, text):
        lines = []
        for line in text.split("\n"):
            for rx, reason in self.config.line_drops:
                if rx.search(line):
                    line = f"*[Line redacted at the user's request: {reason}.]*"
                    self.count("line redacted: " + reason)
                    break
            if self.config.pay_line.search(line):
                line, n = AMOUNT.subn("$[amount]", line)
                self.count("pay amount", n)
            lines.append(line)
        text = "\n".join(lines)
        for prefix in self.config.repo_paths:
            text, n = re.subn(re.escape(prefix), "", text)
            self.count("repo path", n)
        for rx, repl in PATTERNS:
            text, n = rx.subn(repl, text)
            self.count(repl, n)
        text, n = HOME.subn("~", text)
        self.count("home path", n)
        for term in self.config.terms:
            text, n = re.subn(r"(?i)(?<!\w)" + re.escape(term) + r"(?!\w)", "[private]", text)
            self.count("private term", n)
        return text


# ---------- reading the transcript ----------

TAG_BLOCKS = [
    re.compile(r"<system-reminder>.*?</system-reminder>", re.S),
    re.compile(r"<ide_selection>.*?</ide_selection>", re.S),
    re.compile(r"<ide_opened_file>.*?</ide_opened_file>", re.S),
    re.compile(r"<command-[a-z-]+>.*?</command-[a-z-]+>", re.S),
    re.compile(r"<local-command-[a-z-]+>.*?</local-command-[a-z-]+>", re.S),
]
PASTE = re.compile(r"<pasted_content[^>]*>(.*?)</pasted_content[^>]*>", re.S)
NOT_TYPED = ("This session is being continued",  # compaction summary
             "<task-notification>",              # a background task finishing
             "Base directory for this skill:")   # a skill's instructions loading


def user_text(entry):
    """The prompt the user typed, or None if this entry isn't one."""
    if entry.get("isCompactSummary") or entry.get("isVisibleInTranscriptOnly"):
        return None
    content = entry.get("message", {}).get("content")
    if isinstance(content, str):
        parts = [content]
    elif isinstance(content, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return None
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
    else:
        return None
    text = PASTE.sub(lambda m: f"\n\n*[pasted content removed: {m.group(1).count(chr(10)) + 1} lines]*\n\n",
                     "\n".join(parts))
    for rx in TAG_BLOCKS:
        text = rx.sub("", text)
    text = text.strip()
    if not text or text.startswith(NOT_TYPED) or text in ("Tool loaded.", "[Request interrupted by user]"):
        return None
    return text


def tool_line(block):
    """One line saying what a tool call did, or None for internal plumbing."""
    name = block.get("name", "tool")
    inp = block.get("input", {}) or {}
    if name == "Bash":
        return f"Ran a command: {inp.get('description') or 'shell command'}"
    if name in ("Read", "Edit", "Write", "NotebookEdit"):
        verb = {"Read": "Read", "Edit": "Edited", "Write": "Wrote", "NotebookEdit": "Edited notebook"}[name]
        return f"{verb} `{inp.get('file_path') or inp.get('notebook_path') or 'a file'}`"
    if name in ("Grep", "Glob"):
        return f"Searched files ({name})"
    if name == "Agent":
        return f"Started a subagent: {inp.get('description', '')}"
    if name == "AskUserQuestion":
        return "Asked the user: " + " / ".join(q.get("question", "") for q in inp.get("questions", []))
    if name == "ToolSearch":
        return None
    if name.startswith("mcp__"):
        bits = name.split("__")
        return f"Used {bits[1].replace('claude_ai_', '').replace('_', ' ')}: {bits[-1].replace('_', ' ')}"
    if name in ("WebFetch", "WebSearch"):
        return f"{name}: {inp.get('url') or inp.get('query') or ''}"
    return f"Used {name}"


def was_denied(result_block):
    c = result_block.get("content")
    text = c if isinstance(c, str) else json.dumps(c)
    return "Permission for this action was denied" in text or "user doesn't want to proceed" in text


def load_entries(paths):
    """Main-conversation entries from every transcript, oldest first, each once.

    Returns (entries, forks). A forked session copies its parent's history with new
    ids but the same timestamps and content, so entries are matched on those; forks
    maps a fork's session id to (parent session id, time of the last shared entry).
    Pass parents before forks (oldest file first)."""
    seen, entries, forks, unreadable = {}, [], {}, 0
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    unreadable += 1
                    continue
                if e.get("type") not in ("user", "assistant") or e.get("isSidechain"):
                    continue
                key = (e.get("type"), e.get("timestamp"), json.dumps(e.get("message"), sort_keys=True))
                if key in seen:
                    parent, sid = seen[key], e.get("sessionId")
                    if parent != sid:
                        forks[sid] = (parent, max(forks.get(sid, (parent, ""))[1], e.get("timestamp", "")))
                    continue
                seen[key] = e.get("sessionId")
                entries.append(e)
    if unreadable:
        # a session still being written can end in a partial line; anything more is worth a look
        warn(f"{unreadable} unreadable transcript line(s) skipped")
    entries.sort(key=lambda e: e.get("timestamp", ""))
    return entries, forks


def local(ts):
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone()


def build_turns(entries, since, until):
    """Turns (a prompt plus everything Claude did until the next prompt) whose prompt
    falls on a local date in [since, until)."""
    turns, cur, denied = [], None, set()
    for e in entries:
        content = e.get("message", {}).get("content")
        if e["type"] == "user" and isinstance(content, list):
            denied |= {b.get("tool_use_id") for b in content
                       if isinstance(b, dict) and b.get("type") == "tool_result" and was_denied(b)}
        t = local(e["timestamp"])
        if e["type"] == "user":
            text = user_text(e)
            if text is not None:
                cur = {"time": t, "prompt": text, "items": []} if since <= t.date() < until else None
                if cur:
                    turns.append(cur)
            continue
        if cur is None:
            continue
        for b in content or []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text" and b.get("text", "").strip():
                cur["items"].append(["reply", b["text"].strip(), None])
            elif b.get("type") == "tool_use":
                line = tool_line(b)
                if line:
                    cur["items"].append(["tool", line, b.get("id")])
    for turn in turns:
        for item in turn["items"]:
            if item[0] == "tool" and item[2] in denied:
                item[1] += " — **blocked by permission check**"
    return turns


# ---------- writing the record ----------

def tool_block(tools):
    s = "s" if len(tools) > 1 else ""
    return [f"<details><summary>{len(tools)} tool call{s}</summary>", ""] + [f"- {t}" for t in tools] + \
           ["", "</details>", ""]


def render(turns, redact, config, exclude, title, note=None):
    out = [f"# {title}", ""]
    if note:
        out += [note, ""]
    out += ["Sanitized transcript of a Claude Code session. The user's prompts and Claude's replies "
            "are word for word. Each tool call is shown as one line; its output is omitted. "
            "Pasted content and Claude's internal reasoning are removed, and personal details are "
            "redacted. Nothing is removed by topic. See `../README.md` for how these files are made.", ""]
    report, n = [], 0
    while n < len(turns):
        turn = turns[n]
        n += 1
        if n in exclude:
            # a run of redacted turns with the same reason becomes one note
            reason, start = exclude[n], n
            while n + 1 in exclude and exclude[n + 1] == reason and n < len(turns):
                n += 1
            report += [f"| {i} | {turns[i - 1]['time']:%H:%M} | REDACTED (by request) | | |"
                       for i in range(start, n + 1)]
            span = f"Turn {start}" if start == n else f"Turns {start}–{n}"
            times = f"{turn['time']:%H:%M}" + ("" if start == n else f"–{turns[n - 1]['time']:%H:%M}")
            out += [f"## {span} ({times})", "", f"*[Redacted at the user's request: {reason}.]*", ""]
            continue
        blob = turn["prompt"] + "\n" + "\n".join(i[1] for i in turn["items"])
        m = config.flag.search(blob)
        first = redact(turn["prompt"].splitlines()[0][:100])
        report.append(f"| {n} | {turn['time']:%H:%M} | {'kept — check redaction' if m else 'kept'} | "
                      f"{('“' + m.group(0) + '”') if m else ''} | {first} |")
        out += [f"## Turn {n} ({turn['time']:%H:%M})", "", "**User:**", ""]
        out += ["> " + l if l else ">" for l in redact(turn["prompt"]).splitlines()]
        out.append("")
        tools = []
        for kind, text, _ in turn["items"]:
            if kind == "tool":
                tools.append(redact(text))
                continue
            if tools:
                out += tool_block(tools)
                tools = []
            out += ["**Claude:**", "", redact(text), ""]
        if tools:
            out += tool_block(tools)
    return "\n".join(out).rstrip() + "\n", report


def export(turns, out_dir, review_dir, stem, config, author=None, note=None, exclude_turns=()):
    """Write the record <stem>.md to out_dir and its review report to review_dir.
    Returns the record's path."""
    redact = Redactor(config)
    first, last = turns[0]["time"].date(), turns[-1]["time"].date()
    who = f"{author}'s " if author else ""
    title = f"{who}Claude Code session, {first:%-d %B %Y}" if first == last else \
        f"{who}Claude Code session, {first:%-d %b} – {last:%-d %b %Y}"
    exclude = {t: "private" for t in exclude_turns}
    for d in config.drops:
        if d.get("export") == stem:
            if d["turns"][1] > len(turns):
                raise ValueError(f"drops.json asks to redact turns {d['turns']} of {stem}, "
                                 f"which has only {len(turns)} turns")
            exclude.update({t: d["reason"] for t in range(d["turns"][0], d["turns"][1] + 1)})
    md, report = render(turns, redact, config, exclude, title, note)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.md").write_text(md)
    rep = [f"# Review report: {stem}", "", "Read the record before anything is committed.", "",
           "## Turns", "", "| # | Time | Status | Why | First line of prompt |", "|---|---|---|---|---|"]
    rep += report
    rep += ["", "## Redactions", ""] + [f"- {k}: {v}" for k, v in sorted(redact.counts.items())]
    rep += ["", "To redact a turn or line, add it to `drops.json` in your config folder, with the reason "
            "to show in the record, and run again."]
    review_dir = Path(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / f"{stem}.REVIEW.md").write_text("\n".join(rep) + "\n")
    return out_dir / f"{stem}.md"


def default_repo_paths():
    """The checkout this script lives in, if it's inside a git repo."""
    try:
        top = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, check=True).stdout.strip()
        return [top] if top else []
    except (OSError, subprocess.CalledProcessError):
        return []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", help="Claude Code transcript files (.jsonl)")
    ap.add_argument("--since", required=True, type=dt.date.fromisoformat, help="first date, YYYY-MM-DD")
    ap.add_argument("--until", required=True, type=dt.date.fromisoformat, help="end date, exclusive")
    ap.add_argument("--out", required=True, help="folder for the records and review reports")
    ap.add_argument("--review-dir", help="folder for the review reports, which name private terms; "
                    "default: <config-dir>/reviews, outside the repo")
    ap.add_argument("--author", help="whose sessions these are, shown in each title")
    ap.add_argument("--config-dir", default=os.environ.get("MOE_AI_LOG_DIR", "~/moe-ai-log"))
    ap.add_argument("--repo-path", action="append",
                    help="a local checkout; paths inside it are shown relative to it (repeatable)")
    ap.add_argument("--per-session", action="store_true", help="one record per session per day")
    ap.add_argument("--name", default="session", help="file name, without --per-session")
    ap.add_argument("--exclude-turn", type=int, action="append", default=[],
                    help="without --per-session: redact this turn as private (repeatable)")
    a = ap.parse_args(argv)

    config = Config(a.config_dir, a.repo_path if a.repo_path is not None else default_repo_paths())
    review_dir = a.review_dir or Path(a.config_dir).expanduser() / "reviews"
    entries, forks = load_entries(sorted(a.sessions, key=lambda p: Path(p).stat().st_mtime))
    written = []
    if not a.per_session:
        turns = build_turns(entries, a.since, a.until)
        if turns:
            written.append(export(turns, a.out, review_dir, f"{a.since:%Y-%m-%d}_{a.name}", config, a.author,
                                  exclude_turns=a.exclude_turn))
    else:
        groups = {}
        for e in entries:
            groups.setdefault(e.get("sessionId", "unknown"), []).append(e)
        for sid, group in groups.items():
            note = None
            if sid in forks:
                parent, last = forks[sid]
                note = (f"*Session `{sid[:8]}` was forked from session `{parent[:8]}` at "
                        f"{local(last):%-d %B %Y, %H:%M}. Their shared history appears only "
                        f"in `{parent[:8]}`'s transcripts.*")
            day = a.since
            while day < a.until:
                turns = build_turns(group, day, day + dt.timedelta(days=1))
                if turns:
                    written.append(export(turns, a.out, review_dir, f"{day:%Y-%m-%d}_claude-code_{sid[:8]}",
                                          config, a.author, note))
                day += dt.timedelta(days=1)
    stems = {p.stem for p in written}
    for d in config.drops:
        if "export" in d and d["export"] not in stems and \
                a.since.isoformat() <= d["export"][:10] < a.until.isoformat():
            warn(f"drops.json names {d['export']}, but no such record was written in this run")
    print(f"{len(written)} export(s) written")
    return written


if __name__ == "__main__":
    main()
