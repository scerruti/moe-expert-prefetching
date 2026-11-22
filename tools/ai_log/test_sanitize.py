"""Tests for the session-record sanitizer, on small synthetic transcripts."""

import json
import os

import pytest

import sanitize

DAY = "2026-11-21"
WINDOW = ["--since", "2026-11-20", "--until", "2026-11-23"]


def ts(minute):
    # midday UTC, so the local date is the same in every US and European time zone
    return f"{DAY}T18:{minute:02d}:00.000Z"


def prompt(minute, text, session="aaaaaaaa-1"):
    return {"type": "user", "timestamp": ts(minute), "sessionId": session, "uuid": f"u{minute}{session}",
            "message": {"role": "user", "content": text}}


def reply(minute, blocks, session="aaaaaaaa-1"):
    return {"type": "assistant", "timestamp": ts(minute), "sessionId": session, "uuid": f"a{minute}{session}",
            "message": {"role": "assistant", "content": blocks}}


def text(t):
    return {"type": "text", "text": t}


def tool(tool_id, name, **inp):
    return {"type": "tool_use", "id": tool_id, "name": name, "input": inp}


def result(minute, tool_id, output, session="aaaaaaaa-1"):
    return {"type": "user", "timestamp": ts(minute), "sessionId": session, "uuid": f"r{minute}{session}",
            "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id,
                                                     "content": output}]}}


def write_session(path, entries, extra_lines=()):
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
        for line in extra_lines:
            f.write(line + "\n")
    return path


@pytest.fixture
def config_dir(tmp_path):
    d = tmp_path / "config"
    d.mkdir()
    (d / "private_terms.txt").write_text("# a comment\nAnn Smith\n")
    return d


def run(tmp_path, config_dir, sessions, *extra):
    out = tmp_path / "out"
    sanitize.main([*map(str, sessions), *WINDOW, "--out", str(out), "--config-dir", str(config_dir),
                   "--repo-path", "/Users/someone/repo", *extra])
    return out


def record(out, name="2026-11-20_session"):
    return (out / f"{name}.md").read_text()


def test_prompts_and_replies_are_kept_word_for_word_and_tool_output_is_left_out(tmp_path, config_dir):
    s = write_session(tmp_path / "s.jsonl", [
        prompt(0, "Why did the collection fail?"),
        reply(1, [text("Let me check the log."), tool("t1", "Bash", command="cat log", description="Read the log")]),
        result(2, "t1", "SECRET TOOL OUTPUT"),
        reply(3, [text("It ran out of memory.")]),
    ])
    md = record(run(tmp_path, config_dir, [s]))
    assert "> Why did the collection fail?" in md
    assert "Let me check the log." in md and "It ran out of memory." in md
    assert "- Ran a command: Read the log" in md
    assert "SECRET TOOL OUTPUT" not in md and "cat log" not in md


def test_personal_data_is_redacted(tmp_path, config_dir):
    s = write_session(tmp_path / "s.jsonl", [
        prompt(0, "Email ann@example.edu or call 858-555-0100. Ask Ann Smith."),
        reply(1, [text("Paid with Mastercard ending 4242, shown as •••• 4242. Token ghp_abcdefghijklmnop.")]),
        reply(2, [text("The stipend is $8,000. Colab cost $29.97.")]),
        reply(3, [text("Edited /Users/someone/repo/phase_1/x.py and /Users/someone/Documents/plan.md")]),
    ])
    md = record(run(tmp_path, config_dir, [s]))
    for leaked in ["ann@example.edu", "858-555-0100", "Ann Smith", "4242", "ghp_abc", "$8,000", "plan.md",
                   "/Users/someone"]:
        assert leaked not in md, leaked
    assert "The stipend is $[amount]." in md  # amounts on a pay line are redacted, whatever they are
    assert "phase_1/x.py" in md  # repo paths are shown relative to the repo


def test_other_amounts_are_shown(tmp_path, config_dir):
    s = write_session(tmp_path / "s.jsonl", [prompt(0, "What did the GPU cost?"),
                                             reply(1, [text("Three runs at $9.99 each.")])])
    assert "$9.99" in record(run(tmp_path, config_dir, [s]))


def test_pasted_content_is_removed_and_counted(tmp_path, config_dir):
    s = write_session(tmp_path / "s.jsonl", [
        prompt(0, "Here's the log:\n<pasted_content id=\"x\">line 1\nline 2\nline 3</pasted_content id=\"x\">"),
        reply(1, [text("Thanks.")]),
    ])
    md = record(run(tmp_path, config_dir, [s]))
    assert "*[pasted content removed: 3 lines]*" in md
    assert "line 2" not in md


def test_text_the_user_did_not_type_is_not_shown_as_a_prompt(tmp_path, config_dir):
    s = write_session(tmp_path / "s.jsonl", [
        prompt(0, "Base directory for this skill: /x\nSkill instructions"),
        prompt(1, "<task-notification>done</task-notification>"),
        prompt(2, "This session is being continued from a previous conversation"),
        prompt(3, "Real question <system-reminder>hidden</system-reminder>"),
        reply(4, [text("Answer.")]),
    ])
    md = record(run(tmp_path, config_dir, [s]))
    assert md.count("**User:**") == 1
    assert "> Real question" in md and "hidden" not in md and "Skill instructions" not in md


def test_a_blocked_action_is_marked(tmp_path, config_dir):
    s = write_session(tmp_path / "s.jsonl", [
        prompt(0, "Edit the issue"),
        reply(1, [tool("t1", "Bash", description="Update issue 8")]),
        result(2, "t1", "Permission for this action was denied by the classifier"),
    ])
    assert "Update issue 8 — **blocked by permission check**" in record(run(tmp_path, config_dir, [s]))


def test_a_fork_shows_only_its_own_turns_and_says_where_it_came_from(tmp_path, config_dir):
    shared = [prompt(0, "Shared question"), reply(1, [text("Shared answer.")])]
    parent = write_session(tmp_path / "parent.jsonl", shared)
    fork_entries = [dict(e, sessionId="bbbbbbbb-2", uuid=e["uuid"] + "-copy") for e in shared]
    fork = write_session(tmp_path / "fork.jsonl", fork_entries + [
        prompt(5, "Fork question", session="bbbbbbbb-2"), reply(6, [text("Fork answer.")], session="bbbbbbbb-2")])
    os.utime(parent, (1, 1))  # the parent is older
    out = run(tmp_path, config_dir, [fork, parent], "--per-session")
    fork_md = record(out, f"{DAY}_claude-code_bbbbbbbb")
    assert "Fork question" in fork_md and "Shared question" not in fork_md
    assert "was forked from session `aaaaaaaa`" in fork_md
    assert "Shared question" in record(out, f"{DAY}_claude-code_aaaaaaaa")


def test_requested_redactions_are_marked_with_their_reason(tmp_path, config_dir):
    (config_dir / "drops.json").write_text(json.dumps([
        {"export": "2026-11-20_session", "turns": [2, 3], "reason": "personal"},
        {"line_match": "(?i)secret plan", "reason": "off-project"},
    ]))
    s = write_session(tmp_path / "s.jsonl", [
        prompt(0, "One"), reply(1, [text("Mentions the secret plan here.\nKeeps this line.")]),
        prompt(2, "Two, private"), reply(3, [text("Private answer.")]),
        prompt(4, "Three, private"), prompt(5, "Four"),
    ])
    md = record(run(tmp_path, config_dir, [s]))
    assert "## Turns 2–3" in md and "*[Redacted at the user's request: personal.]*" in md
    assert "Two, private" not in md and "Private answer" not in md
    assert "*[Line redacted at the user's request: off-project.]*" in md and "Keeps this line." in md
    assert "## Turn 4 " in md  # numbering is unchanged after a redaction


def test_a_redaction_beyond_the_last_turn_is_an_error(tmp_path, config_dir):
    (config_dir / "drops.json").write_text(json.dumps(
        [{"export": "2026-11-20_session", "turns": [5, 6], "reason": "personal"}]))
    s = write_session(tmp_path / "s.jsonl", [prompt(0, "Only turn")])
    with pytest.raises(ValueError, match="has only 1 turns"):
        run(tmp_path, config_dir, [s])


def test_a_malformed_redaction_request_is_an_error(tmp_path, config_dir):
    (config_dir / "drops.json").write_text(json.dumps([{"export": "x", "turns": [3, 1]}]))
    s = write_session(tmp_path / "s.jsonl", [prompt(0, "Hi")])
    with pytest.raises(ValueError, match="needs a reason"):
        run(tmp_path, config_dir, [s])


def test_unreadable_lines_are_reported_not_silently_skipped(tmp_path, config_dir, capsys):
    s = write_session(tmp_path / "s.jsonl", [prompt(0, "Hi")], extra_lines=['{"partial'])
    run(tmp_path, config_dir, [s])
    assert "1 unreadable transcript line(s) skipped" in capsys.readouterr().err


def test_review_reports_stay_out_of_the_record_folder(tmp_path, config_dir):
    s = write_session(tmp_path / "s.jsonl", [prompt(0, "Ask Ann Smith about the invoice")])
    out = run(tmp_path, config_dir, [s])
    assert not list(out.glob("*.REVIEW.md"))
    report = (config_dir / "reviews" / "2026-11-20_session.REVIEW.md").read_text()
    assert "check redaction" in report and "Ann Smith" not in report
