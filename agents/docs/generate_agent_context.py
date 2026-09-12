#!/usr/bin/env python3
"""
Generate condensed AGENT_CONTEXT.md from project documentation.

This script reads the main documentation files and extracts key information
needed by the autonomous agent, creating a focused, efficient context document.

Usage:
    python agents/docs/generate_agent_context.py

This should be run whenever SYSTEM_DESIGN.md or phase_*/docs/*.md files change.
"""

import re
from pathlib import Path


def extract_phase_info():
    """Extract phase structure and key files from documentation."""
    phases = {}

    # Read SYSTEM_DESIGN.md for phase overview
    try:
        design = Path("SYSTEM_DESIGN.md").read_text()
        # Extract phase descriptions from "## N. Phase X:" sections
        phase_sections = re.findall(r'### Phase (\d+)[:\s]+([^\n]+)\n+(.*?)(?=### Phase|\Z)', design, re.DOTALL)
        for phase_num, title, content in phase_sections:
            phases[int(phase_num)] = {
                'title': title.strip(),
                'description': content[:200].strip()
            }
    except FileNotFoundError:
        pass

    return phases


def extract_issue_info():
    """Extract issue requirements from GitHub issues documentation."""
    issues = {}

    # Read phase_1/docs/GITHUB_ISSUES.md for issue details
    try:
        issues_doc = Path("phase_1/docs/GITHUB_ISSUES.md").read_text()

        # Extract issue blocks
        issue_blocks = re.findall(
            r'### Issue #(\d+)[:\s]+([^\n]+)\n+(.*?)(?=### Issue|\Z)',
            issues_doc,
            re.DOTALL
        )

        for issue_num, title, content in issue_blocks:
            # Extract files to create and acceptance criteria
            files = re.findall(r'- Create[:\s]+`([^`]+)`', content)
            acceptance = re.findall(r'- (\w.*?)(?:\n|$)', content)[:3]

            issues[int(issue_num)] = {
                'title': title.strip(),
                'files': files,
                'acceptance': acceptance
            }
    except FileNotFoundError:
        pass

    return issues


def extract_directory_structure():
    """Extract expected directory structure from documentation."""
    structure = {}

    try:
        status = Path("phase_1/docs/STATUS.md").read_text()

        # Extract What's Missing section which shows expected structure
        missing = re.search(r'### ❌ Code\n```\n(.*?)\n```', status, re.DOTALL)
        if missing:
            structure['code'] = missing.group(1).strip()

        missing_data = re.search(r'### ❌ Data Outputs\n```\n(.*?)\n```', status, re.DOTALL)
        if missing_data:
            structure['data'] = missing_data.group(1).strip()
    except FileNotFoundError:
        pass

    return structure


def generate_context():
    """Generate the condensed agent context document."""

    phases = extract_phase_info()
    issues = extract_issue_info()
    structure = extract_directory_structure()

    content = """# Agent Context for Autonomous Issue Processing

This is a condensed, agent-focused version of project documentation.
**Generated automatically** - do not edit directly.

To update: Run `python agents/docs/generate_agent_context.py` when docs change.

---

## Project: Speculative Expert Prefetching in Mixture-of-Experts Models

**Goal**: Predict expert activations in sparse MoE models to enable speculative weight prefetching.

**Technology**: PyTorch, Transformers, streaming data collection from MoE models.

---

## 5-Phase Implementation Roadmap

"""

    # Add phase overviews
    for phase_num in sorted(phases.keys()):
        phase = phases[phase_num]
        content += f"### Phase {phase_num}: {phase['title']}\n"
        content += f"{phase['description']}\n\n"

    # Add issue-specific guidance
    content += "---\n\n## Issue Reference Guide\n\n"

    for issue_num in sorted(issues.keys()):
        issue = issues[issue_num]
        content += f"### Issue #{issue_num}: {issue['title']}\n\n"

        if issue['files']:
            content += "**Files to Create**:\n"
            for f in issue['files']:
                content += f"- `{f}`\n"
            content += "\n"

        if issue['acceptance']:
            content += "**Acceptance Criteria**:\n"
            for criteria in issue['acceptance']:
                content += f"- {criteria}\n"
            content += "\n"

    # Add directory structure
    content += "---\n\n## Expected Directory Structure\n\n"
    content += "```\nphase_1/\n├── scripts/\n│   ├── verify_environment.py\n│   ├── data_collection.py\n│   ├── validation.py\n│   └── dataset_loading.py\n├── docs/\n│   ├── ARCHITECTURE.md\n│   ├── STATUS.md\n│   └── CHECKLIST.md\n└── data/\n    ├── gsm8k/\n    ├── mbpp/\n    ├── processed/\n    └── validation_reports/\n```\n\n"

    # Add key patterns
    content += """---

## Key Patterns for Implementation

### File Creation
- Use `write_file()` tool to create Python scripts
- Include proper docstrings and type hints
- Follow PEP 8 standards

### Testing & Validation
- Use `bash` tool to run tests
- Create `verify_*.py` scripts for validation
- Test environment setup before implementation

### Dependencies
- Create `requirements.txt` for core dependencies
- Create `requirements-dev.txt` for development tools
- List all versions explicitly

### Git Workflow
- Create feature branch from issue
- Commit changes with clear message
- Create PR with acceptance criteria checklist

---

## Quick Command Reference

**Environment Setup**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python phase_1/scripts/verify_environment.py --create-dirs
```

**Testing**:
```bash
pytest phase_1/scripts/test_*.py -v
```

**Running Agent**:
```bash
python agents/autonomous_agent.py --phase 1 --single
```

---

**Last Updated**: Auto-generated from source documentation.
**Next Update**: Run `python agents/docs/generate_agent_context.py`
"""

    return content


def main():
    """Generate and save the agent context document."""
    context = generate_context()

    output_path = Path("agents/docs/AGENT_CONTEXT.md")
    output_path.write_text(context)

    print(f"✅ Generated {output_path}")
    print(f"   {len(context)} characters")
    print(f"   Ready for agent use")


if __name__ == "__main__":
    main()
