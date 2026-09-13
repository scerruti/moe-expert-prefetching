# Agent Context for Autonomous Issue Processing

This is a condensed, agent-focused version of project documentation.
**Generated automatically** - do not edit directly.

To update: Run `python agents/scripts/generate_agent_context.py` when docs change.

---

## Project: Speculative Expert Prefetching in Mixture-of-Experts Models

**Goal**: Predict expert activations in sparse MoE models to enable speculative weight prefetching.

**Technology**: PyTorch, Transformers, streaming data collection from MoE models.

---

## 5-Phase Implementation Roadmap

### Phase 1: Telemetry Hooks & Deterministic Data Collection
**Objective:** Capture pre-softmax logits, post-softmax routing distributions, and top-$k$ assignments directly from model forward passes.

**Mechanism:** PyTorch forward hooks registered on router li

### Phase 2: Statistical Verification & Domain Cross-Comparison
**Objective:** Verify router distribution determinism and profile cross-domain probability shifts.

**Run Strategy:**

* **Sampling Subsets:** Stratified random sample of 200 prompts per dataset to mi

### Phase 3: Speculative Predictor Architecture
**Objective:** Train a lightweight auxiliary model to forecast future expert activations ahead of actual forward execution.

**Predictor Architecture:**

* **Sliding-Window Lightweight Transformer:**

### Phase 4: Cache Simulation & Prefetching Pipeline
**Objective:** Simulate asynchronous hardware memory pipeline; measure prefetch effectiveness under realistic latency constraints.

**Prefetch Execution Rules:**

* **Prediction Horizon:** $H$ = numbe

### Phase 5: Pipeline Profiling & Validation Metrics
**Objective:** Benchmark the speedup, accuracy, and pipeline stall reductions achieved by speculative prefetching.

**Primary Evaluation Metrics:**

* **Prediction Recall@$(k+m)$:** Fraction of actual

---

## Expected Directory Structure

```
phase_1/
├── scripts/
│   ├── verify_environment.py
│   ├── data_collection.py
│   ├── validation.py
│   └── dataset_loading.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── STATUS.md
│   └── CHECKLIST.md
└── data/
    ├── gsm8k/
    ├── mbpp/
    ├── processed/
    └── validation_reports/
```

---

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
**Next Update**: Run `python agents/scripts/generate_agent_context.py`
