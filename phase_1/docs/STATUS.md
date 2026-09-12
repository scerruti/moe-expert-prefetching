# Phase 1 Status Report

**Date:** 2026-09-12  
**Project:** Speculative Expert Prefetching in Mixture-of-Experts Models  
**Phase:** 1 of 5 (Telemetry Hooks & Deterministic Data Collection)

---

## Summary

Phase 1 is **fully designed** and **ready for implementation**. All architectural decisions have been made. Development can begin immediately.

---

## What's Done (Design Phase)

### ✅ Architecture & Specification
- [x] SYSTEM_DESIGN.md (320 lines): Complete technical specification with math
- [x] Data collection methodology: PyTorch forward hooks on router layers
- [x] Storage schema: Columnar Parquet format with 9 columns
- [x] Validation strategy: Determinism checks + randomization consistency
- [x] Dataset selection: GSM8K (math) + MBPP (code) for domain diversity

### ✅ Key Decisions Locked In
- **Model:** Mixtral 8x7B (8 experts, top-2 routing)
- **Datasets:** 8.5k + 974 examples (9.5k total)
- **Runs:** 5 randomized passes per dataset
- **Hook Target:** `block_sparse_moe.gate` module at each layer
- **Output Format:** Apache Parquet (compressed, columnar)
- **Validation:** Bitwise determinism across 5 runs, 100% match expected

### ✅ Documentation
- [x] README.md (overview & quick start)
- [x] SYSTEM_DESIGN.md (detailed 5-phase roadmap)
- [x] CONTRIBUTORS.md (team members)
- [x] Design discussions in docs/conversations/ (meeting notes)
- [x] phase_1/docs/REVIEW.md (this checklist & status)
- [x] phase_1/docs/CHECKLIST.md (granular task tracking)
- [x] phase_1/docs/ARCHITECTURE.md (implementation reference)

---

## What's Missing (Implementation Phase)

### ❌ Code
```
phase_1/scripts/
├── data_collection.py           [NOT STARTED]
├── validation.py                 [NOT STARTED]
├── dataset_loading.py            [NOT STARTED]
└── README.md                     [NOT STARTED]

requirements.txt                  [NOT STARTED]
```

### ❌ Data Outputs
```
data/
├── gsm8k_routing_traces.parquet  [NOT GENERATED]
├── mbpp_routing_traces.parquet   [NOT GENERATED]
├── validation_reports/
│   └── phase_1_validation.md     [NOT GENERATED]
└── phase_1_metadata.json         [NOT GENERATED]
```

---

## Implementation Roadmap

| Week | Milestone | Status |
|------|-----------|--------|
| 1 | Environment setup, dataset loaders, model loading | 🔲 TODO |
| 2 | Hook implementation, data collection pipeline, Parquet I/O | 🔲 TODO |
| 3 | Validation testing, full-scale collection, analysis | 🔲 TODO |
| 4 | Documentation, sign-off, Phase 1 completion | 🔲 TODO |

**Target Completion:** 2026-10-03 (3 weeks)

---

## Dependencies

### Python Packages (to be added to requirements.txt)
```
torch>=2.0
transformers>=4.30
datasets>=2.10
pyarrow>=12.0
tqdm>=4.60
numpy>=1.20
```

### Optional (for Phase 2+)
```
duckdb>=0.8
polars>=0.18
matplotlib>=3.5
```

### Hardware Requirements
- GPU: A100 or H100 (80GB+ HBM)
- CPU RAM: 32GB+
- Storage: 50GB for Parquet files

---

## Success Criteria for Phase 1

All of the following must be true:

- [ ] 9.5k+ examples processed without errors
- [ ] Routing data captured for all 32 MoE layers (Mixtral)
- [ ] Parquet files pass integrity checks (valid schema, no corruption)
- [ ] Determinism verification: 100% of sample prompts show identical routing across 5 runs
- [ ] Randomization check: 100% of shared tokens show same top-k experts across orderings
- [ ] Validation report documents all findings
- [ ] Code is well-commented and reproducible
- [ ] All artifacts committed to git with clear commit messages

---

## Known Unknowns (TBD during implementation)

| Question | Impact | Decision Window |
|----------|--------|-----------------|
| Exact Mixtral layer path (`block_sparse_moe.gate` vs alternatives)? | HIGH | Days 1-2 (quick test) |
| What batch size fits in A100/H100 HBM? | HIGH | Days 2-3 (profiling) |
| Max sequence length for GSM8K/MBPP? | MEDIUM | Days 1-2 (data inspection) |
| Hook overhead (memory, latency)? | MEDIUM | Days 3-4 (benchmarking) |
| Parquet compression ratio (estimate memory on disk)? | LOW | Days 4-5 (first writes) |

---

## Resource Requirements

### Personnel
- **Implementation:** 1 lead engineer (3-4 weeks, full-time)
- **Oversight:** Project lead (weekly check-ins)
- **Infrastructure:** DevOps (GPU allocation, monitoring)

### Infrastructure
- GPU cloud instance (A100 or H100): 2-4 weeks × 1 instance
- Storage: S3 or local (50GB for Parquet files)
- Version control: GitHub (already set up)

### Timeline
- **Start:** Week of 2026-09-12
- **Completion:** Week of 2026-10-03
- **Buffer:** 1 week (debugging, unexpected issues)

---

## Documentation Artifacts (Now Available)

1. **SYSTEM_DESIGN.md** – Full technical specification (read first)
2. **phase_1/docs/REVIEW.md** – This review, component breakdown, risks
3. **phase_1/docs/CHECKLIST.md** – Granular task list (80+ items)
4. **phase_1/docs/ARCHITECTURE.md** – Implementation reference guide
5. **phase_1/docs/STATUS.md** – Status report (this file)

**Recommendation:** Read in order: SYSTEM_DESIGN → ARCHITECTURE → CHECKLIST

---

## Next Steps

### Immediate (This Week)
1. [ ] Assign implementation lead
2. [ ] Provision GPU instance (A100/H100)
3. [ ] Create `requirements.txt` and test environment
4. [ ] Begin dataset loader implementation
5. [ ] Test model loading and hook mechanism

### This Month
6. [ ] Complete main data collection loop
7. [ ] Implement Parquet serialization
8. [ ] Run full Phase 1 collection (5 runs, both datasets)
9. [ ] Complete validation testing
10. [ ] Finalize documentation and commit

### Sign-Off
- [ ] Phase 1 review meeting with team
- [ ] Validation report reviewed and approved
- [ ] Code committed to main branch
- [ ] Phase 2 kickoff: Statistical analysis

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Hook registration fails | LOW | HIGH | Test on toy model first day |
| OOM during collection | MEDIUM | HIGH | Batch writes, incremental processing |
| Non-deterministic routing (unexpected) | LOW | CRITICAL | Verify immediately if detected |
| Parquet schema issues | LOW | MEDIUM | Validate schema on first batch |
| GPU timeout / instance interruption | MEDIUM | HIGH | Checkpoint every 100 prompts |
| Tokenization errors in edge cases | MEDIUM | LOW | Log & skip problematic prompts |

---

## Questions & Clarifications

### For Project Lead
1. **Hardware Access:** Which cloud provider (RunPod, Lambda, etc.)? Who handles provisioning?
2. **Data Privacy:** Can we commit Parquet files to git, or use S3 storage?
3. **Model Licensing:** Confirm Mixtral 8x7B download is authorized.
4. **Timeline Flexibility:** If Phase 1 takes longer, are downstream phases pushed?

### For Implementation Team
1. **Python Version:** Should we target 3.10, 3.11, or 3.12?
2. **Testing Infrastructure:** Pytest or unittest? Any CI/CD pipeline?
3. **Code Review Process:** How many reviewers? Approval process?
4. **Deployment:** Will Phase 1 code run locally or only on cloud GPU?

---

## Approval & Sign-Off

| Role | Name | Date | Sign-Off |
|------|------|------|----------|
| Project Lead | [TBD] | — | ☐ Approved |
| Implementation Lead | [TBD] | — | ☐ Ready |
| Data/ML Lead | [TBD] | — | ☐ OK to proceed |

---

## Contact & Support

- **Project Repo:** /Users/scerruti/moe
- **Git Branches:** Use `feature/phase-1` for development
- **Documentation:** See phase_1/docs/CHECKLIST.md for detailed tasks
- **Questions:** Open issues in git or email team

---

**Status:** 🟢 **READY FOR IMPLEMENTATION**

All planning complete. Implementation can begin immediately.
