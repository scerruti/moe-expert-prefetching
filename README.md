# Speculative Expert Prefetching in Mixture-of-Experts Models

Research on predicting expert activations in sparse MoE models to enable speculative weight prefetching and reduce GPU pipeline stalls.

## Quick Start

- **Design Document:** [`SYSTEM_DESIGN.md`](SYSTEM_DESIGN.md) – Full technical specification (5 phases)
- **Prior Work:** [`docs/PRIOR_WORK_McNAIR.md`](docs/PRIOR_WORK_McNAIR.md) – Lessons from concurrent spatial reasoning research
- **Conversations:** [`docs/conversations/`](docs/conversations/) – Design discussions and meeting notes

## Project Structure

```
moe/
├── SYSTEM_DESIGN.md          # Main technical document
├── docs/
│   ├── PRIOR_WORK_McNAIR.md  # Prior research context
│   ├── conversations/         # Conversation logs & meeting notes
│   └── references/            # PDFs and external resources
├── scripts/                   # Phase-specific implementation (TBD)
├── data/                      # Results and artifacts (TBD)
└── README.md                  # This file
```

## Overview

Mixture-of-Experts (MoE) models like Mixtral 8x7B route each token to a sparse subset of expert networks. The bottleneck: transferring expert weights from host memory to GPU HBM creates pipeline stalls. This project investigates whether we can **predict upcoming expert activations ahead of time** and prefetch their weights speculatively.

**Key Insight:** During the prefill phase, expert routing is deterministic—we can reliably profile which experts will activate on which tokens without stochastic noise.

## Current Status

**Phase:** Design finalized, ready for Phase 1 implementation (telemetry hooks)

See [`SYSTEM_DESIGN.md`](SYSTEM_DESIGN.md) for full technical roadmap.

---

## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for team details.

---

For questions or context, see [`docs/conversations/`](docs/conversations/) for design discussions.
