# SpecRhythm vLLM Plugin

This repository develops a vLLM research plugin for reproducing and extending
SpecRhythm, an ATC 2026 submission on heterogeneous SLO-aware speculative
decoding for large language model serving.

The intended workflow is:

1. Edit and commit code on a local Mac.
2. Push changes to GitHub.
3. Pull the same branch on a GPU server.
4. Run vLLM and benchmark experiments on the server.
5. Bring logs and failures back into the development loop.

## What This Work Does

SpecRhythm targets online LLM serving where requests have different
time-per-output-token (TPOT) service-level objectives. Instead of optimizing raw
throughput only, it optimizes goodput: output tokens from requests that meet
their latency SLO.

The planned vLLM plugin will add:

- request-level SLO state tracking;
- TPOT, SLO attainment, and goodput measurement;
- SLO-aware speculative budget allocation;
- dual-batch draft/verify execution planning;
- rolling eager continuation for urgent requests;
- guarded promotion or discard of ahead-of-turn draft continuations.

## Current Status

This repository currently contains the initial plugin skeleton. The plugin does
not patch vLLM internals yet. Its first purpose is to prove that vLLM can
discover and load the out-of-tree plugin on the GPU server.

Target vLLM version for the first implementation:

```text
vllm==0.23.0
```

The version is pinned because SpecRhythm will likely need internal scheduler and
speculative decoding patches. Those APIs are not stable across vLLM releases.

## Repository Layout

```text
.
|-- README.md
|-- pyproject.toml
|-- benchmarks/
|   |-- datasets/
|   `-- results/
|-- figures/
|-- paper/
|   |-- ATC26_#1664_SpecRhythm.pdf
|   |-- ATC26_#1664_SpecRhythm_extended_abstract.pdf
|   |-- ATC26_SpecRhythm.pdf
|   |-- ATC26_SpecRhythm__two_page_extended_abstract_.pdf
|   `-- ATC26_SpecRhythm_extended_abstract_revised.tex
|-- scripts/
|   |-- inspect_vllm.py
|   `-- run_smoke_import.py
`-- src/
    `-- vllm_specrhythm/
        |-- config.py
        |-- metrics.py
        |-- patching.py
        |-- plugin.py
        `-- scheduler/
            |-- budget_shaper.py
            |-- dual_batch.py
            `-- slo_state.py
```

## Local Mac Development

The local Mac does not need vLLM or a GPU for basic code edits and syntax
checks.

Install the package in editable mode without vLLM:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python scripts/run_smoke_import.py
```

Expected smoke output:

```text
[SpecRhythm] plugin loaded (target vLLM version: 0.23.0)
```

## GPU Server Setup

On the server, clone or pull this repository:

```bash
git clone https://github.com/rzwang22/SpecRhythm-vllm-plugin.git
cd SpecRhythm-vllm-plugin
```

Install vLLM and this plugin. If vLLM is already installed in the server
environment, install only this package:

```bash
python -m pip install -e .
```

If vLLM is not installed yet, use the pinned extra:

```bash
python -m pip install -e ".[vllm]"
```

Inspect the server-side vLLM installation:

```bash
python scripts/inspect_vllm.py
```

Verify that vLLM loads the plugin:

```bash
VLLM_PLUGINS=specrhythm python -c "import vllm"
```

You should see:

```text
[SpecRhythm] plugin loaded (target vLLM version: 0.23.0)
```

## Development Milestones

1. Confirm plugin loading on the GPU server.
2. Inspect vLLM 0.23.0 scheduler and speculative decoding internals.
3. Add non-invasive TPOT, SLO attainment, and goodput measurement.
4. Run vLLM autoregressive and official speculative decoding baselines.
5. Patch request state to carry SLO metadata.
6. Implement fixed speculative budget shaping.
7. Implement SLO-aware budget shaping.
8. Implement dual-batch scheduling.
9. Add draft/verify overlap and rolling eager continuation.
10. Add profiling-based verification budget tables.

## Related Work

- SpecRhythm paper materials are stored in `paper/`.
- MineDraft is a useful example of a vLLM out-of-tree patch plugin for parallel
  speculative decoding.
- nano-PEARL is a useful reference for draft-target disaggregation and parallel
  speculative decoding, but it is not the implementation base for this
  repository.
