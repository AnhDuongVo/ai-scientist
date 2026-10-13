# Validation evidence — 9 October 2026

`pip install -e ".[dev]"` succeeded in a fresh virtual environment for this repository, independently of the other projects. Tests ran on Python 3.12.14 / Darwin arm64 CPU. Other Python versions in CI have not been executed locally here.

| Check | Result |
|---|---|
| Offline test suite | 15 passed |
| Ruff lint | Passed |
| Ruff formatting | Passed |
| Mocked/simulated integration | Tested within the scope below |
| Live NVIDIA hosted endpoint | Not executed |
| Self-hosted GPU endpoint | Not executed |
| Clinical/scientific domain validation | Not completed |

## Tested scope

Binder-only chain selection, sequence-length mapping, cross-process stable seeds, seed payloads, mocked HTTP errors and async parsing, placeholder rejection. Optional NAT registration and simulator workflow executed separately.

The GitHub workflows have been added or retained, but their remote execution has not been verified after these changes. Unit tests establish behavior on fixtures; they do not establish semantic or clinical correctness.

## NAT configuration

`nat validate --config_file configs/workflow.yml` passed with the installed toolkit. This checks configuration/schema validity and plugin discovery; it does not execute a live model workflow or verify the example model IDs.

The optional NAT registration test was additionally executed with `nvidia-nat[langchain]` installed: 1 passed. It invokes the registered function on a simulator target. The isolated base dev environment skips that optional test. Live contract tests remain skipped.

## Opt-in live contract test

Requires access and a real target fixture containing `pdb`, `contigs`, optional `hotspot_res`, and the expected generated `binder_chain`:

```bash
RUN_LIVE_NVIDIA=1 BIONEMO_TARGET_JSON=target.json pytest -q tests/test_live_contract.py
```

Set `NGC_API_KEY` or `NVIDIA_API_KEY`, and optionally `BIONEMO_BASE_URL`. Install the relevant live extra first (`.[bio]` for ai-scientist, `.[live]` for mcp-bionemo). The manual workflow requires `NVIDIA_API_KEY` and `BIONEMO_TARGET_JSON` secrets in the `nvidia-integration` environment. It exercises RFdiffusion and ProteinMPNN response contracts, not all scientific endpoints or biological validity.

The request changes follow the [RFdiffusion request schema](https://docs.api.nvidia.com/nim/reference/ipd-rfdiffusion-infer) and [ProteinMPNN chain-selection schema](https://docs.nvidia.com/nim/bionemo/proteinmpnn/latest/endpoints.html), checked 9 October 2026. Runtime behavior remains unverified.

## Re-review follow-up, 10 October 2026

The dedicated NAT job installs `.[dev,nat]`, requires the import, and executes registration plus a simulated workflow on push/PR/manual dispatch. The NAT test passed locally with NAT installed; live contract remains skipped.


## Workflow recording — 13 October 2026

The recording runs this repository’s actual candidate-design orchestration and ranking using SimulatedBioNeMo, rather than the companion app’s standalone ranking function. A placeholder target, contigs, hotspots and design settings are shown. Two backbone calls, two sequence-design calls and six folding calls produce six simulated candidates; viewers inspect a sequence, the recorded stage calls and the assembled research report. No abstracts are retrieved in this offline run. RFdiffusion, ProteinMPNN and Boltz-2 are simulated; no live inference occurs. The placeholder has no target sequence, so folding is binder-only. Structures, sequences and scores have no biological meaning.

The added optional Streamlit AppTest executes the repository pipeline and verifies the resulting output and review behavior. Install `.[demo,dev]` to include it. This does not validate hosted/self-hosted endpoints or domain correctness.

Validation for this change: full local suite **17 passed** and **1 optional integration test skipped**. Ruff checks passed for the interface, fixtures and added regression test. MP4 decoding passed without errors. No live endpoint or domain validation was performed.
