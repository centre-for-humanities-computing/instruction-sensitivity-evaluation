# One prompt is not enough: Instruction Sensitivity Undermines Embedding Model Evaluation

This repository contains the code and data for the paper _"One prompt is not enough: Instruction Sensitivity Undermines Embedding Model Evaluation"_.

## Overview

Instruction-tuned embedding models are typically evaluated with a single, fixed prompt per task. This single-point evaluation hides a critical problem: **these models are highly sensitive to prompt phrasing**. We present an empirical study across **6 embedding models**, **11 datasets**, and **15 task-specific prompts per dataset** (990 total evaluations) showing that:

- **Prompt deflation & inflation**: Reported scores can systematically understate or overstate a model's true performance distribution.
- **Leaderboard fragility**: By selecting prompts favorably, *any* model in our study can be promoted to rank 1 on a simulated leaderboard.
- **Prompt hacking**: Most models report scores above their expected performance, analogous to p-hacking — selectively reporting a favorable prompt without modifying the model itself.

We recommend that benchmarks transition from single-prompt evaluation to distribution-based robustness metrics.

## Repository Structure

```
prompt-hacking/
├── src/
│   ├── prompt_generation/       # Synthetic prompt generation via LLM
│   │   ├── generate_prompts.py  # Main script: generates 15 prompts per task using vLLM
│   │   └── prompt_template.py   # Prompt templates & validation for structured output
│   ├── running_experiments/     # MTEB evaluation runners
│   │   ├── run_models.py        # Running one model single-thread
│   │   └── run_models_parallel.py  # Running one model in parallel
├── data/
│   ├── prompts/                 # Generated prompts (JSON)
│   └── cache_data/mteb_cache/   # Cached MTEB evaluation results
│       └── results/             # Per-model result directories
├── pyproject.toml               # Project metadata & dependencies
├── makefile                     # Setup shortcuts
└── uv.lock                     # Locked dependency versions
```

## Models Evaluated

| Model | Type |
|---|---|
| `Qwen/Qwen3-Embedding-0.6B` | Multilingual |
| `intfloat/multilingual-e5-large-instruct` | Multilingual |
| `KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5` | Multilingual |
| `BAAI/bge-small-en-v1.5` | English |
| `BAAI/bge-base-en-v1.5` | English |
| `BAAI/bge-large-en-v1.5` | English |

## Tasks & Datasets

Covering 4 task types from MTEB/MMTEB:

| Task Type | Datasets |
|---|---|
| **Retrieval** | MIRACLRetrievalHardNegatives.v2, Touche2020Retrieval.v3, FEVERHardNegatives |
| **Classification** | TweetSentimentClassification, ImdbClassification, AmazonCounterfactualClassification |
| **Clustering** | MedrxivClusteringP2P.v2, StackExchangeClustering.v2 |
| **Semantic Similarity** | STS14, STS15, STS22.v2 |

## Setup

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
make install
# or equivalently:
uv sync
```

## Usage

### 1. Generate Prompts

Generate 15 synthetic task-specific prompts per task using a language model (default: `openai/gpt-oss-120b`) with vLLM structured outputs:

```bash
uv run python src/prompt_generation/generate_prompts.py
```

This produces a JSON file in `data/prompts/` containing the generated prompts along with metadata (task type, dataset description, language).

### 2. Run Experiments

Evaluate embedding models on all tasks with each generated prompt. The runner loads the model once and iterates over all prompt/task combinations:

```bash
# Multi-GPU (recommended)
uv run python src/running_experiments/run_models.py <model_name> \
    --prompts_path data/prompts/openai-gpt-oss-120b-gen-prompts.json \
    --task_description <optional: filter by task type> \
    --enc_batch_size 32

# Single-GPU
uv run python src/running_experiments/run_models_single_gpu.py
```

Results are cached under `data/cache_data/mteb_cache/results/` following the MTEB result format, with custom prompts stored under an `experiments/` subdirectory per model revision.

## Result Cache Format

```
data/cache_data/mteb_cache/results/
└── <model_name>/
    └── <revision>/
        ├── <TaskName>.json           # Baseline (default prompt) results
        ├── model_meta.json
        └── experiments/
            └── <prompt_config>/
                └── <TaskName>.json   # Custom prompt results
```

## Citation

```bibtex
@misc{kostiuk2026promptenoughinstructionsensitivity,
      title={One prompt is not enough: Instruction Sensitivity Undermines Embedding Model Evaluation}, 
      author={Yevhen Kostiuk and Kenneth Enevoldsen},
      year={2026},
      eprint={2605.22544},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.22544}, 
}
```

## License

See the repository for license details.
