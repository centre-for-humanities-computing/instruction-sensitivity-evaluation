import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mteb
from mteb.models.instruct_wrapper import InstructSentenceTransformerModel

# setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

CACHE_PATH = Path("cache_data") / "mteb_cache"

# Globals populated once per worker process by _worker_init(). Each worker
# loads its own model instance so processes never share mutable model state
# (model.prompts_dict / model.model_prompts are mutated per-task, which is
# NOT safe to share across threads/processes).
_model = None
_model_meta = None
_cache = None


def _set_prompt_on_model(model, prompt_dict):
    """Set prompt on the model using the correct mechanism for its wrapper type.

    InstructSentenceTransformerModel uses `prompts_dict`:
      - Values are instruction text, formatted via the model's instruction_template.
      - Keys can be: task name, task type ("Classification", "Retrieval", ...),
        or prompt type ("query", "document").

    SentenceTransformerEncoderWrapper uses `model_prompts`:
      - Values are raw prefixes prepended directly to each input.
      - Keys follow the same priority: task name > task type > prompt type.
        Additionally supports compound keys like "{task_name}-{prompt_type}".
    """
    if isinstance(model, InstructSentenceTransformerModel):
        model.prompts_dict = prompt_dict
    elif hasattr(model, 'model_prompts'):
        model.model_prompts = prompt_dict
    else:
        raise ValueError(f"Unknown model wrapper type: {type(model)}. Cannot set prompts.")


def _build_prompt_dict(task, prompt_text):
    """Build a prompt dict keyed appropriately for the task type.

    For non-retrieval tasks (classification, clustering, STS), prompt_type is
    None, so we key on task_name — both wrappers resolve it correctly.

    For retrieval tasks, we key ONLY on query-specific keys to prevent the
    custom prompt from leaking into document/corpus encoding:
      - "{task_name}-query": resolved by SentenceTransformerEncoderWrapper
        via get_prompt_name() priority 1 (task_name + prompt_type).
      - "query": resolved by InstructSentenceTransformerModel via
        get_instruction() priority 3 (prompt_type.value lookup).
    Neither key matches during document encoding, so documents correctly
    fall back to the task default or receive no prompt.

    Args:
        task: The MTEB task object.
        prompt_text: The prompt/instruction string.

    Returns:
        A dict mapping the appropriate key(s) to the prompt text.
    """
    task_name = task.metadata.name

    if task.metadata.type == "Retrieval":
        # Query-only keys — document encoding won't match either of these.
        prompt_dict = {
            f"{task_name}-query": prompt_text,  # SentenceTransformerEncoderWrapper
            "query": prompt_text,               # InstructSentenceTransformerModel
        }
    else:
        # For symmetric tasks, prompt_type is None so task_name is safe.
        prompt_dict = {task_name: prompt_text}

    return prompt_dict


def run_task(model, model_meta, cache, taskname, prompt, tasktype, include_baseline_prompt=True, bs=32):
    """Run evaluation for a single task with optional baseline + custom prompt.

    Args:
        model: The already-loaded model object (loaded once per worker, reused
            across calls within that worker).
        model_meta: The original ModelMeta (used to restore baseline prompts).
        cache: The mteb.ResultCache instance to use for this run.
        taskname: Name of the MTEB task to evaluate.
        prompt: The custom prompt/instruction string.
        tasktype: Task type description (for logging only).
        include_baseline_prompt: If True, run baseline (no custom prompt) first.
        bs: Encoding batch size.
    """
    task = mteb.get_task(taskname, languages=['eng'], exclusive_language_filter=True)

    # Build the prompt dict with correct keys for the task type
    prompt_dict = _build_prompt_dict(task, prompt)

    # Construct the full set of experiments to run
    # None = baseline (no custom prompt), then the custom prompt experiment
    experiments = []
    if include_baseline_prompt:
        experiments.append(None)
    experiments.append(prompt_dict)

    for exp in experiments:
        # Reload task to avoid stale dataset state
        task = mteb.get_task(taskname, languages=['eng'], exclusive_language_filter=True)

        if exp is not None:
            # Set the custom prompt on the loaded model
            _set_prompt_on_model(model, exp)

            # Update experiment_kwargs on the model's meta so cache keys
            # differ between baseline and custom-prompt runs
            model.mteb_model_meta = model.mteb_model_meta.model_copy(
                update={"experiment_kwargs": {"prompts": exp}}
            )
        else:
            # Baseline: clear any custom prompts
            if isinstance(model, InstructSentenceTransformerModel):
                model.prompts_dict = None
            elif hasattr(model, 'model_prompts'):
                # Restore built-in prompts from the original model_meta loader_kwargs
                original_prompts = model_meta.loader_kwargs.get("model_prompts", None)
                model.model_prompts = original_prompts

            model.mteb_model_meta = model.mteb_model_meta.model_copy(
                update={"experiment_kwargs": None}
            )

        # Pass the loaded model, NOT the meta — otherwise evaluate() reloads
        # from scratch and our prompt assignments are lost
        mteb.evaluate(
            model, task, cache=cache, raise_error=True,
            encode_kwargs={"batch_size": bs}
        )


def _worker_init(modelname, gpu_id):
    """Runs once per worker process: pins the process to a GPU (if given) and
    loads the model a single time, so it's reused across every prompt this
    worker handles.
    """
    global _model, _model_meta, _cache

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logger.info(f"[Worker pid={os.getpid()}] Loading model {modelname} "
                f"(GPU={gpu_id if gpu_id is not None else 'default'})...")
    _model_meta = mteb.get_model_meta(modelname)
    _model = _model_meta.load_model()
    # Each worker gets its own ResultCache instance pointed at the same path.
    _cache = mteb.ResultCache(cache_path=CACHE_PATH)
    logger.info(f"[Worker pid={os.getpid()}] Model loaded: {type(_model).__name__}")


def _worker_run_prompt(prompt_dict, exclude_datasets, bs):
    """Runs in the worker process: evaluates a single prompt entry using the
    model that was loaded once in _worker_init.
    """
    global _model, _model_meta, _cache

    taskname = prompt_dict['task']
    if taskname in exclude_datasets:
        logger.info(f"[pid={os.getpid()}] [SKIPPED AS EXCLUDED] Task: {taskname}; "
                     f"Lang: {prompt_dict['language']}; "
                     f"Dataset Descr: {prompt_dict['dataset_description']}")
        return f"SKIPPED: {taskname}"

    logger.info(f"[pid={os.getpid()}] Task: {taskname}; Lang: {prompt_dict['language']}; "
                f"Dataset Descr: {prompt_dict['dataset_description']}")
    run_task(
        model=_model,
        model_meta=_model_meta,
        cache=_cache,
        taskname=taskname,
        prompt=prompt_dict['generated_prompt'],
        include_baseline_prompt=True,
        bs=bs,
        tasktype=prompt_dict['task_description'],
    )
    return f"DONE: {taskname}"


def _chunk(items, n_chunks):
    """Split `items` into n_chunks roughly-even, contiguous lists."""
    n_chunks = max(1, min(n_chunks, len(items))) if items else 0
    if n_chunks == 0:
        return []
    k, m = divmod(len(items), n_chunks)
    return [
        items[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
        for i in range(n_chunks)
    ]


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('modelname')
    arg_parser.add_argument('--prompts_path', default="openai-gpt-oss-120b-gen-prompts_p.json")
    # Accepts one or more task_description values to filter on, e.g.:
    #   --task_descriptions retrieval classification
    # If omitted, all prompts are used (same as before).
    arg_parser.add_argument('--task_descriptions', nargs='+', default=None)
    arg_parser.add_argument('--enc_batch_size', default=32, type=int)
    # Number of parallel worker processes (each loads its own model copy).
    # Defaults to the number of GPUs given via --gpus, or 1 if none given.
    arg_parser.add_argument('--num_workers', default=None, type=int)
    # Optional list of GPU ids to pin workers to, e.g. --gpus 0 1 2 3
    # If provided, num_workers defaults to len(gpus) and worker i is pinned
    # to gpus[i]. If omitted, workers run without explicit GPU pinning.
    arg_parser.add_argument('--gpus', nargs='+', default=None, type=int)

    args = arg_parser.parse_args()

    EXCLUDE_DATASETS = [
        "MasakhaNEWSClusteringS2S"
    ]

    with open(args.prompts_path, 'r') as f:
        prompts = json.load(f)

    if args.task_descriptions:
        wanted = set(args.task_descriptions)
        prompts = [p for p in prompts if p['task_description'] in wanted]
        logger.info(f"Filtered to task_descriptions={sorted(wanted)}: "
                    f"{len(prompts)} prompts remain.")

    if not prompts:
        logger.warning("No prompts to run after filtering. Exiting.")
        raise SystemExit(0)

    # Work out worker count / GPU assignment
    if args.gpus:
        num_workers = args.num_workers or len(args.gpus)
        gpu_ids = [args.gpus[i % len(args.gpus)] for i in range(num_workers)]
    else:
        num_workers = args.num_workers or 1
        gpu_ids = [None] * num_workers

    num_workers = min(num_workers, len(prompts))  # no point spawning idle workers
    gpu_ids = gpu_ids[:num_workers]

    logger.info(f"-------> Splitting {len(prompts)} prompts across {num_workers} "
                f"worker(s), GPUs={gpu_ids}")

    prompt_chunks = _chunk(prompts, num_workers)

    # 'spawn' start method is required/safest for CUDA in subprocesses.
    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    futures = []
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        # One future per worker; each future runs _worker_init once, then
        # iterates its full chunk of prompts sequentially inside that same
        # process (so the model is loaded exactly once per worker).
        def _run_chunk(modelname, gpu_id, chunk, exclude_datasets, bs):
            _worker_init(modelname, gpu_id)
            results = []
            for i, prompt_dict in enumerate(chunk):
                logger.info(f"[Worker pid={os.getpid()}] Prompt {i + 1}/{len(chunk)}")
                results.append(_worker_run_prompt(prompt_dict, exclude_datasets, bs))
            return results

        for gpu_id, chunk in zip(gpu_ids, prompt_chunks):
            if not chunk:
                continue
            futures.append(
                executor.submit(
                    _run_chunk, args.modelname, gpu_id, chunk, EXCLUDE_DATASETS, args.enc_batch_size
                )
            )

        for future in as_completed(futures):
            try:
                worker_results = future.result()
                logger.info(f"-------> Worker finished: {len(worker_results)} prompts processed.")
            except Exception:
                logger.exception("-------> A worker raised an exception:")
                raise

    logger.info("-------> All workers finished.")