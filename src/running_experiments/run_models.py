import logging
from pathlib import Path
import mteb
from mteb.models.instruct_wrapper import InstructSentenceTransformerModel

# setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

cache_path = Path("cache_data") / "mteb_cache"
cache = mteb.ResultCache(cache_path=cache_path)


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


def run_task(model, model_meta, taskname, prompt, tasktype, include_baseline_prompt=True, bs=32):
    """Run evaluation for a single task with optional baseline + custom prompt.

    Args:
        model: The already-loaded model object (loaded once, reused across calls).
        model_meta: The original ModelMeta (used to restore baseline prompts).
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

    # For InstructSentenceTransformerModel: the generated prompts are raw
    # prefixes (e.g. "Query for retrieval:"), but get_task_instruction()
    # wraps them in instruction_template (e.g. "Instruct: {instruction}\nQuery:").
    # Save the original template so we can disable it for custom-prompt
    # experiments and restore it for baseline runs.
    _orig_instruction_template = getattr(model, 'instruction_template', None)

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
        results = mteb.evaluate(
            model, task, cache=cache, raise_error=True,
            encode_kwargs={"batch_size": bs}
        )


if __name__ == '__main__':
    import json
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('modelname')
    arg_parser.add_argument('--prompts_path', default="openai-gpt-oss-120b-gen-prompts_p.json")
    arg_parser.add_argument('--task_description', default=None)
    arg_parser.add_argument('--enc_batch_size', default=32, type=int)

    args = arg_parser.parse_args()

    EXCLUDE_DATASETS = []

    with open(args.prompts_path, 'r') as f:
        prompts = json.load(f)

    if args.task_description:
        prompts = [prompt_dict for prompt_dict in prompts if prompt_dict['task_description'] == args.task_description]

    # Load model ONCE — reuse across all prompt/task iterations
    logger.info(f"-------> Loading model {args.modelname} (one-time load)...")
    model_meta = mteb.get_model_meta(args.modelname)
    model = model_meta.load_model()
    logger.info(f"-------> Model loaded: {type(model).__name__}")

    logger.info(f"-------> Running {args.modelname}...")
    for i, prompt_dict in enumerate(prompts):
        logger.info(f"-------> Running {args.modelname} | Prompt {i+1} / {len(prompts)}...")
        if prompt_dict['task'] not in EXCLUDE_DATASETS:
            logger.info(f"Task: {prompt_dict['task']}; Lang: {prompt_dict['language']}; Dataset Descr: {prompt_dict['dataset_description']}")
            run_task(
                model=model,
                model_meta=model_meta,
                taskname=prompt_dict['task'],
                prompt=prompt_dict['generated_prompt'],
                include_baseline_prompt=True,
                bs=args.enc_batch_size,
                tasktype=prompt_dict['task_description']
            )
        else:
            logger.info(f"[SKIPPED AS EXCLUDED] Task: {prompt_dict['task']}; Lang: {prompt_dict['language']}; Dataset Descr: {prompt_dict['dataset_description']}")
