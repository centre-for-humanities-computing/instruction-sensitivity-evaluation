import logging
from pathlib import Path
import mteb

# setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

cache_path = Path(__file__).parent.parent / "data" / "mteb_cache"
cache = mteb.ResultCache(cache_path=cache_path)


def run_task(taskname, modelname, prompt, include_baseline_prompt=True):

    task = mteb.get_task(taskname, languages=['eng'])

    formatted_task_name = task.metadata.name
    if task.metadata.type == "Retrieval":
        formatted_task_name = f"{task.metadata.name}-query"

    # construct the full set of experiments to run
    experiments = [
        {"prompts": {formatted_task_name: prompt}}
    ]

    if include_baseline_prompt:
        experiments.insert(0, None)


    for exp in experiments:
        meta = mteb.get_model_meta(
            modelname, experiment_kwargs=exp
        )  # doesn't work currently but is fixed in https://github.com/embeddings-benchmark/mteb/pull/4308
        results = mteb.evaluate(meta, task, cache=cache, raise_error=False)


if __name__ == '__main__':
    import json

    MODELS = [
        # "intfloat/multilingual-e5-large-instruct", # highly popular
        # "Qwen/Qwen3-Embedding-0.6B", # vary across size, current sota
        # "Qwen/Qwen3-Embedding-4B",
        # "hkunlp/instructor-large", # the original instruct model, only English
        # "nvidia/NV-Embed-v2" # only English
        # "BAAI/bge-base-en",
        "BAAI/bge-base-en-v1.5"
    ]

    prompts_path = Path(__file__).parent.parent / "data" / "prompts" /"openai-gpt-oss-120b-gen-prompts.json"
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)

    for modelname in MODELS:
        logger.info(f"-------> Running {modelname}...")
        for i, prompt_dict in enumerate(prompts):
            logger.info(f"-------> Running {modelname} | Prompt {i+1} / {len(prompts)}...")

            run_task(
                taskname=prompt_dict['task'],
                modelname=modelname,
                prompt=prompt_dict['generated_prompt'],
                include_baseline_prompt=True
            )



