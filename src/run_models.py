import logging
from pathlib import Path
import mteb

# setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

cache_path = Path( "cache__data") / "mteb_cache"
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
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('modelname')
    arg_parser.add_argument('--prompts_path', default="openai-gpt-oss-120b-gen-prompts.json")

    args = arg_parser.parse_args()

    with open(args.prompts_path, 'r') as f:
        prompts = json.load(f)
    
    logger.info(f"-------> Running {args.modelname}...")
    for i, prompt_dict in enumerate(prompts):
        logger.info(f"-------> Running {args.modelname} | Prompt {i+1} / {len(prompts)}...")
        logger.info(f"Task: {prompt_dict['task']}; Lang: {prompt_dict['language']}; Dataset Descr: {prompt_dict['dataset_description']}")

        run_task(
                taskname=prompt_dict['task'],
                modelname=args.modelname,
                prompt=prompt_dict['generated_prompt'],
                include_baseline_prompt=True
        )

