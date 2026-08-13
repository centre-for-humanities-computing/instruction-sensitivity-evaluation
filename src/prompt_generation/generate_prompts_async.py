import asyncio
import json
from typing import Callable, List, Optional, Type

import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel


class OpenAIAnswerGenerator:
    """
    Drop-in replacement for the vLLM offline generator that uses an
    OpenAI-compatible chat completions endpoint with concurrent requests
    instead of vLLM's offline batch engine.

    Works against:
      - the official OpenAI API (leave base_url=None, pass a real api_key)
      - a self-hosted OpenAI-compatible server, e.g. `vllm serve <model>
        --enable-auto-tool-choice ...` (set base_url="http://host:port/v1")
    """

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",
        format_issue_attempts: int = 10,
        max_concurrent_requests: int = 32,
        request_timeout: float|None= 600.0,
        max_retries: int = 2,
    ):
        self.model_name = model_name
        self.format_issue_attempts = format_issue_attempts
        self.max_concurrent_requests = max_concurrent_requests

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=request_timeout,
            max_retries=max_retries,
        )
        # Created lazily inside the running event loop that actually issues
        # the requests, so this class stays safe to reuse across asyncio.run() calls.
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        return self._semaphore

    async def _generate_one(self, prompt: str, format_basemodel: Type[BaseModel], max_tokens: Optional[int]) -> str:
        json_schema = format_basemodel.model_json_schema()
        semaphore = self._get_semaphore()

        async with semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": format_basemodel.__name__,
                            "schema": json_schema,
                            "strict": True,
                        },
                    },
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                print(f"{self.model_name}; request failed: {e}")
                return ""

    async def process_batch_async(
        self, prompts: List[str], format_basemodel: Type[BaseModel], max_tokens: Optional[int]
    ) -> List[str]:
        tasks = [self._generate_one(prompt, format_basemodel, max_tokens) for prompt in prompts]
        return await asyncio.gather(*tasks)

    def process_batch(self, prompts: List[str], format_basemodel: Type[BaseModel], max_tokens: Optional[int]) -> List[str]:
        return asyncio.run(self.process_batch_async(prompts, format_basemodel, max_tokens))

    def get_issues(self, outputs: List[str], validator_func: Callable):
        issue_idx = []
        no_issue_idx = []
        parsed_outputs = []
        for i, output in enumerate(outputs):
            parsed_output, is_valid = validator_func(output)
            if not is_valid:
                issue_idx.append(i)
            else:
                no_issue_idx.append(i)
                parsed_outputs.append(parsed_output)

        return issue_idx, no_issue_idx, parsed_outputs

    def order_outputs(self, ids: List[int], parsed_outputs: List, N: int) -> List:
        ordered_parsed_outputs = [None for _ in range(N)]
        for idx, parsed_output in zip(ids, parsed_outputs):
            ordered_parsed_outputs[idx] = parsed_output
        return ordered_parsed_outputs

    async def process_batch_fix_format_issues_async(
        self,
        prompts: List[str],
        format_basemodel: Type[BaseModel],
        validator_func: Callable,
        max_tokens: Optional[int] = None,
    ) -> List:
        outputs = await self.process_batch_async(prompts, format_basemodel, max_tokens)
        issue_idx, no_issue_idx, parsed_outputs = self.get_issues(outputs, validator_func)

        attempt = 1
        while attempt <= self.format_issue_attempts and len(issue_idx) > 0:
            print(f"{self.model_name}; Attempt {attempt}; Issues: {len(issue_idx)}...")
            issue_prompts = [prompts[i] for i in issue_idx]
            issue_outputs = await self.process_batch_async(issue_prompts, format_basemodel, max_tokens)

            # Track which issues were fixed
            newly_fixed = []
            for original_idx, issue_output in zip(issue_idx, issue_outputs):
                parsed_issue_output, is_valid = validator_func(issue_output)
                if is_valid:
                    parsed_outputs.append(parsed_issue_output)
                    no_issue_idx.append(original_idx)
                    newly_fixed.append(original_idx)

            # Remove fixed issues
            issue_idx = [idx for idx in issue_idx if idx not in newly_fixed]
            attempt += 1

        return self.order_outputs(no_issue_idx, parsed_outputs, len(prompts))

    def process_batch_fix_format_issues(
        self,
        prompts: List[str],
        format_basemodel: Type[BaseModel],
        validator_func: Callable,
        max_tokens: Optional[int] = None,
    ) -> List:
        return asyncio.run(
            self.process_batch_fix_format_issues_async(prompts, format_basemodel, validator_func, max_tokens)
        )


def load_prompts_for_task(json_path, filter_on_task=None):
    with open(json_path, "r") as f:
        prompts = json.load(f)

    prompts_df = pd.DataFrame(prompts)
    prompts_df = prompts_df.explode("task_mteb_names")

    if filter_on_task:
        prompts_df = prompts_df[prompts_df["task_mteb_names"] == filter_on_task]

    run_prompts = []
    for idx, row in prompts_df.iterrows():
        run_prompts += row["generated_prompts"]["prompts"]
    return run_prompts


if __name__ == "__main__":
    from pathlib import Path
    from dotenv import load_dotenv

    import mteb ,os

    from prompt_template import PromptTemplateOutput, prepare_prompt_template, validate_output

    savedir_path = Path(__file__).parent.parent.parent / "data" / "prompts"
    load_dotenv()

    # MODELNAME = "openai/gpt-oss-120b"
    # MODELNAME = "google/gemma-4-26B-A4B-it"
    MODELNAME = "Qwen/Qwen3.6-35B-A3B"

    # If MODELNAME is served locally (e.g. via `vllm serve`), point BASE_URL
    # at that server's OpenAI-compatible endpoint. Leave as None to hit the
    # real OpenAI API instead (in which case MODELNAME must be an OpenAI model).
    BASE_URL = os.getenv("LLM_BASE_URL")
    API_KEY = os.getenv("LLM_ENDPOINT_KEY")

    LANGUAGE = "eng"

    savefilename = f'{MODELNAME.replace("/", "-")}-gen-prompts.json'
    savepath = savedir_path / savefilename

    N = 5

    # load tasks and datasets
    tasks = {
        "retrieval": [
            "MIRACLRetrievalHardNegatives.v2",  # multilingual, in mmteb
            "Touche2020Retrieval.v3",  # in mteb(eng, v2)
            "FEVERHardNegatives",  # in mteb(eng, v2)
        ],
        "classification": [
            "TweetSentimentClassification",  # in MTEB europe
            "ImdbClassification",  # mteb eng
            "AmazonCounterfactualClassification",  # mteb eng
        ],
        "semantic-similarity": [
            "STS14",  # in mteb eng
            "STS15",  # in mteb eng
            "STS22.v2",  # in mteb eng (and mmteb?)
        ],
        "clustering": [
            "MedrxivClusteringP2P.v2",  # in mteb eng
            "StackExchangeClustering.v2",  # in mteb eng
            "MasakhaNEWSClusteringS2S",  # in mmteb
        ],
    }

    prompts = []
    prompts_metadata = []
    for tasktype_name, task_mteb_names in tasks.items():

        # Adding prompts for the task only
        # ---------------------------------------------
        metadata = {
            "n": N,
            "task_mteb_names": task_mteb_names,
            "task_description": tasktype_name,
            "dataset_description": "",
            "language": "",
        }
        prompts.append(prepare_prompt_template(**metadata))
        prompts_metadata.append(metadata)
        # ---------------------------------------------

        for task_mteb_name in task_mteb_names:
            task = mteb.get_task(task_mteb_name, languages=[LANGUAGE])

            dataset_description = task.metadata.description

            # Adding prompts for the task + data description
            # ---------------------------------------------
            metadata = {
                "n": N,
                "task_mteb_names": [task_mteb_name],
                "task_description": tasktype_name,
                "dataset_description": dataset_description,
                "language": "",
            }
            prompts.append(prepare_prompt_template(**metadata))
            prompts_metadata.append(metadata)
            # ---------------------------------------------

            # Adding prompts for the task + data description + langs
            # ---------------------------------------------
            metadata = {
                "n": N,
                "task_mteb_names": [task_mteb_name],
                "task_description": tasktype_name,
                "dataset_description": dataset_description,
                "language": LANGUAGE,
            }
            prompts.append(prepare_prompt_template(**metadata))
            prompts_metadata.append(metadata)
            # ---------------------------------------------

    print(">> Total num prompts: ", len(prompts))

    processor = OpenAIAnswerGenerator(
        model_name=MODELNAME,
        base_url=BASE_URL,
        api_key=API_KEY,
        request_timeout=None,
        format_issue_attempts=100,
        max_concurrent_requests=128,
    )

    outputs = processor.process_batch_fix_format_issues(
        prompts=prompts,
        format_basemodel=PromptTemplateOutput,
        validator_func=validate_output,
        max_tokens=None,
    )

    for i in range(len(outputs)):
        prompts_metadata[i]["generated_prompts"] = outputs[i]
        prompts_metadata[i]["input_prompt"] = prompts[i]

    # saving to the readable format
    df = pd.DataFrame(prompts_metadata)
    df["generated_prompts"] = df["generated_prompts"].apply(lambda x: x["prompts"])
    df = df.explode("task_mteb_names")
    df = df.explode("generated_prompts")
    df = df.rename(columns={"task_mteb_names": "task", "generated_prompts": "generated_prompt"})

    df["generation_meta"] = df["input_prompt"].apply(lambda x: {"prompt": x})
    df = df.drop(columns=["input_prompt"])

    with open(savepath, "w") as f:
        json.dump(df.to_dict("records"), f)