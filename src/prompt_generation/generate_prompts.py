import json
from vllm.sampling_params import StructuredOutputsParams
from vllm import LLM, SamplingParams
import pandas as pd

class VLLMOfflineAnswerGenerator:
    def __init__(self,
                 model_name,
                 format_issue_attempts=10,
                 pipeline_parallel_size=1,
                 tensor_parallel_size=1,
                 gpu_memory_utilization=0.95,
                 max_model_len=None,
                 dtype='auto',
                 trust_remote_code=True,
                 max_num_seqs=256
                 ):
        self.model_name = model_name
        self.format_issue_attempts = format_issue_attempts

        self.llm = LLM(
            model=model_name,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            max_num_seqs=max_num_seqs,
        )

    def process_batch(self, prompts, format_basemodel, max_tokens):
        json_schema = format_basemodel.model_json_schema()
        guided_decoding_params = StructuredOutputsParams(json=json_schema)
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            structured_outputs=guided_decoding_params
        )

        outputs = self.llm.generate(prompts, sampling_params)
        outputs_texts = [output.outputs[0].text for output in outputs]
        return outputs_texts

    def get_issues(self, outputs, validator_func):

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

    def order_outputs(self, ids, parsed_outputs, N):
        ordered_parsed_outputs = [None for _ in range(N)]
        for idx, parsed_output in zip(ids, parsed_outputs):
            ordered_parsed_outputs[idx] = parsed_output
        return ordered_parsed_outputs

    def process_batch_fix_format_issues(self, prompts, format_basemodel, validator_func, max_tokens=None):
        outputs = self.process_batch(prompts, format_basemodel, max_tokens)
        issue_idx, no_issue_idx, parsed_outputs = self.get_issues(outputs, validator_func)

        attempt = 1
        while attempt <= self.format_issue_attempts and len(issue_idx) > 0:
            print(f"{self.model_name}; Attempt {attempt}; Issues: {len(issue_idx)}...")
            issue_prompts = [prompts[i] for i in issue_idx]
            issue_outputs = self.process_batch(issue_prompts, format_basemodel, max_tokens)

            # Track which issues were fixed
            newly_fixed = []
            for local_idx, (original_idx, issue_output) in enumerate(zip(issue_idx, issue_outputs)):
                parsed_issue_output, is_valid = validator_func(issue_output)
                if is_valid:
                    parsed_outputs.append(parsed_issue_output)
                    no_issue_idx.append(original_idx)
                    newly_fixed.append(original_idx)

            # Remove fixed issues
            issue_idx = [idx for idx in issue_idx if idx not in newly_fixed]
            attempt += 1

        return self.order_outputs(no_issue_idx, parsed_outputs, len(prompts))

def load_prompts_for_task(json_path):
    with open(json_path, 'r') as f:
        prompts = json.load(f)

    prompts_df = pd.DataFrame(prompts)
    prompts_df = prompts_df.explode('task_mteb_names')


    run_prompts = []
    for idx, row in prompts_df.iterrows():
        run_prompts += row['generated_prompts']['prompts']
    return run_prompts

def load_prompts_for_task(json_path, filter_on_task):
    with open(json_path, 'r') as f:
        prompts = json.load(f)

    prompts_df = pd.DataFrame(prompts)
    prompts_df = prompts_df.explode('task_mteb_names')

    if filter_on_task:
        prompts_df = prompts_df[prompts_df['task_mteb_names'] == filter_on_task]

    run_prompts = []
    for idx, row in prompts_df.iterrows():
        run_prompts += row['generated_prompts']['prompts']
    return run_prompts


if __name__ == '__main__':
    from prompt_template import prepare_prompt_template, PromptTemplateOutput, validate_output
    import mteb
    from pathlib import Path

    savedir_path = Path(__file__).parent.parent.parent / "data" / "prompts"


    MODELNAME = "openai/gpt-oss-120b"
#    MODELNAME = "google/gemma-3-27b-it"
#    MODELNAME = 'Qwen/Qwen3.5-27B'

    LANGUAGE = "eng"

    savefilename = f'{MODELNAME.replace("/", "-")}-gen-prompts.json'
    savepath = savedir_path / savefilename

    N = 5

    # load tasks and datasets
    tasks = {
        "retrieval": [
            "MIRACLRetrievalHardNegatives.v2", # multilingual, in mmteb
              "Touche2020Retrieval.v3", # in mteb(eng, v2)
               "FEVERHardNegatives" # in mteb(eng, v2)
            ], 
        "classification": [
              "TweetSentimentClassification", # in MTEB europe
              "ImdbClassification", # mteb eng
              "AmazonCounterfactualClassification" # mteb eng
            ], 
        "semantic-similarity": [  
              "STS14", # in mteb eng
              "STS15", # in mteb eng
              "STS22.v2", # in mteb eng (and mmteb?)
            ],
        "clustering": [  
              "MedrxivClusteringP2P.v2", # in mteb eng
              "StackExchangeClustering.v2", # in mteb eng
              "MasakhaNEWSClusteringS2S", # in mmteb
            ]
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
            "language": ""

        }
        prompts.append(
            prepare_prompt_template(
                **metadata
            )
        )
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
                "language": ""

            }
            prompts.append(
                prepare_prompt_template(
                    **metadata
                )
            )
            prompts_metadata.append(metadata)
            # ---------------------------------------------

            # Adding prompts for the task + data description + langs
            # ---------------------------------------------
            metadata = {
                    "n": N,
                "task_mteb_names": [task_mteb_name],
                "task_description": tasktype_name,
                "dataset_description": dataset_description,
                "language": LANGUAGE

            }
            prompts.append(
                prepare_prompt_template(
                    **metadata
                )
            )
            prompts_metadata.append(metadata)
            # ---------------------------------------------


    print(">> Total num prompts: ", len(prompts))

    processor = VLLMOfflineAnswerGenerator(
        model_name=MODELNAME,
        format_issue_attempts=100,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=40_000,
        dtype='auto',
        trust_remote_code=True,
        max_num_seqs=256
    )

    outputs = processor.process_batch_fix_format_issues(
        prompts=prompts,
        format_basemodel=PromptTemplateOutput,
        validator_func=validate_output,
        max_tokens=None
    )

    for i in range(len(outputs)):
        prompts_metadata[i]['generated_prompts'] = outputs[i]
        prompts_metadata[i]['input_prompt'] = prompts[i]


    # saving to the readable format
    df = pd.DataFrame(prompts_metadata)
    df['generated_prompts'] = df['generated_prompts'].apply(lambda x: x['prompts'])
    df = df.explode('task_mteb_names')
    df =df.explode('generated_prompts')
    df = df.rename(columns={
        "task_mteb_names": "task",
        "generated_prompts": "generated_prompt"
    })

    df['generation_meta'] = df['input_prompt'].apply(lambda x: {'prompt': x})
    df = df.drop(columns=['input_prompt'])

    
    with open(savepath, 'w') as f:
        json.dump(df.to_dict('records'), f)
