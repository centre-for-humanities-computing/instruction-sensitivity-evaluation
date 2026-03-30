from pydantic import BaseModel
import json
import dirtyjson
from typing import Union, Dict, List

class PromptTemplateOutput(BaseModel):
    prompts: list[str]


PROMPT_TEMPLATE_ = """You are a professional prompt engineer for embedding models. You are tasked with the writing instructions/prompts for the embedding models that encode the input text along with the prompt.

You are provided with the **Context**, where the information is provided about the objective of the embedding model, language (if available), and dataset specifics (if available).
Based on the provided context, you need to generate instructions/prompts.


# Generation Instructions
- Each prompt will be added at the beginning of the text. For example, "Query", "Passage", "Summarize the news article about dogs" etc.
- Use ONLY provided information from the **Contenxt**. DO NOT add any other information outside of it.
- Use ALL the provide information for the prompt if available: language (if provided, e.g. "Query in English"), dataset specifics (if provided, e.g. "Encode Tweets in English"). If language or dataset specifics are not provided - DO NOT add anything outise the Context! Use ONLY what is provided!
- You need to come up with a list of diverse prompts that will be used as an input to the instruction-based embedding models (such as E5, Qwen3-Embedding, Instructor, NV_Embed and others).
- DO NOT generate duplicates!
- Generate a list of {n} prompts
- Follow the provided format below
- Each prompt should be in English!

# Format
Generate prompts in the following format:

{{
    "prompts": [str, ..., str] // each str is a generated prompt
}}

Output ONLY json, nothing else!

# Context

Task of the embedding model: {task_description}
Language: {language}
Dataset description: {dataset_description}"""


def generate_task_consistency_section(task_description: str,
                                      language: str = "",
                                      dataset_description: str = "") -> str:
    """
    Generate a TASK CONSISTENCY RULE section for prompt generation.

    Args:
        task_description (str): One of ["retrieval", "classification",
                                       "semantic-similarity", "clustering"]
        language (str): Optional language (e.g., "English")
        dataset_description (str): Optional dataset description

    Returns:
        str: Formatted rule section
    """

    task = task_description.lower().strip()

    # --- Define valid / invalid examples per task ---
    examples = {
        "retrieval": {
            "valid": [
                "Query for retrieval:",
                "Retrieve relevant text:",
                "Encode for retrieval:",
                "Search query representation:",
                "Text for retrieval task:"
            ],
            "invalid": [
                "Summarize the following text:",
                "Extract key entities:",
                "Classify the topic:",
                "Is this about climate change?",
                "Find information about dogs:"
            ]
        },
        "classification": {
            "valid": [
                "Classify this text:",
                "Text classification input:",
                "Assign a label to this text:",
                "Category prediction:",
                "Label this example:"
            ],
            "invalid": [
                "Summarize the text:",
                "Retrieve relevant documents:",
                "Compare similarity between texts:",
                "Cluster these sentences:",
                "Search query:"
            ]
        },
        "semantic-similarity": {
            "valid": [
                "Compare semantic similarity:",
                "Similarity between texts:",
                "Semantic matching input:",
                "Text pair similarity:",
                "Measure similarity:"
            ],
            "invalid": [
                "Classify this text:",
                "Summarize the document:",
                "Retrieve relevant passages:",
                "Cluster these texts:",
                "Extract entities:"
            ]
        },
        "clustering": {
            "valid": [
                "Group similar texts:",
                "Cluster this text:",
                "Text clustering input:",
                "Assign to a cluster:",
                "Cluster representation:"
            ],
            "invalid": [
                "Classify this text:",
                "Retrieve relevant documents:",
                "Compare similarity score:",
                "Summarize the text:",
                "Answer the question:"
            ]
        }
    }

    if task not in examples:
        raise ValueError(f"Unsupported task: {task_description}")

    valid_examples = examples[task]["valid"]
    invalid_examples = examples[task]["invalid"]

    # --- Language constraint ---
    language_rule = ""
    if language.strip():
        language_rule = f"""
### LANGUAGE CONSTRAINT

- All prompts MUST explicitly specify the language: "{language}"
- The language must be clearly mentioned in each prompt (e.g., "in {language}")
"""

    # --- Dataset grounding (optional, soft constraint) ---
    dataset_rule = ""
    if dataset_description.strip():
        dataset_rule = f"""
### DATASET GROUNDING

- Prompts should reflect the dataset context when possible
- Use only information explicitly present: "{dataset_description}"
- Do NOT introduce new domains or entities beyond this description
"""

    # --- Format examples ---
    def format_list(items):
        return "\n".join([f"- \"{x}\"" for x in items])

    section = f"""
## TASK CONSISTENCY RULE (CRITICAL)

All generated prompts must express the SAME task as defined in the Context.

- The task is: "{task}"
- Prompts must ONLY correspond to this task
- Prompts must NOT introduce other NLP tasks or new semantics

### VALID PROMPTS (correct behavior)

{format_list(valid_examples)}

### INVALID PROMPTS (forbidden)

❌ Task change or semantic drift:
{format_list(invalid_examples)}

### HARD CONSTRAINT

Each prompt MUST:
- Preserve the original task ("{task}")
- Be a rephrasing, NOT a reinterpretation
- Avoid introducing new tasks, domains, or entities
- Remain consistent with the provided Context

If a prompt cannot be generated without adding new information → DO NOT generate it.
"""

    return section + language_rule + dataset_rule



PROMPT_TEMPLATE = """You are generating instruction prefixes for embedding models.

These prefixes will be prepended to input text before encoding. Your goal is to create controlled variations of such prefixes.

---

## STRICT RULES

1. Use ONLY information explicitly present in the Context.
2. DO NOT introduce new tasks, domains, or entities.
3. If information is missing (e.g., language or dataset), DO NOT guess or infer it.
4. Every prompt MUST be grounded in the provided task description.
5. Prompts must differ in wording, perspective, or format - NOT in meaning.
6. Each prompt must be a prefix (not a full sentence explanation).
7. No duplicates or near-duplicates.


Carefully read the Context section. Analyze, what information is available to you.
It is CRUCIAL that you will not infer anything that is not in the Context.

HARD CONSTRAINT:
If any field in the Context is empty or missing, you MUST ignore it completely.
Do NOT infer, assume, or complete missing information.
Any violation makes the output invalid.

## PROMPT TYPES
You MUST vary across these styles:

* Label-style (e.g., "Query:", "Passage:")
* Instruction-style (e.g., "Encode this query for retrieval:")
* Task-aware (explicitly referencing the task)
* Minimal semantic hints (short but meaningful)
* Reformulations (same meaning, different phrasing)

---

## COUNT CONSTRAINT

Generate EXACTLY {n} prompts.

---

{add_rules}

---

Before output:

* Count the prompts
* If not exactly {n}, fix the list

---

## OUTPUT FORMAT (STRICT JSON)

{{
"prompts": [
"...",
"...",
...
]
}}

No explanations. No extra text.

---

## CONTEXT

Task: {task_description}
Language: {language}
Dataset: {dataset_description}"""

def prepare_task_description(task_description, language):

    section_prompt = f"""# Objective for which you need to write a prompt\n{task_description}"""
    if language:
        section_prompt += f"\nThis dataset for the objective is in the following language: {language}"
    return section_prompt

def prep_dataset_description(dataset_description: str) -> str:
    return f"""# Dataset Description\nYou need to prepare prompts for the following dataset:\n{dataset_description}"""

def prepare_prompt_template(
        n:int,
        task_description: str, 
    dataset_description: str,
    language: str,
**kwargs
    ):
    
   # if task_description:
   #     task_description = prepare_task_description(task_description, language)
   # if dataset_description:
   #     dataset_description = prep_dataset_description(dataset_description)
    
    add_rules = generate_task_consistency_section(
        task_description=task_description,
        language=language,
        dataset_description=dataset_description
    )
    return PROMPT_TEMPLATE.format(
            n=n,
        task_description=task_description,
        dataset_description=dataset_description,
        language=language,
        add_rules=add_rules
    )



def load_json(content):
    try:
        return json.loads(content)
    except:
        try:
            return dirtyjson.loads(content)
        except:
            return {}


def validate_output(str_gen_output):

    if "assistantfinal" in str_gen_output:
        str_gen_output = str_gen_output.split('assistantfinal')[1].strip()

    loaded_outputs = load_json(str_gen_output)
    if not isinstance(loaded_outputs, dict):
        return {}, False

    if "prompts" in loaded_outputs:
        if not len(loaded_outputs["prompts"]):
            return {}, False
        for sample in loaded_outputs["prompts"]:
            if not isinstance(sample, str):
                return loaded_outputs, False
    else:
        return {}, False
    return loaded_outputs, True
