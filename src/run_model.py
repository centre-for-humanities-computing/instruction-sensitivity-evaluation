import logging
from pathlib import Path

import mteb

taskname = "LccSentimentClassification"  # just a small test task
prompt = "thatasdfgggsdsd"
modelname = "intfloat/multilingual-e5-large-instruct"


meta = mteb.get_model_meta(modelname, experiment_kwargs=exp)
mdl = meta.load_model()
taskname in mdl.model.prompts  # False
mdl = mteb.get_model(modelname, prompts={taskname: prompt})
taskname in mdl.model.prompts  # true


raise ValueError("stop")
# setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


# sample grid entry
# taskname = "WikiClusteringP2P.v2"
# taskname = "FEVERHardNegatives"
taskname = "LccSentimentClassification"  # just a small test task
prompt = "thatasdfgggsdsd"
modelname = "intfloat/multilingual-e5-large-instruct"


cache_path = Path(__file__).parent.parent / "data" / "mteb_cache"
cache = mteb.ResultCache(cache_path=cache_path)

task = mteb.get_task(taskname)

formatted_task_name = task.metadata.name
if task.metadata.type == "Retrieval":
    formatted_task_name = f"{task.metadata.name}-query"

experiments = [
    None,
    {"prompts": {taskname: prompt}},
]  # construct the full set of experiments to run


for exp in experiments:
    meta = mteb.get_model_meta(modelname, experiment_kwargs=exp)
    results = mteb.evaluate(meta, task, cache=cache, raise_error=False)


# load results:
df = cache.load_results(models=[modelname], load_experiments="match_name")
