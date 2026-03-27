---
format:
  html:
    code-fold: true
jupyter: python3
bibliography: references.bib
---

# Prompt-hacking MTEB: On the issues and benefits of instruct-tuned models


# Introduction 
In recent years embeddings models have seen multiple improvements from compute and storage  optimizations such low precision and matroyoska embeddings (cite) to performance improvements such as prompt-based embeddings (cite). 


Prompt-based embeddings allow for a task-specific adaption of the embedding space to the task at hand using instructions such as “{example query from retrieval}”. 

Footnote: certain models, such as the model by Jina (cite), uses task-specific adapters instead of a prompt-prefix, but can generally be considered within the same category. 

Prompt-based adaption can be divided into two categories, early adaption, where the instruction is applied during inference (e.g.., as seen in CITE) and late adaption, where the instruction is applied after inference (cite).

**Figure**: Comparison of instruct-tuned vs non-instruct tuned models. Here we use the two multilingual e5 large models which are generally comparable with the exception of trained using prompt tuning.



**Problem** is that it is possible to tune these prompt such that the model performs better on the test set than what would be expected on new data.
Thus making it possible to overfit to the benchmark.

**goal** Our goal is to explore the extent to which you can tune to prompt
to improve performance on the benchmark and then explore if current models
are overfit to benchmars

---


# Related works
see more here:
https://github.com/KennethEnevoldsen/prompt-hacking/issues/2


# Methodology

## Tasks to evaluate on

```json
{
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
  "WikiClusteringP2P.v2", # in mmteb
]
}
```

## model selection
We select models that vary across providers, size and uses free-form instructions (potentially an ablation without free-form instructions). We choose models that are both designed for English and multilingual inputs.

```py
intfloat/multilingual-e5-large-instruct # highly popular
Qwen/Qwen3-Embedding-0.6B # vary across size, current sota
Qwen/Qwen3-Embedding-4B
hkunlp/instructor-large # the original instruct model, only English
nvidia/NV-Embed-v2 # only English

# potentially:
google/embeddinggemma-300m # uses prompt templates
```

Papers are 
```bibtex
@article{qwen3embedding,
  title={Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models},
  author={Zhang, Yanzhao and Li, Mingxin and Long, Dingkun and Zhang, Xin and Lin, Huan and Yang, Baosong and Xie, Pengjun and Yang, An and Liu, Dayiheng and Lin, Junyang and Huang, Fei and Zhou, Jingren},
  journal={arXiv preprint arXiv:2506.05176},
  year={2025}
}
@article{wang2024multilingual,
  title={Multilingual E5 Text Embeddings: A Technical Report},
  author={Wang, Liang and Yang, Nan and Huang, Xiaolong and Yang, Linjun and Majumder, Rangan and Wei, Furu},
  journal={arXiv preprint arXiv:2402.05672},
  year={2024}
}
@misc{vera2025embeddinggemmapowerfullightweighttext,
      title={EmbeddingGemma: Powerful and Lightweight Text Representations}, 
      author={Henrique Schechter Vera and Sahil Dua and Biao Zhang and Daniel Salz and Ryan Mullins and Sindhu Raghuram Panyam and Sara Smoot and Iftekhar Naim and Joe Zou and Feiyang Chen and Daniel Cer and Alice Lisak and Min Choi and Lucas Gonzalez and Omar Sanseviero and Glenn Cameron and Ian Ballantyne and Kat Black and Kaifeng Chen and Weiyi Wang and Zhe Li and Gus Martins and Jinhyuk Lee and Mark Sherwood and Juyeong Ji and Renjie Wu and Jingxiao Zheng and Jyotinder Singh and Abheesht Sharma and Divyashree Sreepathihalli and Aashi Jain and Adham Elarabawy and AJ Co and Andreas Doumanoglou and Babak Samari and Ben Hora and Brian Potetz and Dahun Kim and Enrique Alfonseca and Fedor Moiseev and Feng Han and Frank Palma Gomez and Gustavo Hernández Ábrego and Hesen Zhang and Hui Hui and Jay Han and Karan Gill and Ke Chen and Koert Chen and Madhuri Shanbhogue and Michael Boratko and Paul Suganthan and Sai Meher Karthik Duddu and Sandeep Mariserla and Setareh Ariafar and Shanfeng Zhang and Shijie Zhang and Simon Baumgartner and Sonam Goenka and Steve Qiu and Tanmaya Dabral and Trevor Walker and Vikram Rao and Waleed Khawaja and Wenlei Zhou and Xiaoqi Ren and Ye Xia and Yichang Chen and Yi-Ting Chen and Zhe Dong and Zhongli Ding and Francesco Visin and Gaël Liu and Jiageng Zhang and Kathleen Kenealy and Michelle Casbon and Ravin Kumar and Thomas Mesnard and Zach Gleicher and Cormac Brick and Olivier Lacombe and Adam Roberts and Qin Yin and Yunhsuan Sung and Raphael Hoffmann and Tris Warkentin and Armand Joulin and Tom Duerig and Mojtaba Seyedhosseini},
      year={2025},
      eprint={2509.20354},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.20354}, 
}
@misc{su2023embeddertaskinstructionfinetunedtext,
      title={One Embedder, Any Task: Instruction-Finetuned Text Embeddings}, 
      author={Hongjin Su and Weijia Shi and Jungo Kasai and Yizhong Wang and Yushi Hu and Mari Ostendorf and Wen-tau Yih and Noah A. Smith and Luke Zettlemoyer and Tao Yu},
      year={2023},
      eprint={2212.09741},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2212.09741}, 
}
@inproceedings{
lee2025nvembed,
title={{NV}-Embed: Improved Techniques for Training {LLM}s as Generalist Embedding Models},
author={Chankyu Lee and Rajarshi Roy and Mengyao Xu and Jonathan Raiman and Mohammad Shoeybi and Bryan Catanzaro and Wei Ping},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=lgsyLSsDRe}
}
```
