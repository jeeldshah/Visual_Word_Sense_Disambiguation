# Visual_Word_Sense_Disambiguation

## Training and Inference of Baseline Model
SemEval 2023 Task1(https://raganato.github.io/vwsd/): The task involves identifying the correct sense of ambiguous words in a text by selecting a clarifying image from a set of potentially misleading options. 

This repository follows the procedure from https://aclanthology.org/2023.semeval-1.199.pdf and uses the code provided in https://github.com/VaianiLorenzo/VWSD/blob/main/README.md to re-create the result and use the model as baseline for our work. We then carried out various experiments to improve the results. we were able to increase our performance from Hit rate 0.65011 and MRR 0.77873 (Baseline) to Hit rate 0.7235 and MRR 0.8280 (facebook/metaclip-h14-fullcc2.5b with weighted average loss)

For Baseline related information, refer [README-Baseline.md](README-Baseline.md)

For Improvements related information, refer [README-Improvements.md](README-Improvements.md)
