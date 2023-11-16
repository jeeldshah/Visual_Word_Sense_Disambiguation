# Visual_Word_Sense_Disambiguation

SemEval 2023 Task1(https://raganato.github.io/vwsd/): The task involves identifying the correct sense of ambiguous words in a text by selecting a clarifying image from a set of potentially misleading options. 

This repository follows the procedure from https://aclanthology.org/2023.semeval-1.199.pdf and uses the code provided in https://github.com/VaianiLorenzo/VWSD/blob/main/README.md to re-create the result and use the model as baseline for out work.

We have used the code base provided in the above mentioned repository to train the Baseline model. We cloned the repository and ran the following steps being in the root directory of the repository.
```
git clone git@github.com:VaianiLorenzo/VWSD.git
cd VWSD
```
before running these, the train/trial/test data was downloaded from the task page(https://raganato.github.io/vwsd/) and saved in VWSD/semeval-2023-task-1-V-WSD-train-v1 directory in the structure as mentioned in the repository.


For their Baseline, they have got inference directly from the pretrained CLIP models (base and large). The results for this were recreated by running the following command:
```
python single_clip_inference.py \
  --log_filename clip_results.txt \
  --log_step 200 \
  --phase val \
  --model_size large
```
or 
```
python single_clip_inference.py \
  --log_filename clip_results.txt \
  --log_step 200 \
  --phase val \
  --model_size base
```
The results published by them had :
```
FullSentence
	HIT RATE: 0.6047
	MRR: 0.7388
```
After recreating this, we got: 

CLIP_base:
```
VERSION -> FullSentence
	HIT RATE: 0.5788336933045356
	MRR: 0.7242269189893379
```
CLIP_large:
```
VERSION -> FullSentence
	HIT RATE: 0.6004319654427646
	MRR: 0.7363819465871299
```
The inference logs for this can be found in 
- inference/base_clip_results.txt
- inference/large_clip_results.txt



They have first augmented their textual data and then used all the data to finetune clip (openai/clip-vit-large-patch14). They did not provide the augmented text files in their repository, so we reproduced them using their script. The process of doing this is available in this repository in the notebook VWSD_Back_Translation.ipynb. This notebook can be run sequentially to get the augmented files.

They then use this to fine-tune clip. 

They got :

```
FullSentence
	HIT RATE: 0.6523
	MRR: 0.7830
```

We recreated this using the command:
```
python clip_finetuning.py \      
  --textual_input full_phrase \                           
  --log_filename clip_training_next.txt \
  --epochs 30 \
  --batch_size 16 \                                                       
  --model_size large \
  --textual_augmentation \
  --visual_augmentation \

```
The training logs for this can be found in logs/clip_training.txt

We then got the inference from this model that was finetuned by us using their script using the command:
```
python single_clip_inference.py \
  --log_filename clip_results.txt \
  --log_step 200 \
  --phase val \
  --clip_finetuned_model_name "/Users/jeelshah/Documents/Fall 2023/ANLP/project/VWSD/checkpoints/clip_large_mono_full_phrase_TrueTrue.pt"
```

The inference logs can be found in logs/clip_results_large_mono_TrueTrue.txt

The results we got were:
```
VERSION -> FullSentence
	HIT RATE: 0.6501079913606912
	MRR: 0.7787342726867561
```
