# Visual_Word_Sense_Disambiguation

## Improvisations


This is an improvisation to  https://aclanthology.org/2023.semeval-1.199.pdf and uses the code provided in https://github.com/VaianiLorenzo/VWSD/blob/main/README.md as a base.

### Inference

A new argument, "mtype" is introduced in both "clip_finetuning.py" and "single_clip_inference.py" that allows for testing the different models. 
- mtype: "metaclip" model_size: "large" for facebook/metaclip-h14-fullcc2.5b
- mtype: "metaclip" model_size: "base" for facebook/metaclip-b32-400m
- mtype: "dcxl" model_size: "large"/"base" for Aixile/CLIP-ViT-L-14-DataComp.XL-s13B-b90K

and otherwise it defaults to the baseline with 
- model_size: "large" for openai/clip-vit-large-patch14
- model_size: "base" for openai/clip-vit-base-patch32

It can be used as an addition to the command used for Baseline like this:

Inference:
```
python single_clip_inference.py \
  --log_filename clip_results.txt \
  --log_step 200 \
  --phase val \
  --model_size large
  --mytype metaclip
```
or 
```
python single_clip_inference.py \
  --log_filename clip_results.txt \
  --log_step 200 \
  --phase val \
  --model_size base
```
Fine-tuning: 

```
python clip_finetuning.py \      
  --textual_input full_phrase \                           
  --log_filename clip_training_next.txt \
  --epochs 30 \
  --batch_size 16 \                                                       
  --model_size large \
  --textual_augmentation \
  --visual_augmentation \
  --mtype

```

### Fine-tuning
We also tried different losses , for which we introduced another command line argument "try_new_loss" that uses value "weighted_average" to use the Weighted Average Loss , "cos_sim" to use Combination of Cosine Similarity and Cross Entropy Loss and defaults to Cross Entropy loss during fine-tuning.

To account for large models and keep experiments fair, we implemented gradient accumulation so that we can run all models with Batch Size of 16. In order to use gradient accumulation, the code allows this by using argument "grad_accum" = "yes" (which is otherwise "no" by default). The "accumulation_steps" argument can also be set to adjust the factor. 

It can be used in addition to other arguments in the standard command like this:

```
python clip_finetuning.py \      
  --textual_input full_phrase \                           
  --log_filename clip_training_next.txt \
  --epochs 30 \
  --batch_size 2 \                                                       
  --model_size large \
  --textual_augmentation \
  --visual_augmentation \
  --mtype \ 
  --try_new_loss cos_sim \
  --grad_accum yes \
  --accumulation_steps 8

```
## Error Analysis

The code for identifying the incorrect predictions and getting a comparative view of the prediction and ground truth has been updated in error_analysis.ipynb. The notebook now also has the code to create a collage for comparing the predictions across models and losses. 

These images show the original
image on the top left corner, with predictions from
different models around it - labeled with the model
in the top left of the image(in yellow). The Full
Phrase text (in black with a red background) is in
the bottom right. 

The cells in the notebook can be followed sequentially to get image outputs of the errors.


