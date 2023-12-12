# The original code was taken from, https://github.com/VaianiLorenzo/VWSD/blob/main/README.md repository, 
# This code was modified to improve the performance of the model proposed in the above repository

from PIL import Image
Image.MAX_IMAGE_PIXELS = 631770000

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm
import os
import pandas as pd
import argparse
import random

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms

from transformers import CLIPProcessor, CLIPModel
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Define the dataset
class CLIP_dataset(data.Dataset):
    def __init__(self, list_image_path, list_txt, processor, list_txt_aug=None, ta=False, va=False):
        self.processor = processor
        self.image_path = list_image_path
        self.txt = list_txt
        if list_txt_aug is not None and ta:
            self.list_txt_aug = list_txt_aug
        self.ta = ta
        self.va = va
        
    def __len__(self):
        return len(self.txt)

    def __getitem__(self, idx):

        if self.ta:
            random_value = random.random()
            if random_value < 0.1:
                text = self.list_txt_aug[0][idx]
            elif random_value < 0.2:
                text = self.list_txt_aug[1][idx]
            elif random_value < 0.3:
                text = self.list_txt_aug[2][idx]
            elif random_value < 0.4:
                text = self.list_txt_aug[3][idx]
            else:
                text = self.txt[idx]       
        else:
            text = self.txt[idx]
        processed_text = self.processor(text=text, return_tensors="pt", max_length=16, padding="max_length", truncation=True)

        if self.va:
            image = self.augment_image(Image.open(self.image_path[idx]))
        else:
            image = Image.open(self.image_path[idx])
        image = self.processor(images=[image], return_tensors="pt")

        #return image, {"input_ids":self.processed_sentences["input_ids"][idx], "attention_mask":self.processed_sentences["attention_mask"][idx]}
        return image, {"input_ids":processed_text["input_ids"][0], "attention_mask":processed_text["attention_mask"][0]}
    
    def augment_image(self, image):
        if random.random() < 0.2:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.2:
            image = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)(image)
        if random.random() < 0.2:
            image = image.convert("L")
        return image

def probabilistic_contrastive_loss(u, v_pos, v_neg, tau=1.0):
    """
    Compute the probabilistic contrastive loss.

    Parameters:
    u (Tensor): Embeddings for the anchor.
    v_pos (Tensor): Embeddings for the positive examples.
    v_neg (Tensor): Embeddings for the negative examples.
    tau (float): Temperature parameter.

    Returns:
    Tensor: Probabilistic contrastive loss.
    """
    # Compute the similarity for the positive pair
    pos_similarity = F.cosine_similarity(u, v_pos)

    # Compute similarities for negative pairs
    neg_similarity = F.cosine_similarity(u.unsqueeze(1), v_neg, dim=2)

    # Concatenate the positive and negative similarities
    all_similarities = torch.cat((pos_similarity.view(-1, 1), neg_similarity), dim=1)

    # Scale the similarities by the temperature parameter
    all_similarities /= tau

    # Apply the log and softmax to compute log-probabilities
    loss = -F.log_softmax(all_similarities, dim=1)[:, 0]

    # Return the mean loss
    return loss.mean()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CLIP finetuning")
    parser.add_argument(
        "--textual_input",
        help="Part of sentence to be used as input",
        default=None,
        choices=['full_phrase', 'target_word', 'main_topic'],
        required=True)
    parser.add_argument(
        "--log_filename",
        help="Name of the log file",
        default="log.txt",
        required=False)
    parser.add_argument(
        "--epochs",
        help="Number of epochs",
        type=int,
        default=30,
        required=False)
    parser.add_argument(
        "--batch_size",
        help="Batch size",
        type=int,
        default=16,
        required=False)
    parser.add_argument(
        "--model_size",
        help="Size of the CLIP model",
        default="large",
        choices=["large", "base"],
        required=False)  
    parser.add_argument(
        "--extra",
        help="file_name extension",
        default="",
        required=False)  
    parser.add_argument(
        "--try_new_loss",
        help="try new loss or not",
        default=None,
        required=False)  
    parser.add_argument(
        "--mtype",
        help="type of model",
        default="clip",
        required=False)   
    parser.add_argument(
        "--continue_training",
        help="flag to continue training from prev checkpoint",
        default=None,
        required=False) 
    parser.add_argument(
        "--grad_accum",
        help="flag to use gradient accumulation",
        default="no",
        required=False) 
    parser.add_argument(
        "--accumulation_steps",
        help="steps for gradient accumulation",
        default=None,
        required=False)   
    ta_parser = parser.add_mutually_exclusive_group(required=False)
    ta_parser.add_argument('--textual_augmentation', dest='textual_augmentation', action='store_true')
    ta_parser.add_argument('--no-textual_augmentation', dest='textual_augmentation', action='store_false')
    parser.set_defaults(textual_augmentation=False)
    va_parser = parser.add_mutually_exclusive_group(required=False)
    va_parser.add_argument('--visual_augmentation', dest='visual_augmentation', action='store_true')
    va_parser.add_argument('--no-visual_augmentation', dest='visual_augmentation', action='store_false')
    va_parser = parser.add_mutually_exclusive_group(required=False)
    va_parser.add_argument('--use_all_data', dest='use_all_data', action='store_true')
    va_parser.add_argument('--no-use_all_data', dest='use_all_data', action='store_false')
    parser.set_defaults(use_all_data=False)
    args = parser.parse_args()

    bs = args.batch_size
    epochs = args.epochs
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps"

    # Load the model
    if args.mtype == "dcxl":
        # if args.model_size == "large":
        print("training dcxl")
        model = CLIPModel.from_pretrained("Aixile/CLIP-ViT-L-14-DataComp.XL-s13B-b90K").to(device)
        processor = CLIPProcessor.from_pretrained("Aixile/CLIP-ViT-L-14-DataComp.XL-s13B-b90K") 
        # else:
        #     print("training dcxl small")
        #     model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K").to(device)
        #     processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K") 
    elif args.mtype == "metaclip":
        if args.model_size == "large":
            print("training metaclip")
            model = CLIPModel.from_pretrained("facebook/metaclip-h14-fullcc2.5b").to(device)
            processor = CLIPProcessor.from_pretrained("facebook/metaclip-h14-fullcc2.5b")
        else:
            print("training metaclip small")
            model = CLIPModel.from_pretrained("facebook/metaclip-b32-400m").to(device)
            processor = CLIPProcessor.from_pretrained("facebook/metaclip-b32-400m")
    else:
        print("training normal clip")
        if args.model_size == "large":
            model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        elif args.model_size == "base":
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    if args.continue_training:
        model = torch.load(os.path.join("checkpoints", args.continue_training),map_location=lambda storage, loc: storage)
        print("loaded prev checkpopint")
    model.to(device)

    # Load the dataset
    input_folder_path = os.path.join("semeval-2023-task-1-V-WSD-train-v1", "train_v1")
    train_data_path = os.path.join(input_folder_path, "train_data_v1.txt")
    train_label_path = os.path.join(input_folder_path, "train_label_v1.txt")
    val_data_path = os.path.join(input_folder_path, "val_data_v1.txt")
    val_label_path = os.path.join(input_folder_path, "val_label_v1.txt")
    images_path = os.path.join(input_folder_path, "train_images_v1")
    output_path_root = "checkpoints"

    # Load the dataset
    train_df = pd.read_csv(train_data_path, sep="\t", header=None, names=["target_word", "full_phrase", "image_0", "image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7", "image_8", "image_9"])
    with open(train_label_path, "r") as f:
        train_labels = f.readlines()
    train_gt_image_paths = [os.path.join(images_path, image_name[:-1]) for image_name in train_labels]
    if args.textual_augmentation:
        train_augmented_sentences = []
        languages = ["it", "de", "fr","fa"]
        for language in languages:
            with open(os.path.join(input_folder_path, "train_back_translation_aug_"+language+".txt"), "r") as f:
                train_augmented_sentences.append([sentence[:-1] for sentence in f.readlines()])
    else:
        train_augmented_sentences = None

    val_df = pd.read_csv(val_data_path, sep="\t", header=None, names=["target_word", "full_phrase", "image_0", "image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7", "image_8", "image_9"])
    with open(val_label_path, "r") as f:
        val_labels = f.readlines()
    val_gt_image_paths = [os.path.join(images_path, image_name[:-1]) for image_name in val_labels]
    if args.textual_augmentation:
        val_augmented_sentences = []
        languages = ["it", "de", "fr","fa"]
        for language in languages:
            with open(os.path.join(input_folder_path, "val_back_translation_aug_"+language+".txt"), "r") as f:
                val_augmented_sentences.append([sentence[:-1] for sentence in f.readlines()])
    else:
        val_augmented_sentences = None

    with open(os.path.join("logs", args.log_filename), "w") as f:
        f.write("TRAINING LOG\n")
    # Select the part of sentence to use to train the model        
    train_sentences = list(train_df["full_phrase"])
    train_ambiguities = list(train_df["target_word"])
    train_main_topics = [(sentence[:sentence.find(ambiguity)] + sentence[sentence.find(ambiguity)+len(ambiguity):]).strip() for sentence, ambiguity in zip(train_sentences, train_ambiguities)]

    val_sentences = list(val_df["full_phrase"])
    val_ambiguities = list(val_df["target_word"])
    val_main_topics = [(sentence[:sentence.find(ambiguity)] + sentence[sentence.find(ambiguity)+len(ambiguity):]).strip() for sentence, ambiguity in zip(val_sentences, val_ambiguities)]

    if args.textual_input == "full_phrase":
        train_textual_input = train_sentences
        val_textual_input = val_sentences
    elif args.textual_input == "target_word":
        train_textual_input = train_ambiguities
        val_textual_input = val_ambiguities
    elif args.textual_input == "main_topic":
        train_textual_input = train_main_topics
        val_textual_input = val_main_topics

    if args.use_all_data:
        train_gt_image_paths = train_gt_image_paths + val_gt_image_paths
        train_textual_input = train_textual_input + val_textual_input
        if args.textual_augmentation:
            train_augmented_sentences = [train_augmented_sentences[i] + val_augmented_sentences[i] for i in range(len(train_augmented_sentences))]
        train_dataset = CLIP_dataset(train_gt_image_paths, train_textual_input, processor, list_txt_aug=train_augmented_sentences, ta=args.textual_augmentation, va=args.visual_augmentation)
        train_dataloader = data.DataLoader(train_dataset, batch_size=bs, shuffle = True, num_workers=24)
    else:
        train_dataset = CLIP_dataset(train_gt_image_paths, train_textual_input, processor, list_txt_aug=train_augmented_sentences, ta=args.textual_augmentation, va=args.visual_augmentation)
        train_dataloader = data.DataLoader(train_dataset, batch_size=bs, shuffle = True, num_workers=10)
        val_dataset = CLIP_dataset(val_gt_image_paths, val_textual_input, processor)
        val_dataloader = data.DataLoader(val_dataset, batch_size=bs, num_workers=10)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    #start_lr = 1e-7
    #end_lr = 1e-8
    # compute delta for linear scheduler
    #gamma = (end_lr / start_lr) ** (1 / epochs)
    #print("GAMMA:", gamma)

    optimizer = optim.Adam(model.parameters(), lr=1e-7,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer,  step_size=len(train_dataloader), gamma=gamma)

    best_val_loss = None
    best_train_loss = None
    total_loss = 0

    # finetuning clip
    for epoch in range(epochs):

        print("EPOCH-------->", epoch+1)
        # TRAIN 
        # model_saved = False
        model.train()      
        if args.grad_accum == "yes":
            optimizer.zero_grad()
        for i, batch in enumerate(tqdm(train_dataloader)):
            if not args.grad_accum == "yes":
                optimizer.zero_grad()
            images,texts = batch 
            images = images["pixel_values"].squeeze(1).to(device)
            for k in texts.keys():
                texts[k] = texts[k].to(device)
            outputs = model(input_ids=texts["input_ids"], attention_mask=texts["attention_mask"], pixel_values=images)
            ground_truth = torch.arange(len(images), device=device)
            # if i == 0:
            # print("<===================>")
            # print(outputs.logits_per_image.shape,outputs.logits_per_text.shape)
            if args.try_new_loss == "weighted_average":
                loss = 0.4*loss_img(outputs.logits_per_image,ground_truth) + 0.6*loss_txt(outputs.logits_per_text,ground_truth)
            elif args.try_new_loss == "cos_sim":
                        ce_loss = (loss_img(outputs.logits_per_image,ground_truth) + loss_txt(outputs.logits_per_text,ground_truth))/2
                        targets_one_hot = F.one_hot(ground_truth, outputs.logits_per_image.shape[1]).float()
                        # Normalize logits and targets to unit vectors
                        normalized_logits_image = F.normalize(outputs.logits_per_image, p=2, dim=1)
                        normalized_targets_image = F.normalize(targets_one_hot, p=2,dim=1)
                        normalized_logits_text = F.normalize(outputs.logits_per_text, p=2,dim=1)
                        normalized_targets_text = F.normalize(targets_one_hot,p=2, dim=1)

                        # Calculate cosine similarity (assuming targets are one-hot encoded)
                        cosine_similarity = (F.cosine_similarity(normalized_logits_image, normalized_targets_image, dim=1) + F.cosine_similarity(normalized_logits_text, normalized_targets_text, dim=1))/2
                        # Convert cosine similarity to a loss. We subtract from 1 because higher similarity should mean lower loss.
                        cosine_loss = 1 - cosine_similarity

                        # Mean cosine loss
                        cosine_loss = cosine_loss.mean()

                        # Combine the losses
                        alpha = 0.5
                        loss = alpha * ce_loss + (1 - alpha) * cosine_loss
            # elif args.try_new_loss == "nt-xent":
            #     temperature = 0.07
            #     loss = (loss_img(outputs.logits_per_image/temperature,ground_truth) + loss_txt(outputs.logits_per_text/temperature,ground_truth))/2
            # elif args.try_new_loss == "cpc":
            #     image_embeddings = outputs.logits_per_image#outputs.image_embeds
            #     text_embeddings = outputs.logits_per_text#outputs.text_embeds

            #     # Normalize embeddings
            #     image_embeddings = F.normalize(image_embeddings, dim=1)
            #     text_embeddings = F.normalize(text_embeddings, dim=1)

            #     # Compute similarity scores
            #     similarity = torch.matmul(image_embeddings, text_embeddings.T)

            #     loss = loss_img(similarity,ground_truth) 
            else:
                loss = (loss_img(outputs.logits_per_image,ground_truth) + loss_txt(outputs.logits_per_text,ground_truth))/2
            loss.backward()
            if args.grad_accum == "yes":
                # print("updating grads")
                if (i + 1) % int(args.accumulation_steps) == 0 or i + 1 == len(train_dataloader):
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                optimizer.step()
            #scheduler.step()
            total_loss += loss.item()
            if best_train_loss == None or total_loss < best_train_loss:
                print("BEST train model found")
                torch.save(model, os.path.join(output_path_root, str(epoch)+"ep-train_clip_"+args.extra+"_" + args.model_size + "_mono_" + args.textual_input + "_" + str(args.textual_augmentation) + str(args.visual_augmentation) + ".pt"))
                best_train_loss = total_loss
                best_epoch = epoch

        print("Train Loss at epoch", epoch+1, "->", total_loss/len(train_dataloader))
        with open(os.path.join("logs", args.log_filename), "a") as f:
            f.write("Epoch " + str(epoch+1) + "\n")
            f.write("\tTraining loss: " + str(total_loss/len(train_dataloader)) + "\n")
        total_loss = 0

        if not args.use_all_data:
            # VALIDATION 
            model.eval()    
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_dataloader)):
                    images,texts = batch 
                    images = images["pixel_values"].squeeze(1).to(device)
                    for k in texts.keys():
                        texts[k] = texts[k].to(device)
                    outputs = model(input_ids=texts["input_ids"], attention_mask=texts["attention_mask"], pixel_values=images)
                    ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

                    if args.try_new_loss == "weighted_average":
                        # print("trying weighted_average")
                        # loss = 0.3*loss_img(outputs.logits_per_image,ground_truth) + 0.7*loss_txt(outputs.logits_per_text,ground_truth)
                        loss = 0.4*loss_img(outputs.logits_per_image,ground_truth) + 0.6*loss_txt(outputs.logits_per_text,ground_truth)
                    elif args.try_new_loss == "cos_sim":
                        ce_loss = (loss_img(outputs.logits_per_image,ground_truth) + loss_txt(outputs.logits_per_text,ground_truth))/2
                        targets_one_hot = F.one_hot(ground_truth, outputs.logits_per_image.shape[1]).float()
                        # Normalize logits and targets to unit vectors
                        normalized_logits_image = F.normalize(outputs.logits_per_image, dim=1)
                        normalized_targets_image = F.normalize(targets_one_hot, dim=1)
                        normalized_logits_text = F.normalize(outputs.logits_per_text, dim=1)
                        normalized_targets_text = F.normalize(targets_one_hot, dim=1)

                        # Calculate cosine similarity (assuming targets are one-hot encoded)
                        cosine_similarity = (F.cosine_similarity(normalized_logits_image, normalized_targets_image, dim=1) + F.cosine_similarity(normalized_logits_text, normalized_targets_text, dim=1))/2
                        # Convert cosine similarity to a loss. We subtract from 1 because higher similarity should mean lower loss.
                        cosine_loss = 1 - cosine_similarity

                        # Mean cosine loss
                        cosine_loss = cosine_loss.mean()

                        # Combine the losses
                        alpha = 0.5
                        loss = alpha * ce_loss + (1 - alpha) * cosine_loss
                    # elif args.try_new_loss == "nt-xent":
                    #     # temperature = 0.07
                    #     loss = (loss_img(outputs.logits_per_image/0.07,ground_truth) + loss_txt(outputs.logits_per_text/0.07,ground_truth))/2
                    # elif args.try_new_loss == "cpc":
                    #     image_embeddings = outputs.logits_per_image#outputs.image_embeds
                    #     text_embeddings = outputs.logits_per_text#outputs.text_embeds

                    #     # Normalize embeddings
                    #     image_embeddings = F.normalize(image_embeddings, dim=1)
                    #     text_embeddings = F.normalize(text_embeddings, dim=1)

                    #     # Compute similarity scores
                    #     similarity = torch.matmul(image_embeddings, text_embeddings.T)

                    #     loss = loss_img(similarity,ground_truth) 
                    else:
                        loss = (loss_img(outputs.logits_per_image,ground_truth) + loss_txt(outputs.logits_per_text,ground_truth))/2
                    total_loss += loss.item()
            
            if best_val_loss == None or total_loss < best_val_loss:
                print("BEST model found")
                torch.save(model, os.path.join(output_path_root, "clip_"+args.extra+"_" + args.model_size + "_mono_" + args.textual_input + "_" + str(args.textual_augmentation) + str(args.visual_augmentation) + ".pt"))
                best_val_loss = total_loss
                best_epoch = epoch
                

            print("Validation", epoch+1, "->", total_loss/len(val_dataloader))
            with open(os.path.join("logs", args.log_filename), "a") as f:
                f.write("\tValidation loss: " + str(total_loss/len(val_dataloader)) + "\n")
            total_loss = 0

    if not args.use_all_data:
        with open(os.path.join("logs", args.log_filename), "a") as f:
            f.write("Best model found at epoch " + str(best_epoch+1) + " with validation loss: " + str(best_val_loss/len(val_dataloader)) + "\n")
    else:
        torch.save(model, os.path.join(output_path_root, "all_clip_" + args.model_size + "_mono_" + args.textual_input + "_" + str(args.textual_augmentation) + str(args.visual_augmentation) + ".pt"))

# if __name__ == '__main__':
    # torch.multiprocessing.freeze_support()