# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from functools import partial
from torch.cuda.amp import autocast
import torch.distributed as dist
from tqdm import tqdm
from torchvision.utils import save_image
import sys
import pdb
import logging
import torch.nn.functional as F
from third_party.open_clip.clip import tokenize, _transform
import pickle
import pdb
from utils import is_master
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Union, Dict, Literal,Tuple
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from data import CsvDataset, CustomFolder, ImageList, CsvCOCO, FashionIQ, CIRR, CIRCODataset, collate_fn
import pandas as pd
from sklearn.cluster import KMeans

def get_templates():
    """
    Return a list of templates
    Same templates as in PALAVRA: https://arxiv.org/abs/2204.01694
    """
    return [
        "This is a photo of a {}",
        "This photo contains a {}",
        "A photo of a {}",
        "This is an illustration of a {}",
        "This illustration contains a {}",
        "An illustrations of a {}",
        "This is a sketch of a {}",
        "This sketch contains a {}",
        "A sketch of a {}",
        "This is a diagram of a {}",
        "This diagram contains a {}",
        "A diagram of a {}",
        "A {}",
        "We see a {}",
        "{}",
        "We see a {} in this photo",
        "We see a {} in this image",
        "We see a {} in this illustration",
        "We see a {} photo",
        "We see a {} image",
        "We see a {} illustration",
        "{} photo",
        "{} image",
        "{} illustration",
    ]

def get_dict_embedding(m,args):
    df = pd.read_csv("./data/oidv7-class-descriptions.csv")
    concept_texts = list(set(df["DisplayName"].values.tolist()))

    # Compute the embeddings for the dictionary
    bs = 128
    dictionary_embeddings = torch.zeros((0, 768)).cuda(args.gpu, non_blocking=True)
    with torch.no_grad():
        for i in tqdm(range(0, len(concept_texts), bs)):
            if i + bs > len(concept_texts) - 1:
                bs = len(concept_texts) - i
            
            """
            for k in range(i, i + bs):
                prompts = [f"{template.format(f' {concept_texts[k]} ')}" for template in templates]
                tokens = tokenize(prompts).cuda(args.gpu, non_blocking=True)
                feat = m.encode_text(tokens)
                feat /= feat.norm(dim=-1, keepdim=True)
                feat = feat.mean(dim=0, keepdim=True)
                feat /= feat.norm()
                #pdb.set_trace()
                dictionary_embeddings = torch.vstack((dictionary_embeddings, feat))
            """
            prompts = [concept_texts[k] for k in range(i, i + bs)]
            tokens = tokenize(prompts).cuda(args.gpu, non_blocking=True)
            feat = m.encode_text(tokens)
            feat /= feat.norm(dim=-1, keepdim=True)
            dictionary_embeddings = torch.vstack((dictionary_embeddings, feat))

    """
    k = 100
    text_embeddings = dictionary_embeddings.detach().cpu().numpy()
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster_labels = kmeans.fit_predict(text_embeddings)

    # Get cluster centroids
    cluster_centers = kmeans.cluster_centers_

    # Create a new tensor based on cluster centroids
    clustered_embeddings = cluster_centers[cluster_labels]
    dictionary_embeddings = torch.from_numpy(clustered_embeddings)
    dictionary_embeddings = dictionary_embeddings.cuda(args.gpu, non_blocking=True)
    """
    #dictionary_embeddings = dictionary_embeddings[:5000,:]

    # Normalize the embeddings
    #dictionary_embeddings = F.normalize(dictionary_embeddings, dim=-1)
    return dictionary_embeddings, concept_texts

def prepare_img(img_file, transform):
    return transform(Image.open(img_file))

def visualize_results(model, img2text, args, prompt, dataloader):        
    model.eval()
    img2text.eval()   
    if not os.path.exists(args.demo_out):
        os.makedirs(args.demo_out)        
    if not os.path.exists(os.path.join(args.demo_out, "images")):
        os.makedirs(os.path.join(args.demo_out, "images"))
    text = []
    id_split = tokenize(["*"])[0][1]
    #pdb.set_trace() # make sure that the text prompt contains "*"
    for p in prompt:
        text_tokens = tokenize(p)
        text.append(text_tokens)
        assert id_split in text_tokens
    text = torch.cat(text, dim=0)    
    text = text.cuda(args.gpu, non_blocking=True)    
    all_image_features, all_image_filenames = [], []
    m = model.module if args.distributed or args.dp else model
    query_file = args.query_file
    path_save = os.path.join("./data", args.retrieval_data.split('/')[-1].split('.')[0]+".pkl")
    if os.path.exists(path_save):
        with open(path_save, 'rb') as f:
            data = pickle.load(f)
        all_image_features = data['feats']
        all_image_filenames = data['path']
        all_image_features = torch.from_numpy(all_image_features).cuda(args.gpu, non_blocking=True)
    else:
        ## Extract features of target images. 
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images, filenames = batch
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                image_features = m.encode_image(images)           
                image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
                all_image_features.append(image_features)
                for name in filenames:
                    all_image_filenames.append(name)
            all_image_features = torch.cat(all_image_features, dim=0)
            dict_save = {}
            dict_save['feats'] = all_image_features.data.cpu().numpy()
            dict_save['path'] = all_image_filenames
            with open(path_save,"wb") as f:
                pickle.dump(dict_save,f)
    f = open(os.path.join(args.demo_out, "index.html"), 'w')
    html_txt = """"""
    ## For each domain, compute composed features and evaluate.
    for query in query_file.split(","):        
        logging.info("retrieve image of {}".format(query))
        transform = _transform(model.visual.input_resolution)
        query_img = prepare_img(query, transform)
        query_img = torch.unsqueeze(query_img, 0)    
        query_img = query_img.cuda(args.gpu, non_blocking=True)
        img_feature = m.encode_image(query_img) 
        query_img_feature = img2text(img_feature)
        composed_feature = m.encode_text_img_vis(text, query_img_feature, split_ind=id_split)
        composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)
        img_feature = img_feature / img_feature.norm(dim=-1, keepdim=True)
        text_feature = m.encode_text(text)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        similarity = composed_feature @ all_image_features.T
        _, indices = torch.sort(similarity, descending=True)        
        logging.info("Composed feature result")
        for i, caption in enumerate(prompt):
            logging.info("for prompt {}".format(caption))
            for j, ind in enumerate(indices[i][:10]):
                logging.info("top {} filename {}".format(j, all_image_filenames[ind]))
        image_paths = [[all_image_filenames[ind] for j, ind in enumerate(indices[i][:10])] 
                        for i, caption in enumerate(prompt)]
        html_txt += make_html(prompt, query, image_paths, args.demo_out)
    f.write(html_txt)

def make_html(prompts, query_image, images, path_html):
    import shutil
    html_all = """"""        
    for i in range(len(prompts)):
        prompt = prompts[i]            
        query_image_local = os.path.join(path_html, "images", query_image.split("/")[-1])
        query_image_local_path = os.path.join("images", query_image.split("/")[-1])
        shutil.copy(query_image, query_image_local)
        image_list = images[i]        
        html = """<table><tr>"""    
        html += """<td><p style="display:inline-block;vertical-align;font-size:20px">%s</p></td>"""%(prompt)
        html += """<td><p style="margin-right: 50px;"><img src="%s" height="100"></p></td>"""%(query_image_local_path)
        for image in image_list:
            image_local = os.path.join(path_html, "images", image.split("/")[-1])
            image_path = os.path.join("images", image.split("/")[-1])
            shutil.copy(image, image_local)
            html += """<td><img src="%s" height=%s></td>"""%(image_path, 200)
        html += """</tr></table>"""
        html_all += html
    return html_all
    #f.write(html_all)


def evaluate_imgnet_retrieval(model, img2text, args, prompt, query_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    all_image_features = []  
    all_target_labels = []      
    m = model.module if args.distributed or args.dp else model
    n_class = 1000
    with open("./imgnet_class_label_mapping.txt","r") as f_d:
        lines = f_d.readlines()
        label_text_dict = {line.split()[0]: line.split()[1] for line in lines if line.strip()}


    dictionary_embeddings, concept_texts = get_dict_embedding(m,args)
    with torch.no_grad():
        
        
        # just a test of images classification and tsne visualization
        label_text_origin = [value.replace("_", " ") for value in list(label_text_dict.values())]
        label_text = tokenize(label_text_origin)#.view(1, -1)            
        label_text = label_text.cuda(args.gpu, non_blocking=True)                        
        label_text_features = m.encode_text(label_text)
        label_text_features_normed = label_text_features / label_text_features.norm(dim=-1, keepdim=True)
        """
        all_query_features = []
        all_labels = []
        all_mapping_features = []
        for batch in tqdm(query_loader):
            images, labels = batch
            labels = labels % 1000
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                labels = labels.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(images)
            #image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
            
            image_mapping_features = img2text(image_features)

            all_query_features.append(image_features)
            all_mapping_features.append(image_mapping_features)
            gt_text = [label_text_dict[str(label.item())].replace("_", " ") for label in labels]
            all_labels += gt_text
            #pdb.set_trace()

        all_query_features = torch.cat(all_query_features)
        #all_query_features = torch.cat(all_mapping_features) @ m.text_projection

        image_features_np = all_query_features.cpu().numpy()

        # Initialize t-SNE model
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)

        # Apply t-SNE to image features
        embedded_features = tsne.fit_transform(image_features_np)

        # Scatter plot to visualize the embedded features
        plt.switch_backend('Agg')
        plt.figure(figsize=(10, 8))

        for class_label in set(all_labels):
            #class_indices = [i for i, label in enumerate(all_labels) if label == class_label]
            class_indices = np.where(np.array(all_labels) == class_label)[0]
            plt.scatter(embedded_features[class_indices, 0], embedded_features[class_indices, 1], label=f'Class {class_label}', marker='o')
        #plt.scatter(embedded_features[:, 0], embedded_features[:, 1], marker='o')
        plt.title('t-SNE Visualization of Image Features')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')

        # Save the plot to an image file
        plt.savefig('tsne_visualization.png')

        logits_per_image = (all_query_features @ label_text_features_normed.t()).detach().cpu()     
        predicted_indices = torch.argmax(logits_per_image, dim=1)
        predicted_labels = [label_text_origin[idx] for idx in predicted_indices]  
        
        correct_predictions = sum(1 for predicted, actual in zip(predicted_labels, all_labels) if predicted == actual)
        accuracy = correct_predictions / len(all_labels)       
        pdb.set_trace()
        """

        ## Extract target image features. 
        for batch in tqdm(target_loader):
            images, labels = batch
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                labels = labels.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            all_image_features.append(image_features)
            all_target_labels.append(labels)
            logit_scale = m.logit_scale.exp()
            logit_scale = logit_scale.mean() 


        ## Extract query features 
        for p_ind, p in enumerate(prompt):            
            ## which token has to be replaced with image features
            id_split = tokenize(["*"])[0][1]
            text = tokenize(p).view(1, -1)
            text = text.cuda(args.gpu, non_blocking=True)
            ## text only features (domain name only)
            text_only = p.replace("*", "")
            text_only = tokenize(text_only).view(1, -1)            
            text_only = text_only.cuda(args.gpu, non_blocking=True)                        
            text_only_features = m.encode_text(text_only)
            text_only_features_normed = text_only_features / text_only_features.norm(dim=-1, keepdim=True)

            all_query_features = []
            all_query_image_features = []
            all_query_mixture_features = []
            all_gt_text_features = []
            all_query_labels = []
            all_text_features = []
            for batch in tqdm(query_loader):
                images, labels = batch
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                    labels = labels.cuda(args.gpu, non_blocking=True)

                image_features = m.encode_image(images) 

                # test classification text
                #logits_per_image = (image_features @ label_text_features_normed.t()).detach().cpu()
                logits_per_image = (image_features @ dictionary_embeddings.t()).detach().cpu()  
                predicted_indices = torch.argmax(logits_per_image, dim=1)

               
                predicted_indices = torch.argmax(logits_per_image, dim=1)
                #_, topk_indices = torch.topk(logits_per_image, 5,dim=1)
                gt_text = [p.replace("*",concept_texts[index.item()]) for index in predicted_indices] # this is prediction text
                #gt_text = [cap.replace("*"," ".join([concept_texts[index.item()] for index in index_list])) for index_list, cap in zip(topk_indices,raw_captions)]

                #pdb.set_trace()
                #gt_text = [p.replace("*",label_text_origin[index.item()]) for index in predicted_indices] # this is prediction text
                #gt_text = [p.replace("*",label_text_dict[str(label.item())].replace("_", " ")) for label in labels] # this is gt text
                #pdb.set_trace()
                gt_text = tokenize(gt_text)#.view(1, -1)            
                gt_text = gt_text.cuda(args.gpu, non_blocking=True)                        
                gt_text_features = m.encode_text(gt_text)
                gt_text_features_normed = gt_text_features / gt_text_features.norm(dim=-1, keepdim=True)

                ## Label is decided by class label and images' domain
                labels += n_class * p_ind

                ## Composed feature extraction
                image_features_query = img2text(image_features)                      
                composed_feature = m.encode_text_img_retrieval(text, image_features_query, split_ind=id_split)                         
                composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)            
                ## Image feature only
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)  
                ## average of image and text features
                mixture_features = image_features + text_only_features_normed
                mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)       

                all_text_features.append(text_only_features_normed.repeat((image_features.shape[0], 1)))
                all_query_features.append(composed_feature)
                all_query_image_features.append(image_features)
                all_query_mixture_features.append(mixture_features)

                all_gt_text_features.append(gt_text_features_normed)
                
                all_query_labels.append(labels)
                #pdb.set_trace()

            metric_func = partial(get_metrics_imgnet, 
                image_features=torch.cat(all_image_features), 
                query_labels=torch.cat(all_query_labels),
                target_labels=torch.cat(all_target_labels),
                )

            feats = {'composed': torch.cat(all_query_features), 
                    'image': torch.cat(all_query_image_features),
                    'text': torch.cat(all_text_features),
                    'mixture': torch.cat(all_query_mixture_features),
                    'gt_text': torch.cat(all_gt_text_features),}        

            for key, value in feats.items():
                metrics = metric_func(query_features=value)
                logging.info("Current prompt:"+ str(p) +"Eval {key} Feature")
                logging.info(
                f"Eval {key} Feature"
                + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

    return metrics


def evaluate_coco(model, img2text, args, loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()

    all_image_features = []  
    all_query_image_features = []  
    all_mixture_features = []  
    all_composed_features_with_class = []  
    all_text_full_features = [] 

    m = model.module if args.distributed or args.dp else model
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean()
    with torch.no_grad():
        for batch in tqdm(loader):
            images, region_images, text_full, text_with_blank, text_with_blank_query, filename, raw_text = batch            
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                region_images = region_images.cuda(args.gpu, non_blocking=True)
                text_full = text_full.cuda(args.gpu, non_blocking=True)
                text_with_blank = text_with_blank.cuda(args.gpu, non_blocking=True)
                text_with_blank_query = text_with_blank_query.cuda(args.gpu, non_blocking=True)
            pdb.set_trace()
            ## Target image features 
            image_features = m.encode_image(images)             
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  
            id_split = tokenize(["*"])[0][1]
            ## Composed image features
            query_image_features = m.encode_image(region_images)
            query_image_tokens = img2text(query_image_features)          
            composed_feature_with_class = m.encode_text_img_retrieval(text_with_blank_query, query_image_tokens, split_ind=id_split, repeat=False)                        
            composed_feature_with_class = composed_feature_with_class / composed_feature_with_class.norm(dim=-1, keepdim=True)        
            ## Text only features
            text_full_features = m.encode_text(text_full)
            text_full_features = text_full_features / text_full_features.norm(dim=-1, keepdim=True)            
            ## Query only features
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)                               
            ## Mixed featurs
            mixture_features = query_image_features + text_full_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)            

            all_image_features.append(image_features.cpu())
            all_text_full_features.append(text_full_features.cpu())       
            all_query_image_features.append(query_image_features.cpu())
            all_mixture_features.append(mixture_features.cpu())                        
            all_composed_features_with_class.append(composed_feature_with_class.cpu())            

        metric_func = partial(get_metrics_coco, 
                image_features=torch.cat(all_image_features), 
                logit_scale=logit_scale
                )
        feats = {'composed': torch.cat(all_composed_features_with_class), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_text_full_features),
                 'mixture': torch.cat(all_mixture_features)}        

        for key, value in feats.items():
            metrics = metric_func(ref_features=value)
            logging.info(
            f"Eval {key} Feature"
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

    return metrics


def evaluate_cirr(model, img2text, args, query_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()

    templates = get_templates()
    all_image_features = []  
    all_query_image_features = []  
    all_composed_features = []  
    all_mixture_features = []  
    all_caption_features = []  
    all_ref_paths = []
    all_target_paths = []
    all_answer_paths = []
    all_raw_captions = []
    all_gt_text_features = []

    m = model.module if args.distributed or args.dp else model
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean()   

    dictionary_embeddings, concept_texts = get_dict_embedding(m,args)

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)

        for batch in tqdm(query_loader):
            ref_images, text_with_blank, caption_only, ref_paths, answer_paths, raw_captions = batch
            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                text_with_blank = text_with_blank.cuda(args.gpu, non_blocking=True)
                caption_only = caption_only.cuda(args.gpu, non_blocking=True)
            id_split = tokenize(["*"])[0][1]                        
            for path in ref_paths:
                all_ref_paths.append(path)
            for path in answer_paths:
                all_answer_paths.append(path)
            for cap in raw_captions:
                all_raw_captions.append(cap)
            
            #pdb.set_trace()

            caption_features = m.encode_text(caption_only)
            ## Composed features
            query_image_features = m.encode_image(ref_images)

            logits_per_image = (query_image_features @ dictionary_embeddings.t()).detach().cpu()     
            predicted_indices = torch.argmax(logits_per_image, dim=1)
            _, topk_indices = torch.topk(logits_per_image, 5,dim=1)
            #pdb.set_trace()
            #gt_text = [cap.replace("*",concept_texts[index.item()]) for index,cap in zip(predicted_indices,raw_captions)] # this is prediction text
            #pdb.set_trace()
            gt_text = [cap.replace("*"," ".join([concept_texts[index.item()] for index in index_list])) for index_list, cap in zip(topk_indices,raw_captions)] # this is prediction topk text
            #gt_text = [p.replace("*",label_text_dict[str(label.item())].replace("_", " ")) for label in labels] # this is gt text
            #pdb.set_trace()
            gt_text = tokenize(gt_text)#.view(1, -1)            
            gt_text = gt_text.cuda(args.gpu, non_blocking=True)                        
            gt_text_features = m.encode_text(gt_text)
            gt_text_features_normed = gt_text_features / gt_text_features.norm(dim=-1, keepdim=True)



            query_image_tokens = img2text(query_image_features)
            composed_feature = m.encode_text_img_retrieval(text_with_blank, query_image_tokens, split_ind=id_split, repeat=False)                

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)                       
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)   
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)            
            mixture_features = query_image_features + caption_features            
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_composed_features.append(composed_feature)            
            all_mixture_features.append(mixture_features) 
            all_gt_text_features.append(gt_text_features_normed)
            #pdb.set_trace()    

        #pdb.set_trace()                   

        all_target_paths = np.array(all_target_paths)
        all_ref_paths = np.array(all_ref_paths)
        all_answer_paths = np.array(all_answer_paths)
        
        
        metric_func = partial(get_metrics_cirr, 
                image_features=torch.cat(all_image_features), 
                reference_names=all_ref_paths, 
                index_names=all_target_paths, 
                target_names=all_answer_paths)

        feats = {'composed': torch.cat(all_composed_features), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_caption_features),
                 'mixture': torch.cat(all_mixture_features),
                 'gt_text': torch.cat(all_gt_text_features)}
        
        for key, value in feats.items():
            metrics = metric_func(ref_features=value)
            logging.info(
            f"Eval {key} Feature"
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
    return metrics


def evaluate_cirr_test(model, img2text, args, query_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()

    all_image_features = []  
    all_query_image_features = []  
    all_composed_features = []  
    all_composed_plus_image_features = []  
    all_mixture_features = []  
    all_caption_features = []  
    all_ref_paths = []
    all_target_paths = []
    all_answer_paths = []
    all_ids = []

    m = model.module if args.distributed or args.dp else model   
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean()   

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)

        for batch in tqdm(query_loader):
            ref_images, text_with_blank, caption_only, ref_paths, pairids, _ = batch
            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                text_with_blank = text_with_blank.cuda(args.gpu, non_blocking=True)
                caption_only = caption_only.cuda(args.gpu, non_blocking=True)
            id_split = tokenize(["*"])[0][1]                        
            for ids in pairids:
                all_ids.append(ids)
            for path in ref_paths:
                all_ref_paths.append(path)

            caption_features = m.encode_text(caption_only)
            query_image_features = m.encode_image(ref_images)

            #if args.eval_combiner:
            #    composed_feature = img2text(query_image_features, caption_features)
            #else:
            query_image_tokens = img2text(query_image_features)
            composed_feature = m.encode_text_img_retrieval(text_with_blank, query_image_tokens, split_ind=id_split, repeat=False)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)                       
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)   
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)            
            mixture_features = query_image_features + caption_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_composed_features.append(composed_feature)            
            all_mixture_features.append(mixture_features)            

        all_target_paths = np.array(all_target_paths)
        all_ref_paths = np.array(all_ref_paths)
        all_answer_paths = np.array(all_answer_paths)
        res_all = {}
        metrics_func = partial(get_cirr_testoutput, 
                               image_features=torch.cat(all_image_features),
                               reference_names=all_ref_paths,
                               index_names=all_target_paths,
                               id_names=all_ids)
        feats = {'composed': torch.cat(all_composed_features), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_caption_features),
                 'mixture': torch.cat(all_mixture_features)}
        #pdb.set_trace()        
        for key, value in feats.items():
            res_all[key] = metrics_func(ref_features=value)
    return res_all


def evaluate_fashion(model, img2text, args, source_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    all_target_paths = []
    all_answer_paths = []
    all_image_features = []  
    all_query_image_features = []  
    all_composed_features = []  
    all_caption_features = []  
    all_mixture_features = [] 
    all_gt_text_features = [] 
    all_reference_names = []
    all_captions = []     
    m = model.module if args.distributed or args.dp else model
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean() 

    dictionary_embeddings, concept_texts = get_dict_embedding(m,args)

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)

    with torch.no_grad():
        for batch in tqdm(source_loader):
            ref_images, target_images, target_caption, caption_only, answer_paths, ref_names, captions = batch
            for path in answer_paths:
                all_answer_paths.append(path)
            all_reference_names.extend(ref_names)
            all_captions.extend(captions)
            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                target_images = target_images.cuda(args.gpu, non_blocking=True)
                target_caption = target_caption.cuda(args.gpu, non_blocking=True)
                caption_only = caption_only.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            query_image_features = m.encode_image(ref_images)
            id_split = tokenize(["*"])[0][1]    
            caption_features = m.encode_text(target_caption)    


            logits_per_image = (query_image_features @ dictionary_embeddings.t()).detach().cpu()     
            predicted_indices = torch.argmax(logits_per_image, dim=1)
            #pdb.set_trace()
            gt_text = [cap.replace("*",concept_texts[index.item()]) for index,cap in zip(predicted_indices,captions)] # this is prediction text
            #gt_text = [p.replace("*",label_text_dict[str(label.item())].replace("_", " ")) for label in labels] # this is gt text
            #pdb.set_trace()
            gt_text = tokenize(gt_text)#.view(1, -1)            
            gt_text = gt_text.cuda(args.gpu, non_blocking=True)                        
            gt_text_features = m.encode_text(gt_text)
            gt_text_features_normed = gt_text_features / gt_text_features.norm(dim=-1, keepdim=True)


            query_image_tokens = img2text(query_image_features)          
            composed_feature = m.encode_text_img_retrieval(target_caption, query_image_tokens, split_ind=id_split, repeat=False)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)                       
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)   
            mixture_features = query_image_features + caption_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)

            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_composed_features.append(composed_feature)            
            all_mixture_features.append(mixture_features)     
            all_gt_text_features.append(gt_text_features_normed)                    

        metric_func = partial(get_metrics_fashion, 
                              image_features=torch.cat(all_image_features),
                              target_names=all_target_paths, answer_names=all_answer_paths)
        feats = {'composed': torch.cat(all_composed_features), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_caption_features),
                 'mixture': torch.cat(all_mixture_features),
                 'gt_text': torch.cat(all_gt_text_features)}
        #pdb.set_trace()
        
        for key, value in feats.items():
            metrics = metric_func(ref_features=value)
            logging.info(
            f"Eval {key} Feature"
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
    return metrics


def get_metrics_coco(image_features, ref_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale.cpu() * image_features @ ref_features.t()).detach().cpu()
    logits_per_ref = logits_per_image.t().detach().cpu()
    logits = {"image_to_ref": logits_per_image, "ref_to_image": logits_per_ref}
    ground_truth = torch.arange(len(ref_features)).view(-1, 1)
    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10, 50, 100]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)
    return metrics


def get_metrics_fashion(image_features, ref_features, target_names, answer_names):
    metrics = {}
    distances = 1 - ref_features @ image_features.T    
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(target_names)[sorted_indices]
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(answer_names), len(target_names)).reshape(len(answer_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(answer_names)).int())
    # Compute the metrics
    for k in [1, 5, 10, 50, 100]:
        metrics[f"R@{k}"] = (torch.sum(labels[:, :k]) / len(labels)).item() * 100
    return metrics


def get_metrics_cirr(image_features, ref_features, reference_names, index_names, target_names):
    metrics = {}
    distances = 1 - ref_features @ image_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    for i in range(sorted_index_names.shape[0]):
        for j in range(sorted_index_names.shape[1]):
            sorted_index_names[i][j] = os.path.basename(sorted_index_names[i][j])

    # Delete the reference image from the results
    #pdb.set_trace()
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), 
        len(index_names)).reshape(len(target_names), -1))        
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), 
        len(index_names) - 1).reshape(len(target_names), -1))

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    for k in [1, 5, 10, 50, 100]:
        metrics[f"recall_R@{k}"] = (torch.sum(labels[:, :k]) / len(labels)).item() * 100

    return metrics


def get_cirr_testoutput(image_features, ref_features, reference_names, index_names, id_names):
    metrics = {}
    distances = 1 - ref_features @ image_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    result_dict = {"version": "rc2", "metric": "recall"}
    for ind in range(len(id_names)):
        pairid = str(id_names[ind].item())
        result_dict[pairid] = []
        for t in range(50):
            result_dict[pairid].append(sorted_index_names[ind][t].replace(".png", ""))
    return result_dict


def get_metrics_imgnet(query_features, image_features, query_labels, target_labels):
    metrics = {}
    num_classes = 7000
    query_onehot = F.one_hot(query_labels, num_classes=num_classes).float()
    target_onehot = F.one_hot(target_labels, num_classes=num_classes).float()
    batches = [(query_features[x:x+100], query_onehot[x:x+100]) for x in range(0, len(query_features), 100)]
    for k in [1, 5, 10, 50, 100, 200]:
        metrics[f"Real2Sketch_R@{k}"] = 0
        metrics[f"Real2Sketch_P@{k}"] = 0
        #metrics[f"upper_R@{k}"] = 0
    for batch in batches:
        feats, labels = batch[0], batch[1]
        logits_per_query = (feats @ image_features.t()).detach().cpu()
        label_matrix = (labels @ target_onehot.t()).detach().cpu()                
        ranking = torch.argsort(logits_per_query, descending=True)

        gt_ranking = torch.argsort(label_matrix, descending=True)
        for k in [1, 5, 10, 50, 100, 200]:
            matrix_k = torch.zeros_like(label_matrix)
            rank_k = ranking[:, :k]
            matrix_k[torch.arange(matrix_k.size(0)).unsqueeze(1), rank_k] = 1
            consistency = matrix_k * label_matrix
            num_correct = torch.sum(consistency, dim=1)
            num_predicted = torch.sum(matrix_k, dim=1)            
            num_total = torch.sum(label_matrix, dim=1)
            recall = torch.mean(num_correct / (num_total+1e-5))
            precision = torch.mean(num_correct / num_predicted)
            metrics[f"Real2Sketch_R@{k}"] += recall * len(feats)
            metrics[f"Real2Sketch_P@{k}"] += precision * len(feats)

            #calculate upperbound
            """
            gt_rank_k = gt_ranking[:, :k]
            gt_matrix_k = torch.zeros_like(label_matrix)
            gt_matrix_k[torch.arange(gt_matrix_k.size(0)).unsqueeze(1), gt_rank_k] = 1
            gt_consistency = gt_matrix_k * label_matrix
            gt_num_correct = torch.sum(gt_consistency, dim=1)
            gt_recall = torch.mean(gt_num_correct / (num_total+1e-5))
            metrics[f"upper_R@{k}"] += gt_recall * len(feats)
            """
    for k in [1, 5, 10, 50, 100, 200]:
        metrics[f"Real2Sketch_R@{k}"] /= len(query_features)
        metrics[f"Real2Sketch_P@{k}"] /= len(query_features)
        #metrics[f"upper_R@{k}"] /= len(query_features)
    return metrics



@torch.no_grad()
def circo_generate_val_predictions(model, img2text, relative_val_dataset,args) -> Tuple[torch.Tensor, List[str], list]:
    #Generates features predictions for the validation set of CIRCO
    # Create the data loader
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=128, num_workers=16,
                                     pin_memory=False, collate_fn=collate_fn, shuffle=False)

    predicted_features_list = []
    target_names_list = []
    gts_img_ids_list = []

    dictionary_embeddings, concept_texts = get_dict_embedding(model,args)

    id_split = tokenize(["*"])[0][1]
    # Compute the features
    for batch in tqdm(relative_val_loader):
        reference_names = batch['reference_name']
        target_names = batch['target_name']
        relative_captions = batch['relative_caption']
        gt_img_ids = batch['gt_img_ids']
        reference_images = batch['reference_image']
        
        gt_img_ids = np.array(gt_img_ids).T.tolist()
        input_captions = [f"a photo of * that {caption}" for caption in relative_captions]
        """
        batch_tokens = torch.vstack([pseudo_tokens[ref_names_list.index(ref)].unsqueeze(0) for ref in reference_names])
        tokenized_input_captions = clip.tokenize(input_captions, context_length=77).to(device)
        text_features = encode_with_pseudo_tokens(clip_model, tokenized_input_captions, batch_tokens)
        predicted_features = F.normalize(text_features)
        """
        
        # calculate the composed feature
        text = tokenize(input_captions)
        text = text.cuda(args.gpu, non_blocking=True)
        reference_images = reference_images.cuda(args.gpu, non_blocking=True)
        query_image_features = model.encode_image(reference_images)
        query_image_tokens = img2text(query_image_features)
        composed_feature = model.encode_text_img_retrieval(text, query_image_tokens, split_ind=id_split, repeat=False) 
        composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)

        logits_per_image = (query_image_features @ dictionary_embeddings.t()).detach().cpu()     
        predicted_indices = torch.argmax(logits_per_image, dim=1)
        #pdb.set_trace()
        gt_text = [cap.replace("*",concept_texts[index.item()]) for index,cap in zip(predicted_indices,input_captions)] # this is prediction text
        #gt_text = [p.replace("*",label_text_dict[str(label.item())].replace("_", " ")) for label in labels] # this is gt text
        #pdb.set_trace()
        gt_text = tokenize(gt_text)#.view(1, -1)            
        gt_text = gt_text.cuda(args.gpu, non_blocking=True)                        
        gt_text_features = model.encode_text(gt_text)
        gt_text_features_normed = gt_text_features / gt_text_features.norm(dim=-1, keepdim=True)


        
        predicted_features_list.append(query_image_features+gt_text_features_normed)
        #predicted_features_list.append(gt_text_features_normed)
        target_names_list.extend(target_names)
        gts_img_ids_list.extend(gt_img_ids)


    predicted_features = torch.vstack(predicted_features_list)

    return predicted_features, target_names_list, gts_img_ids_list

@torch.no_grad()
def circo_compute_val_metrics(relative_val_dataset, clip_model, img2text, index_features,
                              index_names,args) -> Dict[str, float]:
    #Compute the retrieval metrics on the CIRCO validation set given the dataset, pseudo tokens and the reference names
    # Generate the predicted features
    predicted_features, target_names, gts_img_ids = circo_generate_val_predictions(clip_model, img2text, relative_val_dataset,args)
    ap_at5 = []
    ap_at10 = []
    ap_at25 = []
    ap_at50 = []

    recall_at5 = []
    recall_at10 = []
    recall_at25 = []
    recall_at50 = []

    # Move the features to the device
    index_features = index_features.cuda(args.gpu, non_blocking=True)
    predicted_features = predicted_features.cuda(args.gpu, non_blocking=True)

    # Normalize the features
    index_features = F.normalize(index_features.float())

    for predicted_feature, target_name, gt_img_ids in tqdm(zip(predicted_features, target_names, gts_img_ids)):
        gt_img_ids = np.array(gt_img_ids)[
            np.array(gt_img_ids) != '']  # remove trailing empty strings added for collate_fn
        similarity = predicted_feature @ index_features.T
        sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]
        map_labels = torch.tensor(np.isin(sorted_index_names, gt_img_ids), dtype=torch.uint8)
        precisions = torch.cumsum(map_labels, dim=0) * map_labels  # Consider only positions corresponding to GTs
        precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)  # Compute precision for each position

        ap_at5.append(float(torch.sum(precisions[:5]) / min(len(gt_img_ids), 5)))
        ap_at10.append(float(torch.sum(precisions[:10]) / min(len(gt_img_ids), 10)))
        ap_at25.append(float(torch.sum(precisions[:25]) / min(len(gt_img_ids), 25)))
        ap_at50.append(float(torch.sum(precisions[:50]) / min(len(gt_img_ids), 50)))

        assert target_name == gt_img_ids[0], f"Target name not in GTs {target_name} {gt_img_ids}"
        single_gt_labels = torch.tensor(sorted_index_names == target_name)
        recall_at5.append(float(torch.sum(single_gt_labels[:5])))
        recall_at10.append(float(torch.sum(single_gt_labels[:10])))
        recall_at25.append(float(torch.sum(single_gt_labels[:25])))
        recall_at50.append(float(torch.sum(single_gt_labels[:50])))

    map_at5 = np.mean(ap_at5) * 100
    map_at10 = np.mean(ap_at10) * 100
    map_at25 = np.mean(ap_at25) * 100
    map_at50 = np.mean(ap_at50) * 100
    recall_at5 = np.mean(recall_at5) * 100
    recall_at10 = np.mean(recall_at10) * 100
    recall_at25 = np.mean(recall_at25) * 100
    recall_at50 = np.mean(recall_at50) * 100

    return {
        'circo_map_at5': map_at5,
        'circo_map_at10': map_at10,
        'circo_map_at25': map_at25,
        'circo_map_at50': map_at50,
        'circo_recall_at5': recall_at5,
        'circo_recall_at10': recall_at10,
        'circo_recall_at25': recall_at25,
        'circo_recall_at50': recall_at50,
    }


@torch.no_grad()
def circo_val_retrieval(dataset_path, model, img2text, preprocess: callable, args) -> Dict[str, float]:
    #Compute the retrieval metrics on the CIRCO validation set given the pseudo tokens and the reference names
    # model parameter represents the loaded CLIP model
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    model = model.module if args.distributed or args.dp else model
    logit_scale = model.logit_scale.exp()
    logit_scale = logit_scale.mean() 
    # Extract the index features
    classic_val_dataset = CIRCODataset(dataset_path, 'val', 'classic', preprocess)
    index_features, index_names = extract_image_features(classic_val_dataset, model, args)

    # Define the relative validation dataset
    relative_val_dataset = CIRCODataset(dataset_path, 'val', 'relative', preprocess)

    return circo_compute_val_metrics(relative_val_dataset, model, img2text, index_features, index_names,args)


@torch.no_grad()
def extract_image_features(dataset, clip_model, args, batch_size: Optional[int] = 128,
                           num_workers: Optional[int] = 16) -> Tuple[torch.Tensor, List[str]]:
    """
    Extracts image features from a dataset using a CLIP model.
    """
    # Create data loader
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    index_features = []
    index_names = []
    try:
        print(f"extracting image features {dataset.__class__.__name__} - {dataset.split}")
    except Exception as e:
        pass

    # Extract features
    for batch in tqdm(loader):
        images = batch.get('image')
        names = batch.get('image_name')
        if images is None:
            images = batch.get('reference_image')
        if names is None:
            names = batch.get('reference_name')

        images = images.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            batch_features = clip_model.encode_image(images)
            index_features.append(batch_features.cpu())
            index_names.extend(names)

    index_features = torch.vstack(index_features)
    return index_features, index_names