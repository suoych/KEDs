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
import logging
import torch.nn.functional as F
from third_party.open_clip.clip import tokenize, _transform
import pickle
from utils import is_master
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Union, Dict, Literal,Tuple
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from data import CsvDataset, CustomFolder, ImageList, CsvCOCO, FashionIQ, CIRR, collate_fn
import pandas as pd
from sklearn.cluster import KMeans
import json
import copy


cap_dict = {}
with open("cc3m_have_good.pkl","rb") as f_g:
    img_cap_list_good = pickle.load(f_g)

with open("cc3m_have.pkl","rb") as f:
    img_cap_list = pickle.load(f)

for i in range(len(img_cap_list_good)):
    cap_dict[img_cap_list_good[i]['filename']] = img_cap_list_good[i]['text']

for i in range(len(img_cap_list)):
    cap_dict[img_cap_list[i]['filename']] = img_cap_list[i]['text']

def load_model_without_definition(args,img2text, retrieval_fuse, text_condition, location):
    if args.gpu is None:
        checkpoint = torch.load(location)
    else:
        # Map model to be loaded to specified single gpu.
        loc = "cuda:{}".format(args.gpu)
        checkpoint = torch.load(location, map_location=loc)
    sd_img2text = checkpoint["state_dict_img2text"]
    sd_retrieval_fuse = checkpoint["state_dict_retrieval_fuse"]
    sd_text_condition = checkpoint["state_dict_text_condition"]
    if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    if not args.distributed and next(iter(sd_img2text.items()))[0].startswith('module'):
        sd_img2text = {k[len('module.'):]: v for k, v in sd_img2text.items()}
    if not args.distributed and next(iter(sd_retrieval_fuse.items()))[0].startswith('module'):
        sd_retrieval_fuse = {k[len('module.'):]: v for k, v in sd_retrieval_fuse.items()}
    if not args.distributed and next(iter(sd_text_condition.items()))[0].startswith('module'):
        sd_text_condition = {k[len('module.'):]: v for k, v in sd_text_condition.items()}
    #with torch.no_grad():
    #    for a_param, b_param in zip(model.visual.parameters(), model.visual_mask.parameters()):
    #        b_param.copy_(a_param)
    img2text.load_state_dict(sd_img2text)
    retrieval_fuse.load_state_dict(sd_retrieval_fuse)
    text_condition.load_state_dict(sd_text_condition)
    logging.info(
        f"=> loaded checkpoint '{location}' (epoch {checkpoint['epoch']})"
    )
    return img2text, retrieval_fuse, text_condition

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

"""
def get_retrieved_features(feature, database,args,topk=16):
    #Retrieve features from database according to the given feature.
    
    image_base, text_base = database[0], database[1]

    feature = feature / feature.norm(dim=1, keepdim=True)
    image_base = image_base / image_base.norm(dim=1, keepdim=True)
    text_base = text_base / text_base.norm(dim=1, keepdim=True)

    #feature = feature.to(torch.float32).cpu()
    image_base = image_base.clone().to(feature.device)
    text_base = text_base.clone().to(feature.device)
    #print("feature device",feature.device," ","feature device",image_base.device," ""feature device",text_base.device)
    #image_base = image_base.cpu()
    #text_base = text_base.cpu()

    logits_per_source_image = feature @ image_base.t()
    logits_per_source_text = feature @ text_base.t()

    _, topk_image_indices = logits_per_source_image.topk(k=topk,dim=1) # get the indices
    b,k = topk_image_indices.shape[0],topk_image_indices.shape[1]
    topk_image_features = image_base[topk_image_indices.reshape(-1)]
    topk_image_features = topk_image_features.reshape(b,k,-1)

    _, topk_text_indices = logits_per_source_text.topk(k=topk,dim=1) # get the indices
    b,k = topk_text_indices.shape[0],topk_text_indices.shape[1]
    topk_text_features = text_base[topk_text_indices.reshape(-1)]
    topk_text_features = topk_text_features.reshape(b,k,-1)
    
    return topk_image_features, topk_text_features
"""

def get_retrieved_features(feature, database,args,topk=16,use_faiss=True):
    """
    Retrieve features from database according to the given feature. 
    By default we use faiss-gpu to boost inference speed.
    """
    if use_faiss:
        #image_base, text_base,basenames,subject_base, other_base, image_gpu_index, text_gpu_index, subject_gpu_index, other_gpu_index = database[0], database[1],database[2], database[3], database[4], database[5], database[6], database[7], database[8]
        image_base, text_base,basenames,  image_gpu_index, text_gpu_index = database[0], database[1],database[2], database[3], database[4]
        
        feature = feature / feature.norm(dim=1, keepdim=True)
        #image_base = image_base / image_base.norm(dim=1, keepdim=True)
        #text_base = text_base / text_base.norm(dim=1, keepdim=True)

        #image_base = image_base.clone().cpu().numpy()
        #text_base = text_base.clone().cpu().numpy()

        _, topk_image_indices = image_gpu_index.search(feature.clone().cpu().numpy(), topk) # search topk
        b,k = topk_image_indices.shape[0],topk_image_indices.shape[1]
        topk_image_features = image_base[topk_image_indices.reshape(-1)]
        topk_image_features = topk_image_features.reshape(b,k,-1)
        # random shuffle
        idx = torch.randperm(topk_image_features.shape[1])
        topk_image_features = topk_image_features[:,idx,:]

        _, topk_text_indices = text_gpu_index.search(feature.clone().cpu().numpy(), topk) # search topk
        b,k = topk_text_indices.shape[0],topk_text_indices.shape[1]
        topk_text_features = text_base[topk_text_indices.reshape(-1)]
        topk_text_features = topk_text_features.reshape(b,k,-1)

        topk_image_features = topk_image_features.clone().to(feature.device)
        topk_text_features = topk_text_features.clone().to(feature.device)

    
    return topk_image_features, topk_text_features


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
        
        
        # this part for late fusion cross-attention prediction
        composed_features, collect_ind = m.get_text_tokens(text)
        composed_features = img2text(img_feature.unsqueeze(1), composed_features) # fuse use late cross attention
        composed_feature = composed_features[:,0] @ m.text_projection

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


def evaluate_imgnet_retrieval(model, img2text, retrieval_fuse, text_condition,database, args, prompt, query_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    retrieval_fuse.eval()
    text_condition.eval()
    all_image_features = []  
    all_target_labels = []      
    m = model.module if args.distributed or args.dp else model
    n_class = 1000
    with open("./imgnet_class_label_mapping.txt","r") as f_d:
        lines = f_d.readlines()
        label_text_dict = {line.split()[0]: line.split()[1] for line in lines if line.strip()}


    #dictionary_embeddings, concept_texts = get_dict_embedding(m,args)
    with torch.no_grad():
        
        
        # just a test of images classification and tsne visualization
        label_text_origin = [value.replace("_", " ") for value in list(label_text_dict.values())]
        label_text = tokenize(label_text_origin)#.view(1, -1)            
        label_text = label_text.cuda(args.gpu, non_blocking=True)                        
        label_text_features = m.encode_text(label_text)
        label_text_features_normed = label_text_features / label_text_features.norm(dim=-1, keepdim=True)
        

        ## Extract target image features. 
        for batch in tqdm(target_loader):
            images, labels, basename = batch
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                labels = labels.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            all_image_features.append(image_features)
            all_target_labels.append(labels)
            logit_scale = m.logit_scale.exp()
            logit_scale = logit_scale.mean() 

        
        for j in range(5,10):
            location = f"./image_branch/checkpoints/epoch_{2*j-1}.pt"
            img2text, retrieval_fuse, text_condition = load_model_without_definition(args,img2text, retrieval_fuse, text_condition, location) 
            img2text_tb = copy.deepcopy(img2text) 
            retrieval_fuse_tb = copy.deepcopy(retrieval_fuse) 
            text_condition_tb = copy.deepcopy(text_condition)
            text_location = f"./text_branch/checkpoints/epoch_{2*j}.pt"
            img2text_tb, retrieval_fuse_tb, text_condition_tb = load_model_without_definition(args,img2text_tb, retrieval_fuse_tb, text_condition_tb, text_location) 
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
                    images, labels,basename = batch
                    if args.gpu is not None:
                        images = images.cuda(args.gpu, non_blocking=True)
                        labels = labels.cuda(args.gpu, non_blocking=True)

                    image_features = m.encode_image(images) 


                    ## Label is decided by class label and images' domain
                    labels += n_class * p_ind

                   
                    ## Composed feature extraction
                    topk_image,topk_text = get_retrieved_features(image_features,database,args)
                    topk_image = topk_image.cuda(args.gpu, non_blocking=True)
                    topk_text = topk_text.cuda(args.gpu, non_blocking=True)

                    mapped_features = img2text(image_features)
                    topk_image_features = img2text(topk_image)
                    topk_text_features = img2text(topk_text)
                    #cap_feature = img2text(caption_features)
                    fused_features = retrieval_fuse(mapped_features.unsqueeze(1), topk_image_features,topk_image_features)
                    text_conditioned = text_condition(mapped_features.unsqueeze(1), topk_text_features,topk_text_features)

                    #fused_features = torch.cat([fused_features,text_conditioned,mapped_features.unsqueeze(1),cap_feature.unsqueeze(1)],dim=1)
                    fused_features = torch.cat([fused_features,text_conditioned,mapped_features.unsqueeze(1)],dim=1)

                    #fused_features = retrieval_fuse(image_features.unsqueeze(1), topk_image_features,topk_image_features)
                    #text_conditioned = text_condition(image_features.unsqueeze(1), topk_text_features,topk_text_features)
                    #fused_features = torch.cat([fused_features,text_conditioned,image_features.unsqueeze(1),text_only_features.unsqueeze(1)],dim=1)
                    #token_features = img2text(image_features)


                    #image_features_query = img2text(fused_features)        
                    image_features_query = fused_features                 
                    composed_feature = m.encode_text_img_retrieval(text, image_features_query, split_ind=id_split) 

                    #text branch
                    mapped_features_tb = img2text_tb(image_features)
                    topk_image_features_tb = img2text_tb(topk_image)
                    topk_text_features_tb = img2text_tb(topk_text)
                    #fused_features_tb = retrieval_fuse_tb(query_image_features.unsqueeze(1), topk_image_features_tb,topk_image_features_tb)
                    #text_conditioned_tb = text_condition_tb(query_image_features.unsqueeze(1), topk_text_features_tb,topk_text_features_tb)
                    fused_features_tb = retrieval_fuse_tb(mapped_features_tb.unsqueeze(1), topk_image_features_tb,topk_image_features_tb)
                    text_conditioned_tb = text_condition_tb(mapped_features_tb.unsqueeze(1), topk_text_features_tb,topk_text_features_tb)
                    fused_features_tb = torch.cat([fused_features_tb,text_conditioned_tb,mapped_features_tb.unsqueeze(1)],dim=1)
                    query_image_tokens_tb = fused_features_tb                    
                    composed_feature_tb = m.encode_text_img_retrieval(text, query_image_tokens_tb, split_ind=id_split) 
                    image_features = composed_feature_tb


                    composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)  
                    ## average of image and text features
                    #mixture_features = image_features + text_only_features_normed
                    mixture_features = 0.1* j * image_features + (1- 0.1 * j) * composed_feature 
                    mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)       

                    all_text_features.append(text_only_features_normed.repeat((image_features.shape[0], 1)))
                    all_query_features.append(composed_feature)
                    all_query_image_features.append(image_features)
                    all_query_mixture_features.append(mixture_features)

                    #all_gt_text_features.append(gt_text_features_normed)
                    
                    all_query_labels.append(labels)

                metric_func = partial(get_metrics_imgnet, 
                    image_features=torch.cat(all_image_features), 
                    query_labels=torch.cat(all_query_labels),
                    target_labels=torch.cat(all_target_labels),
                    )

                feats = {'composed': torch.cat(all_query_features), 
                        'image': torch.cat(all_query_image_features),
                        #'text': torch.cat(all_text_features),
                        'mixture': torch.cat(all_query_mixture_features),
                        #'gt_text': torch.cat(all_gt_text_features),
                        }        

                for key, value in feats.items():
                    metrics = metric_func(query_features=value)
                    logging.info("Current prompt:"+ str(p) +"Eval {key} Feature")
                    logging.info(
                    f"Eval {key} Feature"
                    + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

    return metrics


def evaluate_coco(model, img2text, retrieval_fuse, text_condition,database, args, loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    retrieval_fuse.eval()
    text_condition.eval()

    all_image_features = []  
    all_query_image_features = []  
    all_mixture_features = []  
    all_composed_features_with_class = []  
    all_text_full_features = [] 

    m = model.module if args.distributed or args.dp else model
    #m = model
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean()
    with torch.no_grad():
        for j in range(1,26):
            print("j =",0.05*j)
            all_image_features = []  
            all_query_image_features = []  
            all_mixture_features = []  
            all_composed_features_with_class = []  
            all_text_full_features = []

            location = f"./image_branch/checkpoints/epoch_{2*j-1}.pt"
            img2text, retrieval_fuse, text_condition = load_model_without_definition(args,img2text, retrieval_fuse, text_condition, location) 
            img2text_tb = copy.deepcopy(img2text) 
            retrieval_fuse_tb = copy.deepcopy(retrieval_fuse) 
            text_condition_tb = copy.deepcopy(text_condition)
            text_location = f"./image_branch/checkpoints/epoch_{2*j}.pt"
            img2text_tb, retrieval_fuse_tb, text_condition_tb = load_model_without_definition(args,img2text_tb, retrieval_fuse_tb, text_condition_tb, text_location) 

            img2text_tb.eval()
            retrieval_fuse_tb.eval()
            text_condition_tb.eval()
            ## Extract query features 
            for batch in tqdm(loader):
                images, region_images, text_full, text_with_blank, text_with_blank_query, filename, raw_text, basename = batch            
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                    region_images = region_images.cuda(args.gpu, non_blocking=True)
                    text_full = text_full.cuda(args.gpu, non_blocking=True)
                    text_with_blank = text_with_blank.cuda(args.gpu, non_blocking=True)
                    text_with_blank_query = text_with_blank_query.cuda(args.gpu, non_blocking=True)
                ## Target image features 
                image_features = m.encode_image(images)             
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)  
                id_split = tokenize(["*"])[0][1]
                
                query_image_features = m.encode_image(region_images)
                #query_image_tokens = img2text(query_image_features)          
                #composed_feature_with_class = m.encode_text_img_retrieval(text_with_blank_query, query_image_tokens, split_ind=id_split, repeat=False)                        
                #composed_feature_with_class = composed_feature_with_class / composed_feature_with_class.norm(dim=-1, keepdim=True)    

                
                        
                ## Composed feature extraction
                topk_image,topk_text = get_retrieved_features(query_image_features,database,args)
                topk_image = topk_image.cuda(args.gpu, non_blocking=True)
                topk_text = topk_text.cuda(args.gpu, non_blocking=True)
                
                mapped_features = img2text(query_image_features)
                topk_image_features = img2text(topk_image)
                topk_text_features = img2text(topk_text)
                fused_features = retrieval_fuse(mapped_features.unsqueeze(1), topk_image_features,topk_image_features)
                text_conditioned = text_condition(mapped_features.unsqueeze(1), topk_text_features,topk_text_features)
                fused_features = torch.cat([fused_features,text_conditioned,mapped_features.unsqueeze(1)],dim=1)

                query_image_tokens  = fused_features                      
                composed_feature_with_class = m.encode_text_img_retrieval(text_with_blank_query, query_image_tokens, split_ind=id_split, repeat=False)                            
                


                #text branch
                mapped_features_tb = img2text_tb(query_image_features)
                topk_image_features_tb = img2text_tb(topk_image)
                topk_text_features_tb = img2text_tb(topk_text)
                #fused_features_tb = retrieval_fuse_tb(query_image_features.unsqueeze(1), topk_image_features_tb,topk_image_features_tb)
                #text_conditioned_tb = text_condition_tb(query_image_features.unsqueeze(1), topk_text_features_tb,topk_text_features_tb)
                fused_features_tb = retrieval_fuse_tb(mapped_features_tb.unsqueeze(1), topk_image_features_tb,topk_image_features_tb)
                text_conditioned_tb = text_condition_tb(mapped_features_tb.unsqueeze(1), topk_text_features_tb,topk_text_features_tb)
                fused_features_tb = torch.cat([fused_features_tb,text_conditioned_tb,mapped_features_tb.unsqueeze(1)],dim=1)
                query_image_tokens_tb = fused_features_tb                   
                #composed_feature_tb = m.encode_text_img_retrieval(text_with_blank, query_image_tokens_tb, split_ind=id_split, repeat=False)  
                composed_feature_tb = m.encode_text_img_retrieval(text_with_blank_query, query_image_tokens_tb, split_ind=id_split, repeat=False) 
                
                query_image_features = composed_feature_tb
                composed_feature_with_class = composed_feature_with_class / composed_feature_with_class.norm(dim=-1, keepdim=True)
                ## Text only features
                text_full_features = m.encode_text(text_full)
                text_full_features = text_full_features / text_full_features.norm(dim=-1, keepdim=True)            
                ## Query only features
                query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)                               
                ## Mixed features
                #mixture_features = query_image_features + text_full_features
                mixture_features = 0.05* j * query_image_features + (1- 0.05 * j) * composed_feature_with_class
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
                    #'text': torch.cat(all_text_full_features),
                    'mixture': torch.cat(all_mixture_features)}        

            for key, value in feats.items():
                metrics = metric_func(ref_features=value)
                logging.info(
                f"Eval {key} Feature"
                + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

    return metrics


def evaluate_cirr(model, img2text, retrieval_fuse, text_condition,database, args, query_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    retrieval_fuse.eval()
    text_condition.eval()

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


    #dictionary_embeddings, concept_texts = get_dict_embedding(m,args)

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
            all_target_global = all_target_paths

        for j in range(1,31):
            location = f"./image_branch/checkpoints/epoch_{j}.pt"
            img2text, retrieval_fuse, text_condition = load_model_without_definition(args,img2text, retrieval_fuse, text_condition, location) 
            img2text_tb = copy.deepcopy(img2text) 
            retrieval_fuse_tb = copy.deepcopy(retrieval_fuse) 
            text_condition_tb = copy.deepcopy(text_condition)
            text_location = f"./text_branch/checkpoints/epoch_{j}.pt"
            img2text_tb, retrieval_fuse_tb, text_condition_tb = load_model_without_definition(args,img2text_tb, retrieval_fuse_tb, text_condition_tb, text_location) 
            
            all_query_image_features = []  
            all_composed_features = []  
            all_mixture_features = []  
            all_caption_features = [] 
            all_gt_text_features = [] 
            all_target_paths = all_target_global
            all_ref_paths = []
            all_answer_paths = []
            all_raw_captions = []
            all_modi_dict = {}
            for batch in tqdm(query_loader):
                ref_images, text_with_blank, caption_only, ref_paths, answer_paths, raw_captions, target_cap = batch
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
                
                

                caption_features = m.encode_text(caption_only)
                ## Composed features
                query_image_features = m.encode_image(ref_images)

                ## Composed feature extraction
                topk_image,topk_text = get_retrieved_features(query_image_features,database,args)
                topk_image = topk_image.cuda(args.gpu, non_blocking=True)
                topk_text = topk_text.cuda(args.gpu, non_blocking=True)
                
                mapped_features = img2text(query_image_features)
                topk_image_features = img2text(topk_image)
                topk_text_features = img2text(topk_text)
                #cap_feature = img2text(caption_features)
                #fused_features = retrieval_fuse(query_image_features.unsqueeze(1), topk_image_features,topk_image_features)
                #text_conditioned = text_condition(query_image_features.unsqueeze(1), topk_text_features,topk_text_features)
                fused_features = retrieval_fuse(mapped_features.unsqueeze(1), topk_image_features,topk_image_features)
                text_conditioned = text_condition(mapped_features.unsqueeze(1), topk_text_features,topk_text_features)
                

                #fused_features = torch.cat([text_conditioned,mapped_features.unsqueeze(1)],dim=1)
                fused_features = torch.cat([fused_features,text_conditioned,mapped_features.unsqueeze(1)],dim=1)
                #fused_features = mapped_features.unsqueeze(1)

                #fused_features = retrieval_fuse(query_image_features.unsqueeze(1), topk_image_features,topk_image_features)
                #text_conditioned = text_condition(query_image_features.unsqueeze(1), topk_text_features,topk_text_features)
                #fused_features = torch.cat([fused_features,text_conditioned,query_image_features.unsqueeze(1),caption_features.unsqueeze(1)],dim=1)

                #fused_features = torch.cat([fused_features,text_conditioned,query_image_features.unsqueeze(1),caption_features.unsqueeze(1)],dim=1)

                query_image_tokens = fused_features 
                #query_image_tokens  = img2text(fused_features)                      
                composed_feature = m.encode_text_img_retrieval(text_with_blank, query_image_tokens, split_ind=id_split, repeat=False)     
                
                #text branch
                mapped_features_tb = img2text_tb(query_image_features)
                topk_image_features_tb = img2text_tb(topk_image)
                topk_text_features_tb = img2text_tb(topk_text)
                #fused_features_tb = retrieval_fuse_tb(query_image_features.unsqueeze(1), topk_image_features_tb,topk_image_features_tb)
                #text_conditioned_tb = text_condition_tb(query_image_features.unsqueeze(1), topk_text_features_tb,topk_text_features_tb)
                fused_features_tb = retrieval_fuse_tb(mapped_features_tb.unsqueeze(1), topk_image_features_tb,topk_image_features_tb)
                text_conditioned_tb = text_condition_tb(mapped_features_tb.unsqueeze(1), topk_text_features_tb,topk_text_features_tb)
                fused_features_tb = torch.cat([fused_features_tb,text_conditioned_tb,mapped_features_tb.unsqueeze(1)],dim=1)
                query_image_tokens_tb = fused_features_tb                    
                composed_feature_tb = m.encode_text_img_retrieval(text_with_blank, query_image_tokens_tb, split_ind=id_split, repeat=False) 



                query_image_features = composed_feature_tb
                #caption_features = 0.05 * i * composed_feature + (1- 0.05 * i)* composed_feature_tb
                #composed_feature = all_llm_cap
                            

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
                caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)                       
                query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)   
                composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)      
                #mixture_features = 0.1* j * query_image_features + (1- 0.1 * j) * composed_feature 
                mixture_features = 0.5 * query_image_features + 0.5 * composed_feature           
                mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
                all_caption_features.append(caption_features)
                all_query_image_features.append(query_image_features)
                all_composed_features.append(composed_feature)            
                all_mixture_features.append(mixture_features) 
                #all_gt_text_features.append(gt_text_features_normed)
                    

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
                    #'text': torch.cat(all_caption_features),
                    'mixture': torch.cat(all_mixture_features),
                    #'gt_text': torch.cat(all_gt_text_features)
                    }
            
            for key, value in feats.items():
                metrics = metric_func(ref_features=value)
                logging.info(
                f"Eval {key} Feature"
                + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
    return metrics


def evaluate_cirr_test(model, img2text, retrieval_fuse, text_condition,database, args, query_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    retrieval_fuse.eval()
    text_condition.eval()

    location = f"./image_branch/checkpoints/epoch_10.pt"
    img2text, retrieval_fuse, text_condition = load_model_without_definition(args,img2text, retrieval_fuse, text_condition, location) 
    img2text_tb = copy.deepcopy(img2text) 
    retrieval_fuse_tb = copy.deepcopy(retrieval_fuse) 
    text_condition_tb = copy.deepcopy(text_condition)
    text_location = f"./text_branch/checkpoints/epoch_13.pt"
    img2text_tb, retrieval_fuse_tb, text_condition_tb = load_model_without_definition(args,img2text_tb, retrieval_fuse_tb, text_condition_tb, text_location) 
    
    
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


            topk_image,topk_text = get_retrieved_features(query_image_features,database,args)
            topk_image = topk_image.cuda(args.gpu, non_blocking=True)
            topk_text = topk_text.cuda(args.gpu, non_blocking=True)
            
            mapped_features = img2text(query_image_features)
            topk_image_features = img2text(topk_image)
            topk_text_features = img2text(topk_text)
            fused_features = retrieval_fuse(mapped_features.unsqueeze(1), topk_image_features,topk_image_features)
            text_conditioned = text_condition(mapped_features.unsqueeze(1), topk_text_features,topk_text_features)
            fused_features = torch.cat([fused_features,text_conditioned,mapped_features.unsqueeze(1)],dim=1)

            query_image_tokens = fused_features                   
            composed_feature = m.encode_text_img_retrieval(text_with_blank, query_image_tokens, split_ind=id_split, repeat=False)     
            
            #text branch
            mapped_features_tb = img2text_tb(query_image_features)
            topk_image_features_tb = img2text_tb(topk_image)
            topk_text_features_tb = img2text_tb(topk_text)
            #fused_features_tb = retrieval_fuse_tb(query_image_features.unsqueeze(1), topk_image_features_tb,topk_image_features_tb)
            #text_conditioned_tb = text_condition_tb(query_image_features.unsqueeze(1), topk_text_features_tb,topk_text_features_tb)
            fused_features_tb = retrieval_fuse_tb(mapped_features_tb.unsqueeze(1), topk_image_features_tb,topk_image_features_tb)
            text_conditioned_tb = text_condition_tb(mapped_features_tb.unsqueeze(1), topk_text_features_tb,topk_text_features_tb)
            fused_features_tb = torch.cat([fused_features_tb,text_conditioned_tb,mapped_features_tb.unsqueeze(1)],dim=1)
            query_image_tokens_tb = fused_features_tb                    
            composed_feature_tb = m.encode_text_img_retrieval(text_with_blank, query_image_tokens_tb, split_ind=id_split, repeat=False) 
            
            query_image_features = composed_feature_tb
            #composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)
            #composed_feature = composed_feature + 0.05 * caption_features

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)                       
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)   
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)            
            #mixture_features = query_image_features + caption_features
            mixture_features = 0.5 * query_image_features + 0.5 * composed_feature    
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
               
        for key, value in feats.items():
            res_all[key] = metrics_func(ref_features=value)
    return res_all


def evaluate_fashion(model, img2text, retrieval_fuse, text_condition,database, args, source_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    retrieval_fuse.eval()
    text_condition.eval()
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

    #dictionary_embeddings, concept_texts = get_dict_embedding(m,args)

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
    
    all_target_global = all_target_paths
    all_image_features_global = all_image_features

    for j in range(1,16):
        location = f"./image_branch/checkpoints/epoch_{2*j-1}.pt"
        img2text, retrieval_fuse, text_condition = load_model_without_definition(args,img2text, retrieval_fuse, text_condition, location) 
        img2text_tb = copy.deepcopy(img2text) 
        retrieval_fuse_tb = copy.deepcopy(retrieval_fuse) 
        text_condition_tb = copy.deepcopy(text_condition)
        
        text_location = f"./text_branch/checkpoints/epoch_{2*j}.pt"
        img2text_tb, retrieval_fuse_tb, text_condition_tb = load_model_without_definition(args,img2text_tb, retrieval_fuse_tb, text_condition_tb, text_location) 
        
        all_query_image_features = []  
        all_composed_features = []  
        all_mixture_features = []  
        all_caption_features = [] 
        all_gt_text_features = [] 
        all_target_paths = all_target_global
        all_image_features = all_image_features_global 
        all_answer_paths = []
        all_captions = []
        all_modi_dict = {}
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
                
                
                ## Composed feature extraction
                topk_image,topk_text = get_retrieved_features(query_image_features,database,args)
                topk_image = topk_image.cuda(args.gpu, non_blocking=True)
                topk_text = topk_text.cuda(args.gpu, non_blocking=True)

                mapped_features = img2text(query_image_features)
                topk_image_features = img2text(topk_image)
                #cap_feature = img2text(caption_features)

                fused_features = retrieval_fuse(mapped_features.unsqueeze(1), topk_image_features,topk_image_features)
                #text_conditioned = text_condition(mapped_features.unsqueeze(1), topk_text_features,topk_text_features)
                #fused_features = torch.cat([fused_features,text_conditioned,mapped_features.unsqueeze(1)],dim=1)
                fused_features = torch.cat([fused_features,mapped_features.unsqueeze(1)],dim=1)
                query_image_tokens = fused_features

                #fused_features = retrieval_fuse(query_image_features.unsqueeze(1), topk_image_features,topk_image_features)
                #fused_features = fused_features.squeeze(1) + query_image_features + caption_features
                #query_image_tokens  = img2text(fused_features)                      
                composed_feature = m.encode_text_img_train(target_caption, query_image_tokens, split_ind=id_split, repeat=False)                            
                #composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)
                

                #text branch
                mapped_features_tb = img2text_tb(query_image_features)
                topk_image_features_tb = img2text_tb(topk_image)
                #topk_text_features_tb = img2text_tb(topk_text)
                fused_features_tb = retrieval_fuse_tb(mapped_features_tb.unsqueeze(1), topk_image_features_tb,topk_image_features_tb)
                #text_conditioned_tb = text_condition_tb(mapped_features_tb.unsqueeze(1), topk_text_features_tb,topk_text_features_tb)
                #fused_features_tb = torch.cat([fused_features_tb,text_conditioned_tb,mapped_features_tb.unsqueeze(1)],dim=1)
                fused_features_tb = torch.cat([fused_features_tb,mapped_features_tb.unsqueeze(1)],dim=1)
                query_image_tokens_tb = fused_features_tb                    
                composed_feature_tb = m.encode_text_img_train(target_caption, query_image_tokens_tb, split_ind=id_split, repeat=False) 
                
                query_image_features = composed_feature_tb

                composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
                caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)                       
                query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)   
                mixture_features = 0.05* j * query_image_features + (1- 0.05 * j) * composed_feature  
                #mixture_features = 0.3 * query_image_features + 0.7 * composed_feature   
                #mixture_features = 0.025*query_image_features + 0.975*caption_features
                mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
                

                all_caption_features.append(caption_features)
                all_query_image_features.append(query_image_features)
                all_composed_features.append(composed_feature)            
                all_mixture_features.append(mixture_features)     
                #all_gt_text_features.append(gt_text_features_normed)  

            metric_func = partial(get_metrics_fashion, 
                                image_features=torch.cat(all_image_features),
                                target_names=all_target_paths, answer_names=all_answer_paths)
            feats = {'composed': torch.cat(all_composed_features), 
                    'image': torch.cat(all_query_image_features),
                    #'text': torch.cat(all_caption_features),
                    'mixture': torch.cat(all_mixture_features),
                    #'gt_text': torch.cat(all_gt_text_features)
                    }
            
            
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
def extract_image_features(dataset, clip_model, args, batch_size: Optional[int] = 128,
                           num_workers: Optional[int] = 0) -> Tuple[torch.Tensor, List[str]]:
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

