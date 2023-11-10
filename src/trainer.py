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

from torch.cuda.amp import autocast
import torch.distributed as dist
from tqdm import tqdm
from torchvision.utils import save_image
import sys
import pdb
import wandb
import logging
import torch.nn.functional as F
from third_party.open_clip.clip import tokenize, _transform
from third_party.open_clip.simple_tokenizer import SimpleTokenizer
from utils import is_master
from torch.profiler import profile, record_function, ProfilerActivity
import faiss

with open("cc_subject.json","r") as f:
    subject_dict = json.load(f)
    
with open("cc_other.json","r") as f:
    other_dict = json.load(f)


def get_loss_img2text_image(model, img2text,retrieval_fuse, text_condition, images, capss, loss_img, loss_txt, loss_extra, args, database=None):
    caps, subject, other = capss[0], capss[1], capss[2]
    #caps = tokenize(caps)
    #caps = caps.cuda(args.gpu, non_blocking=True)
    with torch.no_grad():
        image_features = images
        ori_cap_feature = caps
        #image_features = model.encode_image(images)
        #cap_feature = model.encode_text(caps)
    
    topk_image_features,topk_text_features = get_retrieved_features(image_features,database,args)
    topk_image_features = topk_image_features.cuda(args.gpu, non_blocking=True)
    topk_text_features = topk_text_features.cuda(args.gpu, non_blocking=True)
    #pdb.set_trace()

    mapped_features = img2text(image_features)
    topk_image_features = img2text(topk_image_features)
    topk_text_features = img2text(topk_text_features)
    #cap_feature = img2text(ori_cap_feature)

    fused_features = retrieval_fuse(mapped_features.unsqueeze(1), topk_image_features,topk_image_features)
    text_conditioned = text_condition(mapped_features.unsqueeze(1), topk_text_features,topk_text_features)
    #fused_features = retrieval_fuse(image_features.unsqueeze(1), topk_text_features,topk_text_features)
    #fused_features = fused_features.squeeze(1) + text_conditioned.squeeze(1) #+ image_features + cap_feature

    fused_features = torch.cat([fused_features,text_conditioned,mapped_features.unsqueeze(1)],dim=1)#,cap_feature.unsqueeze(1)],dim=1)
    #fused_features = torch.cat([text_conditioned,mapped_features.unsqueeze(1)],dim=1)#,cap_feature.unsqueeze(1)],dim=1)
    #token_features = img2text(image_features)
    token_features = fused_features

    text_features = get_text_features(model, token_features, args)

    #other_embedded_features = get_cap_embedded_features(model, token_features,other, args)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    #other_embedded_features = other_embedded_features / other_embedded_features.norm(dim=-1, keepdim=True)
    #ori_cap_feature = ori_cap_feature / ori_cap_feature.norm(dim=-1, keepdim=True)

    logit_scale = model.logit_scale.exp()
    logit_scale = logit_scale.mean()
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        
        #gathered_whole_image_features = [
        #    torch.zeros_like(whole_image_features) for _ in range(world_size)
        #]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        #dist.all_gather(gathered_other_features, other_embedded_features)
        #dist.all_gather(gathered_ori_cap_features, ori_cap_feature)
        #dist.all_gather(gathered_whole_image_features, whole_image_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )
        
       
        
        ground_truth = torch.arange(len(all_image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)
        
        # cosine loss
        #target = torch.as_tensor([1])
        #if args.gpu is not None:
        #    target = target.cuda(args.gpu, non_blocking=True)
        #loss_img_val = loss_img(all_image_features, all_text_features, target)


        # this is needed to send gradients back everywhere.
        # Image loss.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        logits_per_text = logits_per_image.t()
        loss_txt_val = loss_txt(logits_per_text, ground_truth)

        # Text loss.
        #extra_target = torch.as_tensor([1])
        #if args.gpu is not None:
        #    extra_target = extra_target.cuda(args.gpu, non_blocking=True)
        #extra_loss = loss_extra(all_other_features, all_ori_gap_features, extra_target)

        #extra loss with whole image embedding
        #logits_per_image_extra = logit_scale * all_whole_image_features @ all_text_features.t()
        #loss_img_val_extra = loss_img(logits_per_image_extra, ground_truth)
        #logits_per_text_extra = logits_per_image_extra.t()
        #loss_txt_val_extra = loss_txt(logits_per_text_extra, ground_truth)
    
    else:
        ground_truth = torch.arange(len(image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)
        # Image loss.
        logits_per_image = logit_scale * image_features @ text_features.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        logits_per_text = logit_scale * text_features @ image_features.t()
        loss_txt_val = loss_txt(logits_per_text, ground_truth)
        # Text loss.
        extra_logits_per_image = logit_scale * ori_cap_feature @ other_embedded_features.t()
        extra_loss_img_val = loss_img(extra_logits_per_image, ground_truth)
        extra_logits_per_text = extra_logits_per_image.t()
        extra_loss_txt_val = loss_txt(extra_logits_per_text, ground_truth)

    total_loss = (loss_img_val + loss_txt_val) / 2 
    return total_loss


def get_text_features(model, token_features, args):
    text = tokenize("a photo of")
    text = text.cuda(args.gpu, non_blocking=True)
    text = text.view(1, -1)
    text = text.repeat(token_features.size(0), 1)
    text_features = model.encode_text_img(text, token_features)
    return text_features

def get_cap_embedded_features(model, token_features,cap, args):
    text = tokenize(cap, truncate=True)
    text = text.cuda(args.gpu, non_blocking=True)
    id_split = tokenize(["*"])[0][1]  
    #text = text.view(1, -1)
    #text = text.repeat(token_features.size(0), 1)
    text_features = model.encode_text_img_train(text, token_features,split_ind=id_split)
    #text_features = model.encode_text_img_retrieval(text, token_features,split_ind=id_split)
    return text_features

def get_text_attention_features(model, token_features, args):
    """
    transfer image token into cross attention.
    """
    text = tokenize("a photo of")
    text = text.cuda(args.gpu, non_blocking=True)
    text = text.view(1, -1)
    text = text.repeat(token_features.size(0), 1)
    #text_features, collect_ind = model.get_text_tokens(text)
    text_features, collect_ind, feature_list = model.get_text_mid_feature(text)
    return text_features, collect_ind

def get_retrieved_features(feature, database,args,topk=16,use_faiss=True):
    """
    Retrieve features from database according to the given feature. 
    By default we use faiss-gpu to boost inference speed.
    """
    if use_faiss:
        #image_base, text_base,basenames,subject_base, other_base, image_gpu_index, text_gpu_index, subject_gpu_index, other_gpu_index = database[0], database[1],database[2], database[3], database[4], database[5], database[6], database[7], database[8]
        image_base, text_base,basenames, image_gpu_index, text_gpu_index = database[0], database[1],database[2], database[3], database[4]
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
        # random shuffle
        #idx = torch.randperm(topk_text_features.shape[1])
        #topk_text_features = topk_text_features[:,idx,:]

        topk_image_features = topk_image_features.clone().to(feature.device)
        topk_text_features = topk_text_features.clone().to(feature.device)

    else:
        image_base, text_base = database[0], database[1]

        #feature = feature / feature.norm(dim=1, keepdim=True)
        #image_base = image_base / image_base.norm(dim=1, keepdim=True)
        #text_base = text_base / text_base.norm(dim=1, keepdim=True)

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


def get_extra_cap_features(feature, database,args,topk=2):
    """
    Retrieve features from database according to the given feature. 
    By default we use faiss-gpu to boost inference speed.
    """
    #image_base, text_base,basenames,subject_base, other_base, image_gpu_index, text_gpu_index, subject_gpu_index, other_gpu_index = database[0], database[1],database[2], database[3], database[4], database[5], database[6], database[7], database[8]
    image_base, text_base, basenames, image_gpu_index, text_gpu_index = database[0], database[1],database[2], database[3], database[4]
    feature = feature / feature.norm(dim=1, keepdim=True)

    _, topk_text_indices = text_gpu_index.search(feature.clone().cpu().numpy(), topk) # search topk
    b,k = topk_text_indices.shape[0],topk_text_indices.shape[1]
    
    #pdb.set_trace()
    topk_basenames = []
    for i in range(b):
        for j in range(k):
            topk_basenames.append(basenames[topk_text_indices[i][j]])

    topk_text_features = text_base[topk_text_indices.reshape(-1)]
    topk_text_features = topk_text_features.reshape(b,k,-1)
    topk_text_features = topk_text_features.clone().to(feature.device)
    
    return topk_text_features, topk_basenames

def get_loss_img2text(model, img2text,retrieval_fuse, text_condition, images, capss, loss_img, loss_txt, loss_extra, args, database=None):
    caps, subject, other = capss[0], capss[1], capss[2]
    #caps = tokenize(caps)
    #caps = caps.cuda(args.gpu, non_blocking=True)
    with torch.no_grad():
        image_features = images
        ori_cap_feature = caps
        #image_features = model.encode_image(images)
        #cap_feature = model.encode_text(caps)
    
    topk_image_features,topk_text_features = get_retrieved_features(image_features, database, args)
    topk_image_features = topk_image_features.cuda(args.gpu, non_blocking=True)
    topk_text_features = topk_text_features.cuda(args.gpu, non_blocking=True)
    #pdb.set_trace()

    mapped_features = img2text(image_features)
    topk_image_features = img2text(topk_image_features)
    #topk_text_features = img2text(topk_text_features)
    #cap_feature = img2text(ori_cap_feature)

    fused_features = retrieval_fuse(mapped_features.unsqueeze(1), topk_image_features,topk_image_features)
    #text_conditioned = text_condition(mapped_features.unsqueeze(1), topk_text_features,topk_text_features)
    #fused_features = retrieval_fuse(image_features.unsqueeze(1), topk_text_features,topk_text_features)
    #fused_features = fused_features.squeeze(1) + text_conditioned.squeeze(1) #+ image_features + cap_feature

    #fused_features = torch.cat([fused_features,text_conditioned,mapped_features.unsqueeze(1)],dim=1)#,cap_feature.unsqueeze(1)],dim=1)
    fused_features = torch.cat([fused_features,mapped_features.unsqueeze(1)],dim=1)#,cap_feature.unsqueeze(1)],dim=1)
    #token_features = img2text(image_features)
    token_features = fused_features

    other_embedded_features = get_cap_embedded_features(model, token_features,other, args)

    other_embedded_features = other_embedded_features / other_embedded_features.norm(dim=-1, keepdim=True)
    ori_cap_feature = ori_cap_feature / ori_cap_feature.norm(dim=-1, keepdim=True)
    
    
    top2_cap_embedding, top2_basenames = get_extra_cap_features(ori_cap_feature, database, args)
    #top2_extra_other = ["a photo of * * * " + other_dict[name.split(".")[0]].replace("*", " ") for name in top2_basenames]
    top2_extra_other = ["a photo of * * " + other_dict[name.split(".")[0]].replace("*", " ") for name in top2_basenames]
    b,l,d = token_features.shape
    top2_cap_embedding = top2_cap_embedding.reshape(2*b,-1)
    other_extra_embedded_features = get_cap_embedded_features(model, token_features.unsqueeze(1).repeat(1,2,1,1).reshape(2*b,l,d), top2_extra_other, args)
    
    other_extra_embedded_features = other_extra_embedded_features / other_extra_embedded_features.norm(dim=-1, keepdim=True)
    top2_cap_embedding = top2_cap_embedding / top2_cap_embedding.norm(dim=-1, keepdim=True)
    
    logit_scale = model.logit_scale.exp()
    logit_scale = logit_scale.mean()
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        gathered_other_features = [
            torch.zeros_like(other_embedded_features) for _ in range(world_size)
        ]
        gathered_ori_cap_features = [
            torch.zeros_like(ori_cap_feature) for _ in range(world_size)
        ]
        
        gathered_other_extra_features = [
            torch.zeros_like(other_extra_embedded_features) for _ in range(world_size)
        ]
        gathered_top2_cap_features = [
            torch.zeros_like(top2_cap_embedding) for _ in range(world_size)
        ]
        
        dist.all_gather(gathered_other_features, other_embedded_features)
        dist.all_gather(gathered_ori_cap_features, ori_cap_feature)
        dist.all_gather(gathered_other_extra_features, other_extra_embedded_features)
        dist.all_gather(gathered_top2_cap_features, top2_cap_embedding)
        
        all_other_features = torch.cat(
            [other_embedded_features]
            + gathered_other_features[:rank]
            + gathered_other_features[rank + 1 :]
        )
        all_ori_gap_features = torch.cat(
            [ori_cap_feature]
            + gathered_ori_cap_features[:rank]
            + gathered_ori_cap_features[rank + 1 :]
        )
        
        all_other_extra_features = torch.cat(
            [other_extra_embedded_features]
            + gathered_other_extra_features[:rank]
            + gathered_other_extra_features[rank + 1 :]
        )
        all_top2_gap_features = torch.cat(
            [top2_cap_embedding]
            + gathered_top2_cap_features[:rank]
            + gathered_top2_cap_features[rank + 1 :]
        )
        
        # Text loss.
        target = torch.as_tensor([1])
        if args.gpu is not None:
            target = target.cuda(args.gpu, non_blocking=True)
        loss = loss_extra(all_other_features, all_ori_gap_features, target)
        extra_loss = loss_extra(all_other_extra_features, all_top2_gap_features, target)


        #extra loss with whole image embedding
        #logits_per_image_extra = logit_scale * all_whole_image_features @ all_text_features.t()
        #loss_img_val_extra = loss_img(logits_per_image_extra, ground_truth)
        #logits_per_text_extra = logits_per_image_extra.t()
        #loss_txt_val_extra = loss_txt(logits_per_text_extra, ground_truth)
    
    else:
        ground_truth = torch.arange(len(image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)
        # Image loss.
        #logits_per_image = logit_scale * image_features @ text_features.t()
        #loss_img_val = loss_img(logits_per_image, ground_truth)
        #logits_per_text = logit_scale * text_features @ image_features.t()
        #loss_txt_val = loss_txt(logits_per_text, ground_truth)
        # Text loss.
        extra_target = torch.as_tensor([1])
        if args.gpu is not None:
            extra_target = extra_target.cuda(args.gpu, non_blocking=True)

        loss = loss_extra(other_extra_embedded_features, top2_cap_embedding, extra_target)
        extra_loss = loss_extra(other_embedded_features, ori_cap_feature, extra_target)
        
    #print("Loss:", loss, "Extra loss:", extra_loss)
    total_loss = loss + 0.5 * extra_loss #(loss_img_val + loss_txt_val) / 2 + 0.1 * extra_loss
    return total_loss


def train(model, img2text,retrieval_fuse, text_condition, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None, database=None):
    os.environ["WDS_EPOCH"] = str(epoch)
    model.eval()
    data['train'].set_epoch(epoch)
    dataloader, sampler = data['train'].dataloader,  data['train'].sampler
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    loss_extra = nn.CosineEmbeddingLoss()
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)
        loss_extra = loss_extra.cuda(args.gpu)

    #if args.distributed and sampler is not None:
    #    sampler.set_epoch(epoch)
    

    num_batches_per_epoch = dataloader.num_batches

    end = time.time()
    #pdb.set_trace()
    i = 0
    for batch in dataloader:
        step = num_batches_per_epoch * epoch + i
        #if scheduler is not None:
        scheduler(step)

        optimizer.zero_grad()

        #images, texts = batch['image_byte'], batch['caption'] # this is the webdataset format
        #print(images.shape)
        #print(len(batch))
        images, caps, subject, other = batch[0], batch[1], batch[2], batch[3] # this is the original code
        #pdb.set_trace()
        if len(batch) == 3 and args.use_debiased_sampler:
            data_identifier = torch.unique(batch[2])[0].numpy()
        else:
            data_identifier = -1
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            caps = caps.cuda(args.gpu, non_blocking=True)
        
        caps = (caps, subject, other)

        data_time = time.time() - end

        m = model.module if args.distributed or args.dp else model

        #with profile(activities=[ProfilerActivity.CUDA],
        #profile_memory=True, record_shapes=True) as prof:
        # with automatic mixed precision.
        if args.precision == "amp" :#or args.precision == "fp16":
            with autocast():
                #total_loss = get_loss_img2text(m, img2text, retrieval_fuse, text_condition, images, caps, loss_img, loss_txt, loss_extra, args, database=database)
                total_loss = get_loss_img2text_image(m, img2text, retrieval_fuse, text_condition, images, caps, loss_img, loss_txt, loss_extra, args, database=database)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        else:
            #total_loss = get_loss_img2text(m, img2text, retrieval_fuse, text_condition, images, caps, loss_img, loss_txt, loss_extra, args, database=database)
            total_loss = get_loss_img2text_image(m, img2text, retrieval_fuse, text_condition, images, caps, loss_img, loss_txt, loss_extra, args, database=database)
            total_loss.backward()
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        #m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)
        #print(prof.key_averages().table(row_limit=10))

        batch_time = time.time() - end
        end = time.time()

        if is_master(args) and (i % 500) == 0:
            num_samples = i * len(images) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            logging.info(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                f"Loss: {total_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:5f}\tlogit_scale {m.logit_scale.data:.3f}"
            )
            # save train loss / etc.

            timestep = epoch * num_batches_per_epoch + i
            log_data = {
                "loss": total_loss.item(),
                "data_time": data_time,
                "batch_time": batch_time,
                "scale":  m.logit_scale.data.item(),
                "lr": optimizer.param_groups[0]["lr"]
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, timestep)
                if args.wandb:
                    wandb.log({name: val, 'step': timestep})
        i+=1

def save_feature(model, img2text, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    os.environ["WDS_EPOCH"] = str(epoch)
    model.eval()
    data['train'].set_epoch(epoch)
    dataloader, sampler = data['train'].dataloader, data['train'].sampler

    num_batches_per_epoch = dataloader.num_batches

    end = time.time()
    i = 0
    #model = model.cpu()
    #dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    #dinov2_vitb14 = dinov2_vitb14.cuda()
    #dinov2_vitb14.eval()
    for batch in dataloader:
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        #images, caps, base_name = batch[0], batch[1], batch[2] # this is the original code
        images, caps, subject, other, base_name = batch[0], batch[1], batch[2], batch[3], batch[4]
        #pdb.set_trace()
        #if args.gpu is not None:
        #    images = images.cuda(args.gpu, non_blocking=True)

        data_time = time.time() - end

        m = model.module if args.distributed or args.dp else model
        

        with torch.no_grad():
            #image_features = dinov2_vitb14(images)
            #image_features = m.encode_image(images)
            #image_features = result_list[-2]
            #text_p = tokenize(caps,truncate=True)
            #text_p = text_p.cuda(args.gpu, non_blocking=True)
            #text_p = model.encode_text(text_p)
            text_s = tokenize(subject,truncate=True)
            text_s = text_s.cuda(args.gpu, non_blocking=True)
            text_s = model.encode_text(text_s)
            text_o = tokenize(other,truncate=True)
            text_o = text_o.cuda(args.gpu, non_blocking=True)
            text_o = model.encode_text(text_o)
        
        for j in range(images.shape[0]):
            file_name = base_name[j] + '.pt'
            #torch.save(image_features[j].clone(), os.path.join("/home/yucheng/cc_image_feature_folder", file_name))
            torch.save(text_s[j].clone(), os.path.join("/home/yucheng/cc_text_feature_folder_subject", file_name))
            torch.save(text_o[j].clone(), os.path.join("/home/yucheng/cc_text_feature_folder_other", file_name))


        batch_time = time.time() - end
        end = time.time()

        if is_master(args) and (i % 100) == 0:
            num_samples = i * len(images) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            logging.info(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
            )
            # save train loss / etc.

            timestep = epoch * num_batches_per_epoch + i
            log_data = {
                "data_time": data_time,
                "batch_time": batch_time,
                "scale":  m.logit_scale.data.item(),
                "lr": optimizer.param_groups[0]["lr"]
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, timestep)
                if args.wandb:
                    wandb.log({name: val, 'step': timestep})
        i+=1

