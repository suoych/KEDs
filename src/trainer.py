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

def get_loss(model, images, texts, loss_img, loss_txt, args, data_identifier=-1):
    if data_identifier == 1:
        # ImageNet dataset
        image_features, text_features, logit_scale = model(images, texts, extra=True)
    else:
        image_features, text_features, logit_scale = model(images, texts)
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
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

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

        # this is needed to send gradients back everywhere.
        # Image loss.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        logits_per_text = logits_per_image.t()
        loss_txt_val = loss_txt(logits_per_text, ground_truth)
    else:
        ground_truth = torch.arange(len(image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

        # Image loss.
        logits_per_image = logit_scale * image_features @ text_features.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        logits_per_text = logit_scale * text_features @ image_features.t()
        loss_txt_val = loss_txt(logits_per_text, ground_truth)

    total_loss = (loss_img_val + loss_txt_val) / 2
    return total_loss


def get_text_features(model, token_features, args):
    text = tokenize("a photo of")
    text = text.cuda(args.gpu, non_blocking=True)
    text = text.view(1, -1)
    text = text.repeat(token_features.size(0), 1)
    text_features = model.encode_text_img(text, token_features)
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
        image_base, text_base, image_gpu_index, text_gpu_index = database[0], database[1],database[2], database[3]
        
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

    else:
        image_base, text_base = database[0], database[1]

        feature = feature / feature.norm(dim=1, keepdim=True)
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

def get_loss_img2text(model, img2text,retrieval_fuse, images, caps, loss_img, loss_txt, args, database=None):
    with torch.no_grad():
        image_features = model.encode_image(images)
    topk_image_features,topk_text_features = get_retrieved_features(image_features,database,args)
    topk_image_features = topk_image_features.cuda(args.gpu, non_blocking=True)
    topk_text_features = topk_text_features.cuda(args.gpu, non_blocking=True)
    fused_features = retrieval_fuse(image_features.unsqueeze(1), topk_image_features,topk_image_features)
    #fused_features = retrieval_fuse(image_features.unsqueeze(1), topk_text_features,topk_text_features)
    fused_features = fused_features.squeeze(1) + image_features
    #token_features = img2text(image_features)
    token_features = img2text(fused_features)
    text_features = get_text_features(model, token_features, args)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

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
        """
        all_whole_image_features = torch.cat(
            [whole_image_features]
            + gathered_whole_image_features[:rank]
            + gathered_whole_image_features[rank + 1 :]
        )
        """
        
        ground_truth = torch.arange(len(all_image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

        # this is needed to send gradients back everywhere.
        # Image loss.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        logits_per_text = logits_per_image.t()
        loss_txt_val = loss_txt(logits_per_text, ground_truth)

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
    total_loss = (loss_img_val + loss_txt_val) / 2 #+ (loss_img_val_extra + loss_txt_val_extra) / 2
    return total_loss


def train(model, img2text,retrieval_fuse, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None, database=None):
    os.environ["WDS_EPOCH"] = str(epoch)
    model.eval()
    data['train'].set_epoch(epoch)
    dataloader, sampler = data['train'].dataloader,  data['train'].sampler
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    #if args.distributed and sampler is not None:
    #    sampler.set_epoch(epoch)
    

    num_batches_per_epoch = dataloader.num_batches

    end = time.time()
    #pdb.set_trace()
    i = 0
    for batch in dataloader:
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        #images, texts = batch['image_byte'], batch['caption'] # this is the webdataset format
        #print(images.shape)
        #print(len(batch))
        images, caps = batch[0], batch[1] # this is the original code
        #pdb.set_trace()
        if len(batch) == 3 and args.use_debiased_sampler:
            data_identifier = torch.unique(batch[2])[0].numpy()
        else:
            data_identifier = -1
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)


        data_time = time.time() - end

        m = model.module if args.distributed or args.dp else model

        #with profile(activities=[ProfilerActivity.CUDA],
        #profile_memory=True, record_shapes=True) as prof:
        # with automatic mixed precision.
        if args.precision == "amp" :#or args.precision == "fp16":
            with autocast():
                total_loss = get_loss_img2text(m, img2text, retrieval_fuse, images, caps, loss_img, loss_txt, args, database=database)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        else:
            total_loss = get_loss_img2text(m, img2text, retrieval_fuse, images, caps, loss_img, loss_txt, args, database=database)
            total_loss.backward()
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        #m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)
        #print(prof.key_averages().table(row_limit=10))

        batch_time = time.time() - end
        end = time.time()

        if is_master(args) and (i % 100) == 0:
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
    model = model.cpu()
    dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dinov2_vitb14 = dinov2_vitb14.cuda()
    dinov2_vitb14.eval()
    for batch in dataloader:
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images, caps, base_name = batch[0], batch[1], batch[2] # this is the original code
        #pdb.set_trace()
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        data_time = time.time() - end

        m = model.module if args.distributed or args.dp else model
        

        with torch.no_grad():
            image_features = dinov2_vitb14(images)
            #image_features = m.encode_image(images)
            #image_features = result_list[-2]
            #text_p = tokenize(caps,truncate=True)
            #text_p = text_p.cuda(args.gpu, non_blocking=True)
            #text_p = model.encode_text(text_p)
        
        for j in range(image_features.shape[0]):
            file_name = base_name[j] + '.pt'
            torch.save(image_features[j].clone(), os.path.join("/home/yucheng/cc_image_feature_folder", file_name))
            #torch.save(text_p[j].clone(), os.path.join("/home/yucheng/cc_text_feature_folder", file_name))


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

