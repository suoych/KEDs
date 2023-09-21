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


def get_loss_img2text(model, img2text, images, caps, loss_img, loss_txt, args, memory=None):
    #with torch.no_grad():
    #    image_features = model.encode_image(images)
        #kv_image_features = model.visual.get_tokens(images)
        #kv_image_features = model.visual.ln_post(kv_image_features)
        #kv_image_features = kv_image_features @ model.visual.proj
    #token_features = img2text(image_features)
    #text_features = get_text_features(model, token_features, args)

    #text_p = tokenize("a photo of")
    with torch.no_grad():
        text_p = tokenize(caps,truncate=True)
        text_p = text_p.cuda(args.gpu, non_blocking=True)
        text_p = model.encode_text(text_p)
        image_features = text_p # calculate contrastive loss with text 
    text_p = img2text(text_p)
    #text_p = text_p.view(1, -1)
    #text_p = text_p.repeat(kv_image_features.size(0), 1)

    #text_features, collect_ind = get_text_attention_features(model, kv_image_features, args)
    #text_features = img2text(text_features, kv_image_features) # fuse use late cross attention
    text_features = model.get_visual_composed_features(text_p, images, img2text)

    #text_features = text_features[torch.arange(text_features.size(0)), collect_ind+1] @ model.text_projection


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


def train(model, img2text, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
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
                total_loss = get_loss_img2text(m, img2text, images, caps, loss_img, loss_txt, args, data_identifier)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        else:
            total_loss = get_loss_img2text(m, img2text, images, caps, loss_img, loss_txt, args, data_identifier)
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

