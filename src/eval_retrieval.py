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
import logging
from time import gmtime, strftime
from pathlib import Path
import json
from functools import partial
#import wandb
import torch
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.datasets as datasets
import torchvision.transforms as T
from PIL import Image

from model.clip import _transform, load
from model.model import convert_weights, CLIP, IM2TEXT,CrossFormer,T2I
from eval_utils import evaluate_imgnet_retrieval, evaluate_coco, evaluate_fashion, evaluate_cirr, evaluate_cirr_test
from data import CsvDataset, CustomFolder, ImageList, CsvCOCO, FashionIQ, CIRR,collate_fn, LoadDataBase
from params import parse_args, get_project_root
from logger import setup_primary_logging, setup_worker_logging
from utils import is_master, convert_models_to_fp32, TargetPad
import faiss
import copy
import numpy as np
import random


def seed_everything(seed):
    #if seed >= 10000:
    #    raise ValueError("seed number should be less than 10000")
    
    # we should set different seed for different gpu so that they would not generate same data batches
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 10) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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

def load_model(args):
    model, _, preprocess_val = load(
            args.model,
            jit=False)
    img2text = IM2TEXT(embed_dim=model.embed_dim, 
                       middle_dim=args.middle_dim, 
                       output_dim=model.token_embedding.weight.shape[1],
                       n_layer=args.n_layer)
    retrieval_fuse = CrossFormer(q_dim=model.token_embedding.weight.shape[1],k_dim=model.token_embedding.weight.shape[1],v_dim=model.token_embedding.weight.shape[1],num_layers = 3)
    text_condition = CrossFormer(q_dim=model.token_embedding.weight.shape[1],k_dim=model.token_embedding.weight.shape[1],v_dim=model.token_embedding.weight.shape[1],num_layers = 3)
    #img2text = CrossFormer(q_dim=model.visual.proj.shape[0],dim=model.token_embedding.weight.shape[1]) 
    #text_condition = T2I(embed_dim=model.token_embedding.weight.shape[1], 
    #                       middle_dim=args.middle_dim, 
    #                       output_dim=model.visual.proj.shape[0], 
    #                       n_layer=args.n_layer)
    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "amp" or args.precision == "fp32" or args.gpu is None:
        convert_models_to_fp32(model)

    if not torch.cuda.is_available():
        model.float()
        img2text.float()
        retrieval_fuse.float()
        text_condition.float()
        logging.warning("using CPU, this will be slow")
    else:
        model.cuda(args.gpu)
        img2text.cuda(args.gpu)
        retrieval_fuse.cuda(args.gpu)
        text_condition.cuda(args.gpu)
        if args.precision == "fp16":
            convert_weights(model)
            convert_weights(img2text)
            convert_weights(retrieval_fuse)
            convert_weights(text_condition)
        # Previously batch size and workers were global and not per GPU.
        # args.batch_size = args.batch_size / ngpus_per_node)
        # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        
        if args.distributed and args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, 
                device_ids=[args.gpu], 
                find_unused_parameters=model.has_extra)
            img2text = torch.nn.parallel.DistributedDataParallel(img2text, 
                device_ids=[args.gpu], find_unused_parameters=False)
            retrieval_fuse = torch.nn.parallel.DistributedDataParallel(retrieval_fuse, 
                device_ids=[args.gpu], find_unused_parameters=False)
            text_condition = torch.nn.parallel.DistributedDataParallel(text_condition, 
                device_ids=[args.gpu], find_unused_parameters=False)
        if args.dp:
            model = torch.nn.DataParallel(model, device_ids=args.multigpu)
            img2text = torch.nn.DataParallel(img2text, device_ids=args.multigpu)
            retrieval_fuse = torch.nn.DataParallel(retrieval_fuse, device_ids=args.multigpu)
            text_condition = torch.nn.DataParallel(text_condition, device_ids=args.multigpu)
        
        if args.precision == "fp16":
            convert_weights(model)
            convert_weights(img2text)
            convert_weights(retrieval_fuse)
            convert_weights(text_condition)
    if args.resume == 'auto':
        checkpoint_list = os.listdir(args.checkpoint_path)
        checkpoint_list = [ckpt for ckpt in checkpoint_list if ckpt.startswith('epoch')]
        if checkpoint_list:
            latest_epoch = max([int(ckpt.split('_')[1].split('.')[0]) for ckpt in checkpoint_list])
            args.resume = os.path.join(args.checkpoint_path, f'epoch_{latest_epoch}.pt')
        else:
            args.resume = None

    assert args.resume is not None
    if os.path.isfile(args.resume):
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            # Map model to be loaded to specified single gpu.
            loc = "cuda:{}".format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        sd = checkpoint["state_dict"]
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
        model.load_state_dict(sd,strict=False) # manually load weights for masked visual transformers
        #with torch.no_grad():
        #    for a_param, b_param in zip(model.visual.parameters(), model.visual_mask.parameters()):
        #        b_param.copy_(a_param)
        img2text.load_state_dict(sd_img2text)
        retrieval_fuse.load_state_dict(sd_retrieval_fuse)
        text_condition.load_state_dict(sd_text_condition)
        logging.info(
            f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})"
        )
    else:
        logging.info("=> no checkpoint found at '{}'".format(args.resume))
    return model, img2text, retrieval_fuse, text_condition, preprocess_val



def setup_log_save(args):
    if is_master(args):
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"{name}: {val}")
                f.write(f"{name}: {val}\n")
            
    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    if args.dp:
        args.batch_size *= args.world_size
    if args.gpu is not None:
        logging.info(f"Use GPU: {args.gpu} for training")
        torch.cuda.set_device(args.gpu)


def main_worker(gpu, ngpus_per_node, log_queue, args):
    args.gpu = gpu
    args.rank = gpu
    setup_worker_logging(args.rank, log_queue, args.log_level)
    # Log and save params.
    setup_log_save(args)
    # Load trained model
    model, img2text, retrieval_fuse,text_condition, preprocess_val = load_model(args)
    #cudnn.benchmark = True
    #cudnn.deterministic = False   
    cudnn.benchmark = False        # if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    #cudnn.benchmark = True
    #cudnn.deterministic = False
    seed_everything(args.seed)  
    torch.cuda.manual_seed_all(args.seed) 
    torch.use_deterministic_algorithms(True)
    #root_project = os.path.join(get_project_root(), 'data')
    root_project = "/home/comp_data"

    # We load database here.
    
    Base_dataset = LoadDataBase("/home/clip_cc_database_750000") #dino_cc_database")
    print("Loading databases!")
    dataloader = DataLoader(Base_dataset, batch_size=100, shuffle=False, num_workers=10)
    database = {}
    image_bases = []
    text_bases = []
    subject_bases = []
    other_bases = []
    """
    basenames = []
    for batch in dataloader:
        batch_cp = copy.deepcopy(batch)  
        del batch 
        #image_base, text_base, basename, subject_base, other_base = batch_cp[0], batch_cp[1], batch_cp[2], batch_cp[3], batch_cp[4]
        image_base, text_base, basename = batch_cp[0], batch_cp[1], batch_cp[2]#, batch_cp[3], batch_cp[4]
        image_bases.append(image_base)
        text_bases.append(text_base)
        #subject_bases.append(subject_base)
        #other_bases.append(other_base)
        for item in basename:
            basenames.append(item)
    image_bases = torch.cat(image_bases,dim=0)
    text_bases = torch.cat(text_bases,dim=0)
    #subject_bases = torch.cat(subject_bases,dim=0)
    #other_bases = torch.cat(other_bases,dim=0)
    image_bases = image_bases.cpu()
    text_bases = text_bases.cpu()
    #subject_bases = subject_bases.cpu()
    #other_bases = other_bases.cpu()
    image_bases = image_bases / image_bases.norm(dim=1, keepdim=True)
    text_bases = text_bases / text_bases.norm(dim=1, keepdim=True)
    #subject_bases = subject_bases / subject_bases.norm(dim=1, keepdim=True)
    #other_bases = other_bases / other_bases.norm(dim=1, keepdim=True)
    database = [image_bases,text_bases,basenames]#,subject_bases,other_bases]
    """
    print("Loading databases!")
    image_bases = torch.load("/home/cc_image_databases.pt",map_location="cpu")
    text_bases = torch.load("/home/cc_text_databases.pt",map_location="cpu")
    basenames = []
    with open("/home/database_names.txt", "r") as f:
        for line in f:
            basenames.append(line.strip())
    database = [image_bases,text_bases,basenames]#,subject_bases,other_bases]
    
    ngpus = faiss.get_num_gpus()
    print("number of GPUs:", ngpus)
    image_cpu_index = faiss.IndexFlatL2(768)
    image_gpu_index = faiss.index_cpu_to_all_gpus(image_cpu_index)
    image_gpu_index.add(image_bases.numpy())  # add vectors to the image_index
    text_cpu_index = faiss.IndexFlatL2(768)
    text_gpu_index = faiss.index_cpu_to_all_gpus(text_cpu_index)
    text_gpu_index.add(text_bases.numpy())  # add vectors to the image_index
    database.append(image_gpu_index)
    database.append(text_gpu_index)
    print("Loading databases done!")
    print("Loading databases done!")

    ## Padding option
    if args.target_pad:
        trans_tmp = preprocess_val.transforms
        trans_tmp = [TargetPad(1.25)] + trans_tmp
        preprocess_train = T.Compose(trans_tmp)
        preprocess_val = preprocess_train

     ## Load data for each evaluation dataset and perform evaluation.
    if args.eval_mode == 'coco':
        trans_val = preprocess_val.transforms
        n_px = trans_val[1].size
        trans_val = [T.Resize(n_px, interpolation=Image.BICUBIC)] + trans_val[2:]
        preprocess_val_region = T.Compose(trans_val)
        source_dataset = CsvCOCO(transforms=preprocess_val, 
                                 transforms_region=preprocess_val_region, 
                                 root=root_project)
        source_dataloader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)
        evaluate_coco(model, img2text, retrieval_fuse,text_condition,database, args, source_dataloader)

    elif args.eval_mode == 'cirr':
        source_dataset = CIRR(transforms=preprocess_val, 
                              root=root_project)
        target_dataset = CIRR(transforms=preprocess_val, 
                              root=root_project, 
                              mode='imgs')
        source_dataloader = DataLoader(
            source_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        target_dataloader = DataLoader(
            target_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        evaluate_cirr(model, 
                      img2text,
                      retrieval_fuse, 
                      text_condition,
                      database,
                      args, 
                      source_dataloader, 
                      target_dataloader)

    elif args.eval_mode == 'cirr_test':
        source_dataset = CIRR(transforms=preprocess_val, 
                              root=root_project, test=True)
        target_dataset = CIRR(transforms=preprocess_val, 
                              root=root_project, 
                              mode='imgs', 
                              test=True)
        source_dataloader = DataLoader(
            source_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        target_dataloader = DataLoader(
            target_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        results = evaluate_cirr_test(model, 
                                     img2text, 
                                     retrieval_fuse,
                                     text_condition,
                                     database,
                                     args, 
                                     source_dataloader, 
                                     target_dataloader)
        for key, value in results.items():
            with open('res_cirr/' + key + '.json', 'w') as f:
                json.dump(value, f)
    
    elif args.eval_mode == 'fashion':
        assert args.source_data in ['dress', 'shirt', 'toptee']
        source_dataset = FashionIQ(cloth=args.source_data, 
                                   transforms=preprocess_val, 
                                   root=root_project, 
                                   is_return_target_path=True)
        target_dataset = FashionIQ(cloth=args.source_data, 
                                   transforms=preprocess_val, 
                                   root=root_project, 
                                   mode='imgs')
        source_dataloader = DataLoader(
            source_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        target_dataloader = DataLoader(
            target_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        evaluate_fashion(model, img2text, retrieval_fuse, text_condition, database, args, source_dataloader, target_dataloader)
    elif args.eval_mode == 'imgnet':
        domains = ['cartoon', 'origami', 'toy', 'sculpture']
        prompt = ["a {} of *".format(domain) for domain in domains]
        #prompt = ["a {} of ".format(domain) for domain in domains]
        source_path = os.path.join(root_project, "imgnet", "imgnet_real_query.txt")
        target_path = os.path.join(root_project, "imgnet", "imgnet_targets.txt")
        source_dataset = ImageList(source_path, root=root_project, transforms=preprocess_val, is_labels=True)
        target_dataset = ImageList(target_path, root=root_project, transforms=preprocess_val, is_labels=True)
        eval_func = evaluate_imgnet_retrieval
        source_dataloader = DataLoader(
            source_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        target_dataloader = DataLoader(
            target_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        eval_func(model, img2text, retrieval_fuse, text_condition, database, args, prompt, source_dataloader, target_dataloader)
    


def main():
    args = parse_args()

    # get the name of the experiments
    if args.name is None:
        args.name = (f"lr={args.lr}_"
            "wd={args.wd}_"
            "agg={args.aggregate}_"
            "model={args.model}_"
            "batchsize={args.batch_size}_workers={args.workers}")
        if args.time_suffix:
            args.name += "_date=%Y-%m-%d-%H-%M-%S"
            args.name = strftime(args.name, gmtime())

    if args.copy_codebase:
        import sys, subprocess
        from shutil import copytree, ignore_patterns
        new_code_path = os.path.join(args.logs, args.name, "code")
        if os.path.exists(new_code_path):
            print(
                f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
            )
            return -1
        print(f"Copying codebase to {new_code_path}")
        current_code_path = os.path.realpath(__file__)
        for _ in range(3):
            current_code_path = os.path.dirname(current_code_path)
        copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
        print("Done copying code.")
        os.environ["PYTHONPATH"] = f"{os.environ['PYTHONPATH']}:{os.path.join(new_code_path, 'src')}"
        main_file = os.path.join(new_code_path, "src", "training", "main.py")
        argv = sys.argv
        argv.remove('--copy-codebase')
        argv.extend(['--name', args.name])
        command = [sys.executable] + argv
        print("Executing command:", " ".join(command))
        subprocess.check_call(command)
        return 1

    args.log_path = os.path.join(args.logs, args.name, "out.log")
    if os.path.exists(args.log_path) and args.resume is None:
        print(
            "Error. Experiment already exists. Use --name {} to specify a new experiment."
        )
        return -1

    assert args.precision in ['amp', 'fp16', 'fp32']
    #assert args.model in ['RN50', 'RN101', 'RN50x4', 'ViT-B/32'] or os.path.exists(args.model)

    args.ngpus_per_node = torch.cuda.device_count()

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to

    args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    for dirname in [args.tensorboard_path, args.checkpoint_path]:
        if dirname:
            os.makedirs(dirname, exist_ok=True)
    

    # Set multiprocessing type to spawn.
    # This is important for logging to work with multiprocessing.
    torch.multiprocessing.set_start_method("spawn")

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_queue = setup_primary_logging(args.log_path, args.log_level)
    args.world_size = 1
    #try:
    main_worker(args.gpu, None, log_queue, args)
    #except:
    #    print('evaluation done')


if __name__ == "__main__":
    main()
