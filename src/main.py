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
import wandb
import torch
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from third_party.open_clip.scheduler import cosine_lr
from model.clip import _transform, load
from model.model import convert_weights, CLIP, IM2TEXT,CrossFormer,T2I
from trainer import train,save_feature
from data import get_data,LoadDataBase
from params import parse_args
from logger import setup_primary_logging, setup_worker_logging
from utils import is_master, convert_models_to_fp32
import torchvision.transforms as T
import numpy as np
import random
import pdb
from torch.utils.data import DataLoader
import faiss
import copy

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

def main_worker(gpu, ngpus_per_node, log_queue, args,database=None):

    args.gpu = gpu
    args.rank = gpu
    setup_worker_logging(args.rank, log_queue, args.log_level)
    
    cudnn.benchmark = False        # if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    #cudnn.benchmark = True
    #cudnn.deterministic = False
    seed_everything(args.seed)  
    #torch.cuda.manual_seed_all(args.seed) 
    torch.use_deterministic_algorithms(True)

    image_bases,text_bases = database[0], database[1]
    ngpus = faiss.get_num_gpus()
    image_cpu_index = faiss.IndexFlatL2(768)
    res = faiss.StandardGpuResources()
    image_gpu_index = faiss.index_cpu_to_gpu(res, gpu, image_cpu_index)
    #image_gpu_index = faiss.index_cpu_to_all_gpus(image_cpu_index)
    image_gpu_index.add(image_bases.numpy())  # add vectors to the image_index
    #image_cpu_index.add(image_bases.numpy())
    text_cpu_index = faiss.IndexFlatL2(768)
    #text_gpu_index = faiss.index_cpu_to_all_gpus(text_cpu_index)
    text_gpu_index = faiss.index_cpu_to_gpu(res, gpu, text_cpu_index)
    text_gpu_index.add(text_bases.numpy())  # add vectors to the image_index
    #text_cpu_index.add(text_bases.numpy())
    database.append(image_gpu_index)
    database.append(text_gpu_index)
    #database.append(image_cpu_index)
    #database.append(text_cpu_index)
    print("Adding indices done!")

    # Log and save params.
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

    # Do not use skip_reset unless you want to use on of the CLIP model
    if args.openai_pretrained:
        model, preprocess_train, preprocess_val = load(
            args.model,
            jit=False)
    else:
        model_config_file = Path(__file__).parent / f"model_configs/{args.model.replace('/', '-')}.json"
        print('Loading model from', model_config_file)
        assert os.path.exists(model_config_file)
        with open(model_config_file, 'r') as f:
            model_info = json.load(f)
        if args.use_prefix:
            model_info['vocab_size'] += 1
            model_info['use_prefix'] = True
        model = CLIP(**model_info)
        convert_weights(model)        
        preprocess_train = _transform(model.visual.input_resolution, is_train=True)
        preprocess_val = _transform(model.visual.input_resolution, is_train=False)
    try:
        img2text = IM2TEXT(embed_dim=model.embed_dim, 
                           middle_dim=args.middle_dim, 
                           output_dim=model.token_embedding.weight.shape[1], 
                           n_layer=args.n_layer)
        retrieval_fuse = CrossFormer(q_dim=model.token_embedding.weight.shape[1],k_dim=model.token_embedding.weight.shape[1],v_dim=model.token_embedding.weight.shape[1],num_layers = 3, dropout = 0.1)
        text_condition = CrossFormer(q_dim=model.token_embedding.weight.shape[1],k_dim=model.token_embedding.weight.shape[1],v_dim=model.token_embedding.weight.shape[1],num_layers = 3, dropout = 0.1)
        #img2text = CrossFormer(q_dim=model.visual.proj.shape[0],dim=model.token_embedding.weight.shape[1])
        #text_condition = T2I(embed_dim=model.token_embedding.weight.shape[1], 
        #                   middle_dim=args.middle_dim, 
        #                   output_dim=model.visual.proj.shape[0], 
        #                   n_layer=args.n_layer)
        
    except:
        print("Error!!!!")
        img2text = IM2TEXT(embed_dim=1024, output_dim=1024,
        is_normalize=args.normalize_output, is_mlp=args.use_mlp, n_layer=args.n_layer)

    logging.info(f"{args.gpu} reach the line!!!!")

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
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
            torch.cuda.synchronize()
            img2text = torch.nn.parallel.DistributedDataParallel(img2text, device_ids=[args.gpu], find_unused_parameters=False)
            torch.cuda.synchronize()
            retrieval_fuse = torch.nn.parallel.DistributedDataParallel(retrieval_fuse, device_ids=[args.gpu], find_unused_parameters=False)
            torch.cuda.synchronize()
            text_condition = torch.nn.parallel.DistributedDataParallel(text_condition, device_ids=[args.gpu], find_unused_parameters=False)
            torch.cuda.synchronize()
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


    data = get_data(args, (preprocess_train, preprocess_val))
    exclude = lambda n : "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n : not exclude(n)
    named_parameters = list(img2text.named_parameters())
    named_parameters = named_parameters + list(retrieval_fuse.named_parameters())
    named_parameters = named_parameters + list(text_condition.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

    if args.train_data is None:
        optimizer = None
        scheduler = None
    else:
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        total_steps = data["train"].dataloader.num_batches * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    scaler = GradScaler() if args.precision == "amp" else None #or args.precision == "fp16" else None

    
    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume == 'auto':
        checkpoint_list = os.listdir(args.checkpoint_path)
        checkpoint_list = [ckpt for ckpt in checkpoint_list if ckpt.startswith('epoch')]
        if checkpoint_list:
            latest_epoch = max([int(ckpt.split('_')[1].split('.')[0]) for ckpt in checkpoint_list])
            args.resume = os.path.join(args.checkpoint_path, f'epoch_{latest_epoch}.pt')
        else:
            args.resume = None

    if args.resume is not None:
        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint["epoch"]
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
            model.load_state_dict(sd)
            img2text.load_state_dict(sd_img2text)
            retrieval_fuse.load_state_dict(sd_retrieval_fuse)
            text_condition.load_state_dict(sd_text_condition)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(
                f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})"
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    #cudnn.benchmark = True
    #cudnn.deterministic = False
    
    # determine if this worker should save logs and checkpoints.
    # only do so if it is the 0th worker.
    args.save_logs = (args.logs is not None and args.logs != '' and args.logs.lower() != 'none') and (
        (not args.distributed) or args.gpu == 0
    )
    writer = None
    if args.save_logs and args.tensorboard:
        writer = SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project="zcomp",
            notes=args.wandb_notes,
            tags=[],
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    for epoch in range(start_epoch, args.epochs):
        if args.gpu == 0:
            logging.info(f'Start epoch {epoch}')
        if args.pre_save_feature:
            save_feature(model, img2text, data, epoch, optimizer, scaler, scheduler, args, writer)
            break
        else:
            train(model, img2text, retrieval_fuse,text_condition, data, epoch, optimizer, scaler, scheduler, args, writer,database=database)
        steps = data["train"].dataloader.num_batches * (epoch + 1)        
        # Saving checkpoints.
        if args.save_logs and (args.gpu == 0 or (not args.distributed)):
            if (epoch + 1) == args.epochs or (
                args.save_frequency > 0 and ((epoch + 1) % args.save_frequency) == 0
            ):
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "name": args.name,
                        "state_dict": model.state_dict(),
                        "state_dict_img2text": img2text.state_dict(),
                        "state_dict_retrieval_fuse": retrieval_fuse.state_dict(),
                        "state_dict_text_condition": text_condition.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}.pt"),
                )
            if args.save_most_recent:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "name": args.name,
                        "state_dict": model.state_dict(),
                        "state_dict_img2text": img2text.state_dict(),
                        "state_dict_retrieval_fuse": retrieval_fuse.state_dict(),
                        "state_dict_text_condition": text_condition.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(args.checkpoint_path, "epoch_latest.pt"),
                )

    if args.wandb and (args.gpu == 0 or (not args.distributed)):
        wandb.finish()


def main():
    args = parse_args()

    cudnn.benchmark = False        # if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    #cudnn.benchmark = True
    #cudnn.deterministic = False
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)    
    torch.cuda.manual_seed_all(args.seed) 
    random.seed(args.seed)
    
    torch.use_deterministic_algorithms(True)
    # get the name of the experiments
    if args.name is None:
        args.name = (f"lr={args.lr}_"
            "wd={args.wd}_"
            "agg={args.aggregate}_"
            "model={args.model}_"
            "batchsize={args.batch_size}_workers={args.workers}")
        import pdb
        pdb.set_trace
        if args.time_suffix:
            args.name += "_date=%Y-%m-%d-%H-%M-%S"
            args.name = strftime(args.name, gmtime())


    #pdb.set_trace()
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
    
    #pdb.set_trace()
    """
    # We load database here.
    Base_dataset = LoadDataBase("/home/yucheng/clip_cc_database")#dino_cc_database") 
    print("Loading databases!")
    dataloader = DataLoader(Base_dataset, batch_size=512, shuffle=False, num_workers=10)
    database = {}
    image_bases = []
    text_bases = []
    base_names = []
    for batch in dataloader:
        batch_cp = copy.deepcopy(batch)  
        del batch 
        image_base, text_base, basename = batch_cp[0], batch_cp[1], batch_cp[2]
        image_bases.append(image_base)
        text_bases.append(text_base)
        base_names.append(basename)
    image_bases = torch.cat(image_bases,dim=0)
    text_bases = torch.cat(text_bases,dim=0)
    image_bases = image_bases.cpu()
    text_bases = text_bases.cpu()
    image_bases = image_bases / image_bases.norm(dim=1, keepdim=True)
    text_bases = text_bases / text_bases.norm(dim=1, keepdim=True)
    database = [image_bases,text_bases, base_names]
    torch.cuda.empty_cache()
    """
    print("Loading databases!")
    image_bases = torch.load("/home/yucheng/cc_image_databases.pt",map_location="cpu")
    text_bases = torch.load("/home/yucheng/cc_text_databases.pt",map_location="cpu")
    subject_bases = torch.load("/home/yucheng/cc_subject_databases.pt",map_location="cpu")
    other_bases = torch.load("/home/yucheng/cc_other_databases.pt",map_location="cpu")
    basenames = []
    with open("/home/yucheng/database_names.txt", "r") as f:
        for line in f:
            basenames.append(line.strip())
    database = [image_bases,text_bases,basenames,subject_bases,other_bases]
    print("Loading databases done!")
    #pdb.set_trace()

    # Distributed training = training on more than one GPU.
    # Also easily possible to extend to multiple nodes & multiple GPUs.
    args.distributed = (args.gpu is None) and torch.cuda.is_available() and (not args.dp)
    if args.distributed:
        ngpus_per_node = torch.cuda.device_count()
        args.world_size = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, log_queue, args, database))
    else:
        if args.dp:
            args.gpu = args.multigpu[0]
            args.world_size = len(args.multigpu)
        else:
            args.world_size = 1
        main_worker(args.gpu, None, log_queue, args,database=database)


if __name__ == "__main__":
    main()
