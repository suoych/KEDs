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
import sys
import math
import logging
import functools
import braceexpand
import random
import pdb
import json

import pandas as pd
import numpy as np
import pyarrow as pa
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000                                                                                              

from dataclasses import dataclass
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
from torchvision.datasets.folder import DatasetFolder
import torchvision.datasets as datasets
import torchvision.transforms as T
from third_party.open_clip.clip import tokenize,_transform
from pathlib import Path
from typing import List, Optional, Union, Dict, Literal
import webdataset as wds
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample
from multiprocessing import Value
from io import BytesIO
import pickle
from model.clip import _transform, load
#np.random.seed(0)
#random.seed(0)
from typing import List, Optional
from llama import Llama, Dialog



cap_dict = {}
with open("cc3m_have_good.pkl","rb") as f_g:
    img_cap_list_good = pickle.load(f_g)

with open("cc3m_have.pkl","rb") as f:
    img_cap_list = pickle.load(f)

for i in range(len(img_cap_list_good)):
    cap_dict[img_cap_list_good[i]['filename']] = img_cap_list_good[i]['text']

for i in range(len(img_cap_list)):
    cap_dict[img_cap_list[i]['filename']] = img_cap_list[i]['text']


with open("cc_subject.json","r") as f:
    subject_dict = json.load(f)
    
with open("cc_other.json","r") as f:
    other_dict = json.load(f)

global_preprocess_train = _transform(224, is_train=True)

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample

def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples



def collate_fn(batch):
    '''
    function which discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1500,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)

def my_map(sample):
    #pdb.set_trace()
    try:
        #print(sample['__key__'])
        sample["caption"] = tokenize(cap_dict[sample['__key__']], truncate=True)[0]
    except:
        sample["caption"] = None
        print("sample wrong!", sample["__key__"])
    if None in sample.values():
        print("caption None found in batch:",sample["__key__"])
        print(sample)
    return sample

def my_decoder(sample):
    #pdb.set_trace()
    try:
        sample["image_byte"] = global_preprocess_train(Image.open(BytesIO(sample["image_byte"])))
    except:
        sample["image_byte"] = None
        print("sample wrong!", sample["__key__"])
    if None in sample.values():
        print("image_byte None found in batch:",sample["__key__"])
        print(sample)
    return sample

def custom_decode(data, *args, handler=None, **kw):
    for sample in data:
        assert isinstance(sample, dict), sample
        try:
            decoded = my_decoder(sample)
            decoded = my_map(sample)
        except Exception as exn:  # skipcq: PYL-W0703
            raise exn
        yield decoded

class CustomDecode(wds.PipelineStage):
    def __init__(self):
        super().__init__()

    def run(self, src):
        return custom_decode(src)

# adding CIRCO dataset
class CIRCODataset(Dataset):
    """
    CIRCO dataset class for PyTorch.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
        - In 'relative' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_captions', 'shared_concept',
             'gt_img_ids', 'query_id'] when split == 'val'
            - ['reference_image', 'reference_name', 'relative_captions', 'shared_concept', 'query_id'] when split == test
    """

    def __init__(self, dataset_path: Union[str, Path], split: Literal['val', 'test'],
                 mode: Literal['relative', 'classic'], preprocess: callable):
        """
        Args:
            dataset_path (Union[str, Path]): path to CIRCO dataset
            split (str): dataset split, should be in ['test', 'val']
            mode (str): dataset mode, should be in ['relative', 'classic']
            preprocess (callable): function which preprocesses the image
        """

        # Set dataset paths and configurations
        dataset_path = Path(dataset_path)
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.data_path = dataset_path

        # Ensure input arguments are valid
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'val', 'test_gt']:
            raise ValueError("split should be in ['test', 'val']")

        # Load COCO images information
        with open(dataset_path / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
            imgs_info = json.load(f)

        self.img_paths = [dataset_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
                          imgs_info["images"]]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        # get CIRCO annotations
        with open(dataset_path / 'annotations' / f'{split}.json', "r") as f:
            self.annotations: List[dict] = json.load(f)

        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 23  # Maximum number of ground truth images

        print(f"CIRCODataset {split} dataset in {mode} mode initialized")

    def get_target_img_ids(self, index) -> Dict[str, int]:
        """
        Returns the id of the target image and ground truth images for a given query

        Args:
            index (int): id of the query

        Returns:
             Dict[str, int]: dictionary containing target image id and a list of ground truth image ids
        """

        return {
            'target_img_id': self.annotations[index]['target_img_id'],
            'gt_img_ids': self.annotations[index]['gt_img_ids']
        }

    def __getitem__(self, index) -> dict:
        """
        Returns a specific item from the dataset based on the index.

        In 'classic' mode, the dataset yields a dictionary with the following keys: [img, img_id]
        In 'relative' mode, the dataset yields dictionaries with the following keys:
            - [reference_img, reference_img_id, target_img, target_img_id, relative_caption, shared_concept, gt_img_ids,
            query_id]
            if split == val
            - [reference_img, reference_img_id, relative_caption, shared_concept, query_id]  if split == test
        """

        if self.mode == 'relative':
            # Get the query id
            query_id = str(self.annotations[index]['id'])

            # Get relative caption and shared concept
            relative_caption = self.annotations[index]['relative_caption']
            shared_concept = self.annotations[index]['shared_concept']

            # Get the reference image
            reference_img_id = str(self.annotations[index]['reference_img_id'])
            reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
            reference_img = self.preprocess(Image.open(reference_img_path))

            if self.split == 'val':
                # Get the target image and ground truth images
                target_img_id = str(self.annotations[index]['target_img_id'])
                gt_img_ids = [str(x) for x in self.annotations[index]['gt_img_ids']]
                target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
                target_img = self.preprocess(Image.open(target_img_path))

                # Pad ground truth image IDs with zeros for collate_fn
                gt_img_ids += [''] * (self.max_num_gts - len(gt_img_ids))

                return {
                    'reference_image': reference_img,
                    'reference_name': reference_img_id,
                    'target_image': target_img,
                    'target_name': target_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'gt_img_ids': gt_img_ids,
                    'query_id': query_id,
                }

            elif self.split == 'test':
                return {
                    'reference_image': reference_img,
                    'reference_name': reference_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'query_id': query_id,
                }

        elif self.mode == 'classic':
            # Get image ID and image path
            img_id = str(self.img_ids[index])
            img_path = self.img_paths[index]

            # Preprocess image and return
            img = self.preprocess(Image.open(img_path))
            return {
                'image': img,
                'image_name': img_id
            }

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.mode == 'relative':
            return len(self.annotations)
        elif self.mode == 'classic':
            return len(self.img_ids)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


## Structure of dataset directory
## CIRR: under ./data/CIRR
## validation images ./dev/
## caption split ./captions/cap.rc2.val.json
## image split ./image_splits/split.rc2.val.json
class CIRR(Dataset):
    def __init__(self, transforms, mode='caps', 
    vis_mode=False, test=False, root='./data'):
        self.mode = mode
        self.transforms = transforms
        self.vis_mode = vis_mode
        ## mode to use test split of CIRR
        self.test = test
        self.root = os.path.join(root, 'CIRR')
        self.root_img = os.path.join(self.root, 'dev')
        if self.test:
            self.root_img = os.path.join(self.root, 'test1')
            if self.mode == 'caps':
                self.json = os.path.join(self.root , 'captions/cap.rc2.test1.json')
            else:
                self.json = os.path.join(self.root, 'image_splits/split.rc2.test1.json')
        else:
            if self.mode == 'caps':
                self.json = os.path.join(self.root, 'captions/cap.rc2.val.json')
            else:
                self.json = os.path.join(self.root, 'image_splits/split.rc2.val.json')
        logging.debug(f'Loading json data from {self.json}.')
        data = json.load(open(self.json, "r"))                                
        self.ref_imgs = []
        self.target_imgs = []
        self.target_caps = []        
        if self.test:
            self.init_test(data)
        elif self.mode == 'caps':            
            self.init_val(data)                        
        else:
            self.target_imgs = [key + ".png" for key in data.keys()]                    
        if self.vis_mode:
            self.target_imgs = list(set(self.target_imgs))
        logging.info("Use {} imgs".format(len(self.target_imgs)))        

    def init_test(self, data):
        self.pairids = []
        if self.mode == 'caps':
            for d in data:
                ref_path = d['reference']+ ".png"
                self.ref_imgs.append(ref_path)
                self.target_caps.append(d['caption']) 
                self.pairids.append(d['pairid'])
                self.target_imgs.append('dummy')
        else:
            self.target_imgs = [key + ".png" for key in data.keys()]

    def init_val(self, data):
        for d in data:
            ref_path = d['reference']+ ".png"
            tar_path = d['target_hard']+ ".png"
            self.ref_imgs.append(ref_path)
            self.target_imgs.append(tar_path)
            self.target_caps.append(d['caption'])            
    
    def return_testdata(self, idx):
        if self.mode == 'caps':
                ref_path = str(self.ref_imgs[idx])
                img_path = os.path.join(self.root_img, ref_path)
                ref_images = self.transforms(Image.open(img_path))
                target_cap = self.target_caps[idx]
                text_with_blank_raw = 'a photo of * , {}'.format(target_cap)    
                caption_only = tokenize(target_cap)[0]
                text_with_blank = tokenize(text_with_blank_raw)[0]                 
                return ref_images, text_with_blank, \
                    caption_only, str(self.ref_imgs[idx]), \
                        self.pairids[idx], text_with_blank_raw
        else:
            tar_path = str(self.target_imgs[idx])
            img_path = Image.open(os.path.join(self.root_img, tar_path))
            target_images = self.transforms(img_path)
            return target_images, tar_path

    def return_valdata(self, idx):
        if self.mode == 'caps' and not self.vis_mode:
            ref_path = str(self.ref_imgs[idx])
            img_path = os.path.join(self.root_img, ref_path)
            ref_images = self.transforms(Image.open(img_path))
            target_cap = self.target_caps[idx]
            text_with_blank = 'a photo of * , {}'.format(target_cap)    
            caption_only = tokenize(target_cap)[0]
            ref_text_tokens = tokenize(text_with_blank)[0]     
            #print("target_cap: ",target_cap, ". text_with_blank:",text_with_blank)            
            return ref_images, ref_text_tokens, caption_only, \
                str(self.ref_imgs[idx]), str(self.target_imgs[idx]), \
                    text_with_blank, target_cap                       
        else:
            tar_path = str(self.target_imgs[idx])
            img_path = os.path.join(self.root_img, tar_path)
            target_images = self.transforms(Image.open(img_path))
            return target_images, img_path

    def __getitem__(self, idx):
        if self.test:                        
            return self.return_testdata(idx)
        else:
            return self.return_valdata(idx)
    
    def __len__(self):
        return len(self.target_imgs)
        
## Fashion-IQ: under ./data/fashion-iq
## validation images ./images
## caption split ./json/cap.{cloth_type}.val.json, cloth_type in [toptee, shirt, dress]
## image split ./image_splits/split.{cloth_type}.val.json, cloth_type in [toptee, shirt, dress]
class FashionIQ(Dataset):
    def __init__(self, cloth, transforms, is_train=False, vis_mode=False, \
        mode='caps', is_return_target_path=False, root='./data'):
        root_iq = os.path.join(root, 'fashion-iq')
        self.root_img = os.path.join(root_iq, 'images')
        self.vis_mode = vis_mode
        self.mode = mode
        self.is_return_target_path = is_return_target_path
        self.transforms = transforms
        if mode == 'imgs':
            self.json_file = os.path.join(root_iq, 'image_splits', \
                'split.{}.val.json'.format(cloth))
        else:
            self.json_file = os.path.join(root_iq, 'json', \
                'cap.{}.val.json'.format(cloth))                
        logging.debug(f'Loading json data from {self.json_file}.')

        self.ref_imgs = []
        self.target_imgs = []
        self.ref_caps = []
        self.target_caps = []        
        if mode == 'imgs':
            self.init_imgs()
            logging.info("Use {} imgs".format(len(self.target_imgs)))
        else:
            self.init_data()     
            logging.info("Use {} imgs".format(len(self.target_imgs)))

    def init_imgs(self):
        data = json.load(open(self.json_file, "r"))
        self.target_imgs = [key + ".png" for key in data]        

    def init_data(self):
        def load_data(data):
            for d in data:
                ref_path = os.path.join(self.root_img, d['candidate']+ ".png") 
                tar_path = os.path.join(self.root_img, d['target']+ ".png")            
                try:
                    Image.open(ref_path)
                    Image.open(tar_path)
                    self.ref_imgs.append(ref_path)
                    self.target_imgs.append(tar_path)
                    self.ref_caps.append((d['captions'][0], d['captions'][1]))
                    #self.target_caps.append(d['captions'][1])
                except:                
                    print('cannot load {}'.format(d['candidate']))
        if isinstance(self.json_file, str):
            data = json.load(open(self.json_file, "r"))        
            load_data(data)            
        elif isinstance(self.json_file, list):
            for filename in self.json_file:
                data = json.load(open(filename, "r")) 
                load_data(data)         

    def __len__(self):
        if self.mode == 'caps':
            return len(self.ref_imgs)
        else:
            return len(self.target_imgs)

    def return_imgs(self, idx):
        tar_path = str(self.target_imgs[idx])
        img_path = os.path.join(self.root_img, tar_path)
        target_images = self.transforms(Image.open(img_path))
        return target_images, os.path.join(self.root_img, tar_path)

    def return_all(self, idx):
        if self.vis_mode:
            tar_path = str(self.target_imgs[idx])
            target_images = self.transforms(Image.open(tar_path))
            return target_images, tar_path            
        ref_images = self.transforms(Image.open(str(self.ref_imgs[idx])))
        target_images = self.transforms(Image.open(str(self.target_imgs[idx])))
        cap1, cap2 = self.ref_caps[idx]
        text_with_blank = 'a photo of * , {} and {}'.format(cap2, cap1)
        token_texts = tokenize(text_with_blank)[0]                
        if self.is_return_target_path:
            return ref_images, target_images, token_texts, token_texts, \
                str(self.target_imgs[idx]), str(self.ref_imgs[idx]), \
                    text_with_blank #cap1
        else:
            return ref_images, target_images, text_with_blank


    def __getitem__(self, idx):
        if self.mode == 'imgs':            
            return self.return_imgs(idx)
        else:            
            return self.return_all(idx)
        
## COCO: under ./data/coco
## validation images ./val2017
## validation masked images ./val2017_masked
## validation csv file ./coco_eval.csv
class CsvCOCO(Dataset):
    def __init__(self, transforms, transforms_region, sep=",",
                return_data_identifier=False, return_filename=False, 
                root='./data'):
        self.transforms = transforms
        self.transforms_region = transforms_region
        self.root = os.path.join(root, 'coco')
        self.root_img = os.path.join(self.root, 'val2017')
        self.csv_file = os.path.join(self.root, 'coco_eval.csv')
        cap_file = os.path.join(self.root, 'annotations/captions_val2017.json')
        logging.debug(f'Loading csv data from {self.csv_file}.')
        df = pd.read_csv(self.csv_file, sep=sep) 
        with open(cap_file,"r") as f:
            caps = json.load(f)
            caps = caps["annotations"]
        self.cap_dict = {}
        for i in caps:
            self.cap_dict[i["image_id"]] = i["caption"]

        self.images = df['id'].tolist()
        ## query_region contains the box of query regions.
        regions = df['query_regions'].tolist()
        self.regions = []
        for region in regions:
            x1, y1, x2, y2 = map(lambda x: int(float(x)), region.split(";"))
            self.regions.append([x1, y1, x2, y2])

        ## query_classes contains the class of query region in the target.
        self.query_classes = df['query_class'].tolist()
        self.classes = []
        ## classes contains the list of classes in the target.
        for list_class in df['classes'].tolist():
            if isinstance(list_class, str):
                list_class = list_class.split(";")
                self.classes.append(list_class)
            else:
                self.classes.append([""])        
        self.return_data_identifier = return_data_identifier
        logging.debug('Done loading data.')
        self.return_filename = return_filename

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_img, str(self.images[idx]))
        
        basename = os.path.basename(img_path).split(".")[0]
        
        image_id = self.images[idx].split(".")[0].split("0")[-1]
        #cap = self.cap_dict[int(image_id)]
        #cap = tokenize(cap)[0]
        image = Image.open(img_path)        
        masked_path = os.path.join(self.root_img.replace('val2017', 'val2017_masked'), \
            str(self.images[idx]))
        image_masked = Image.open(masked_path)
        
        ## extract query region.
        x1, y1, x2, y2 = self.regions[idx]        
        region_image = image_masked.crop((x1, y1, x2, y2)) 

        image = self.transforms(image)
        ## no cropping is applied to query region.
        region_image = self.transforms_region(region_image)
        query_class = self.query_classes[idx]
        other_classes = self.classes[idx]        
        text_with_blank = 'a photo of * and {}'.format(" and ".join(other_classes))
        text_with_queryclass = 'a photo of * and {} and {}'.format(query_class, \
            " and ".join(other_classes))
        raw_text = text_with_queryclass
        text_full = 'a photo of {} and {}'.format(query_class, " and ".join(other_classes))
        #print("text_full:", text_full, "text with blank:", text_with_blank, "text with queryclass:", text_with_queryclass)        
        text_with_blank = tokenize(text_with_blank)[0]
        text_with_queryclass = tokenize(text_with_queryclass)[0]
        text_full = tokenize(text_full)[0]
        return image, region_image, text_full, text_with_blank, \
            text_with_queryclass, str(self.images[idx]), raw_text, basename


class ImageList(Dataset):
    def __init__(self, input_filename, transforms, root=None, 
                 return_filename=False, is_labels=False):
        logging.debug(f'Loading txt data from {input_filename}.')
        with open(input_filename, 'r') as f:
            lines = f.readlines()
        if not is_labels:
            self.images = [line.strip() for line in lines]
        else:
            filenames = [line.strip() for line in lines]
            self.images = [name.split(" ")[0] for name in filenames] 
            self.labels = [int(name.split(" ")[1]) for name in filenames] 
        self.is_labels = is_labels
        self.transforms = transforms
        self.root = root
        logging.debug('Done loading data.')
        self.return_filename = return_filename

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.root is not None:
            img_path = os.path.join(self.root, str(self.images[idx]))
        else:
            img_path = str(self.images[idx])
        images = self.transforms(Image.open(img_path))
        basename = os.path.basename(img_path).split(".")[0]
        if self.return_filename:
            return images, img_path
        elif self.is_labels:
            target = self.labels[idx]
            return images, target, basename    
        else:
            return images


class CustomFolder(Dataset):
    def __init__(self, folder, transform):
        image_lists = os.listdir(folder)
        self.samples = [os.path.join(folder, name) for name in image_lists]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        #global_preprocess_train(Image.open(BytesIO(value)))
        sample = Image.open(str(path))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, path
    
"""
class CustomFolderCC(Dataset):
    def __init__(self, folder, transform):
        image_lists = os.listdir(folder)
        self.samples = [os.path.join(folder, name) for name in image_lists]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        #Args:
        #    index (int): Index

        #Returns:
        #    tuple: (sample, target) where target is class_index of the target class.
        path = self.samples[index]
        basename = os.path.basename(path).split(".")[0]
        cap = cap_dict[basename]
        subject = subject_dict[basename]
        otherpart = other_dict[basename]
        #global_preprocess_train(Image.open(BytesIO(value)))
        sample = Image.open(str(path))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, cap, subject, otherpart, basename
"""
         
class CustomFolderCC(Dataset):
    #This is the version for loading feature.
    def __init__(self, folder, transform):
        self.folder = folder
        self.image_folder = os.path.join(folder,"cc_image_feature_folder_clipl")
        image_lists = os.listdir(self.image_folder)
        self.text_folder = os.path.join(folder,"cc_text_feature_folder_clipl")
        self.image_samples = [os.path.join(self.image_folder, name) for name in image_lists]
        self.transform = transform

    def __len__(self):
        return len(self.image_samples)

    def __getitem__(self, index: int):
        #Args:
        #    index (int): Index
        #Returns:
        #    tuple: (sample, target) where target is class_index of the target class.
        path = self.image_samples[index]
        basename = os.path.basename(path).split(".")[0]

        cap_path = os.path.join(self.text_folder, os.path.basename(path))
        cap = torch.load(str(cap_path), map_location=torch.device('cpu'))
        #cap = cap_dict[basename]
        subject = subject_dict[basename]
        otherpart = other_dict[basename]
        #otherpart = "a photo of * * * " + otherpart.replace("*", " ")
        otherpart = "a photo of * " + otherpart.replace("*", " ")
        #otherpart = otherpart.replace("*", "* * *")
        #global_preprocess_train(Image.open(BytesIO(value)))
        image_sample = torch.load(str(path), map_location=torch.device('cpu'))#Image.open(str(path))
        #if self.transform is not None:
        #    sample = self.transform(sample)
        return image_sample, cap, subject, otherpart, basename

class LoadDataBase(Dataset):
    """
    Class for loading the retrieval databases.
    """
    def __init__(self, folder):
        self.image_folder = os.path.join(folder,"image_feature_database")
        self.text_folder = os.path.join(folder,"text_feature_database")
        self.subject_folder = os.path.join(folder,"subject_feature_database")
        self.other_folder = os.path.join(folder,"other_feature_database")
        self.image_lists = os.listdir(self.image_folder)

    def __len__(self):
        return len(self.image_lists)

    def __getitem__(self, index: int):
        """
        Args: 
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        image_path = os.path.join(self.image_folder, self.image_lists[index])
        text_path = os.path.join(self.text_folder, self.image_lists[index])
        subject_path = os.path.join(self.subject_folder, self.image_lists[index])
        other_path = os.path.join(self.other_folder, self.image_lists[index])
        image_sample = torch.load(str(image_path), map_location=torch.device('cpu'))
        text_sample = torch.load(str(text_path), map_location=torch.device('cpu'))
        subject_sample = torch.load(str(subject_path), map_location=torch.device('cpu'))
        other_sample = torch.load(str(other_path), map_location=torch.device('cpu'))
        subject_sample = subject_sample.detach().squeeze()
        other_sample = other_sample.detach().squeeze()
        other_sample.requires_grad = False
        subject_sample.requires_grad = False
        other_sample.requires_grad = False
        return image_sample,text_sample, self.image_lists[index], subject_sample, other_sample

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t",
                 return_data_identifier=False, return_filename=False):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.return_data_identifier = return_data_identifier
        logging.debug('Done loading data of {} samples'.format(len(self.images)))
        self.return_filename = return_filename

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        if self.return_filename:
            return images, str(self.images[idx])
        texts = tokenize([str(self.captions[idx])])[0]

        if self.return_data_identifier:
            return images, texts, 0
        return images, texts

"""
@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler
"""

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

def preprocess_txt(text):
    return tokenize([str(text)])[0]

"""
def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    sizes = json.load(open(sizes_filename, 'r'))
    total_size = sum(
        [int(sizes[os.path.basename(shard)]) for shard in shards_list])
    num_shards = len(shards_list)
    return total_size, num_shards
"""

def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path  = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )
    return DataInfo(dataloader, sampler=sampler)

def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches

def get_csv_dataset(args, preprocess_fn, is_train, input_filename=None):
    if input_filename is None:
        input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator)
        
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset,seed=args.seed) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler=sampler)


#
def get_imgnet_r(args, preprocess_fn, is_train, input_filename=None):
    if input_filename is None:
        input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    path_data = os.path.join(args.root_data, 'imgnet/imagenet-r')
    dataset = CustomFolder(path_data, transform=preprocess_fn)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader, sampler=sampler)


def get_directory_dataset(args, preprocess_fn, is_train, input_filename=None):
    if input_filename is None:
        input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CustomFolderCC(
        input_filename,
         transform=preprocess_fn)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset,seed=args.seed) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        #pin_memory=True,
        pin_memory=False,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler=sampler)


def my_decoder_kv(key, value):
    if not key.endswith("image_byte"):
        return None
    assert isinstance(value, bytes)
    #preprocess_train = _transform(224, is_train=True)
    return global_preprocess_train(Image.open(BytesIO(value)))#preprocess_train(Image.open(BytesIO(value)))


 
def get_wds_dataset(args, preprocess_img, is_train, epoch=0):
    shared_epoch = SharedEpoch(epoch=epoch)
    url = args.train_data #args.train_data should have format like "webdataset/cc3m-{00000..00010}.tar"

    if args.train_num_samples is None:
        args.train_num_samples = 2803766 # if no training samples set then we manually set it to 3M
    
    """
    pipeline = [wds.SimpleShardList(url)]
    if not args.dataset_resampled:
        pipeline.extend([
            #detshuffle2(
            #    bufsize=_SHARD_SHUFFLE_SIZE,
            #    initial=_SHARD_SHUFFLE_INITIAL,
            #    seed=args.seed,
            #    epoch=shared_epoch,
            #),
            wds.split_by_node,
            wds.split_by_worker,
        ])
    pipeline.extend([
        # at this point, we have an iterator over the shards assigned to each worker at each node
        tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
        #detshuffle2(
        #        bufsize=_SHARD_SHUFFLE_SIZE,
        #        initial=_SHARD_SHUFFLE_INITIAL,
        #        seed=args.seed,
        #        epoch=shared_epoch,
        #    ),
        wds.shuffle(
            bufsize=_SAMPLE_SHUFFLE_SIZE,
            initial=_SAMPLE_SHUFFLE_INITIAL,
        ),
        CustomDecode(),
        wds.to_tuple("image_byte", "caption"),
        wds.batched(args.batch_size, partial=False)
    ])
    dataset = wds.DataPipeline(*pipeline)
    """
    #dataset = dataset.decode(my_decoder).map(my_map).batched(args.batch_size, partial=False)

    if not args.dataset_resampled:
        num_shards = len(wds.shardlists.expand_urls(url))
        assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'

    global_batch_size = args.batch_size * args.world_size
    num_batches = math.floor(args.train_num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = math.floor(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size

    """
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    """

    # whether to use epoch shuffle?
    dataset = wds.WebDataset(url,shardshuffle=1000, nodesplitter=wds.split_by_node).shuffle(5000).decode(my_decoder_kv).map(my_map).to_tuple("image_byte", "caption").batched(args.batch_size)# don't use too large shuffle buffer since it costs alot memory
    #dataset = wds.WebDataset(url,shardshuffle=True, resampled=True, nodesplitter=wds.split_by_node).shuffle(100000).decode(my_decoder).map(my_map).with_epoch(30000)#.with_length(args.train_num_samples)
    #dataset = dataset

    dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    
    """
    pdb.set_trace()
    j = 0
    dict_key = set()
    for i in dataset:
        j+=1
        print(j)
        dict_key.add(i["__key__"])
        if None in i.values():
            pdb.set_trace()
            print(i["__key__"])
    print("total num:", j)
    print("data num:", len(dict_key))
    pdb.set_trace()
    """
    #dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size) # currently the iterateble dataset does not support multiple worker unless rewrite the class
    
    """
    pdb.set_trace()
    for i, batch in enumerate(dataloader):
        pdb.set_trace()
    pdb.set_trace()
    """

    dataloader.num_batches = num_batches
    dataloader.num_samples = args.train_num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == 'imgnet_r':
        return get_imgnet_r
    elif dataset_type == 'fashion-iq':
        return get_fashion_iq
    elif dataset_type == 'cirr':
        return get_cirr
    elif dataset_type == 'directory':
        return get_directory_dataset
    elif dataset_type == "csv":
        return get_csv_dataset    
    elif dataset_type == "webdataset":
        return get_wds_dataset     
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}
    dataset_type_val = getattr(args, 'dataset_type_val', args.dataset_type)
    if args.train_data:
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
                args, preprocess_train, is_train=True)
    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, dataset_type_val)(
            args, preprocess_val, is_train=False)
    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")
    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")
    return data
