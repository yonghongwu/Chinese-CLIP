import os, cv2, sys
sys.path.append('/database/wuyonghuang/WSA')

import time
import json
import base64
import random

import lmdb
import h5py
import pickle
import logging
import warnings

import torch
import numpy as np
import pandas as pd

from math import ceil
from PIL import Image
from pathlib import Path
from io import BytesIO
from dataclasses import dataclass

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from timm.data import create_transform

from cn_clip.clip import _tokenizer
from cn_clip.clip import tokenize

from monai.transforms import (Compose, RandFlipd, RandAffined, RandGaussianNoised, LoadImaged, 
                                EnsureChannelFirstd, ScaleIntensityRanged, Resized)


def _convert_to_rgb(image):
    return image.convert('RGB')


def _preprocess_text(text):
    # adapt the text to Chinese BERT vocab
    text = text.lower().replace("“", "\"").replace("”", "\"")
    return text


class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, split="val", max_txt_length=64, use_augment=False, resolution=224):
        self.lmdb_path = lmdb_path

        # assert LMDB directories exist
        assert os.path.isdir(lmdb_path), "The LMDB directory {} of {} split does not exist!".format(lmdb_path, split)
        lmdb_pairs = os.path.join(lmdb_path, "pairs")
        assert os.path.isdir(lmdb_pairs), "The LMDB directory {} of {} image-text pairs does not exist!".format(lmdb_pairs, split)
        lmdb_imgs = os.path.join(lmdb_path, "imgs")
        assert os.path.isdir(lmdb_imgs), "The LMDB directory {} of {} image base64 strings does not exist!".format(lmdb_imgs, split)

        # open LMDB files
        self.env_pairs = lmdb.open(lmdb_pairs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_pairs = self.env_pairs.begin(buffers=True)
        self.env_imgs = lmdb.open(lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_imgs = self.env_imgs.begin(buffers=True)

        # fetch number of pairs and images
        self.number_samples = int(self.txn_pairs.get(key=b'num_samples').tobytes().decode('utf-8'))
        self.number_images = int(self.txn_imgs.get(key=b'num_images').tobytes().decode('utf-8'))
        logging.info("{} LMDB file contains {} images and {} pairs.".format(split, self.number_images, self.number_samples))

        super(LMDBDataset, self).__init__()

        # the self.dataset_len will be edited to a larger value by calling pad_dataset()
        self.dataset_len = self.number_samples
        self.global_batch_size = 1 # will be modified to the exact global_batch_size after calling pad_dataset()

        self.split = split
        self.max_txt_length = max_txt_length        

        self.use_augment = use_augment
        self.transform = self._build_transform(resolution)

    def _build_transform(self, resolution):
        if self.split == "train" and self.use_augment:
            transform = create_transform(
                             input_size=resolution,
                             scale=(0.9, 1.0),
                             is_training=True,
                             color_jitter=None,
                             auto_augment='original',
                             interpolation='bicubic',
                             mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711),
                         )
            transform = Compose(transform.transforms[:-3] + [_convert_to_rgb] + transform.transforms[-3:])
        else:
            transform = Compose([
                Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        return transform

    def __del__(self):
        if hasattr(self, 'env_pairs'):
            self.env_pairs.close()
        if hasattr(self, 'env_imgs'):
            self.env_imgs.close()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        sample_index = index % self.number_samples

        pair = pickle.loads(self.txn_pairs.get("{}".format(sample_index).encode('utf-8')).tobytes())
        image_id, text_id, raw_text = pair

        image_b64 = self.txn_imgs.get("{}".format(image_id).encode('utf-8')).tobytes()
        image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64))) # already resized
        image = self.transform(image)

        text = tokenize([_preprocess_text(raw_text)], context_length=self.max_txt_length)[0]
        eos_index = text.numpy().tolist().index(_tokenizer.vocab['[SEP]'])
        return image, text, eos_index


def cns5(x):
    if 'glioblastoma' in x:
        return 'glioblastoma'
    elif 'astrocytoma' in x:
        return 'astrocytoma'
    elif 'oligodendroglioma' in x:
        return 'oligodendroglioma'
    else:
        return None


def idh(x):
    if 'wild' in x:
        return 'IDH wild type'
    elif 'mutant' in x:
        return 'IDH mutant'
    else:
        return None


def random_sample_images(sequences, N, target_size=(224, 224)):
    """
    从所有序列中随机获取N张图像，并进行灰度化和简单变换处理。

    参数：
    sequences (dict): 一个字典，其中key是sequence的ID，value是一个list，list的元素是该序列每一帧图像的地址。
    N (int): 要随机获取的图像数量。
    target_size (tuple): 处理后图像的目标大小 (H, W)。

    返回：
    numpy.ndarray: 处理后的图像数组，形状为 (N, H, W)。
    """
    
    # 获取所有图像地址
    all_images = [img for imgs in sequences.values() for img in imgs]
    
    # 随机选择N张图像
    selected_images = random.sample(all_images, N)
    
    processed_images = []

    for img_path in selected_images:
        # 读取图像并转为灰度图像
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Image at path {img_path} could not be read.")
        
        # 进行简单的图像变换
        h, w = img.shape
        # center_h, center_w = h // 2, w // 2
        # target_h, target_w = target_size
        
        # # 裁剪中间区域并缩放
        # cropped_img = img[max(center_h - target_h // 2, 0):min(center_h + target_h // 2, h), 
        #                   max(center_w - target_w // 2, 0):min(center_w + target_w // 2, w)]
        
        resized_img = cv2.resize(img, target_size)
        
        # 添加到处理后的图像列表
        processed_images.append(resized_img)
    
    # 转为numpy数组
    processed_images_array = np.array(processed_images)
    
    return processed_images_array

# # 示例用法
# sequences = {
#     'seq1': ['path/to/image1.jpg', 'path/to/image2.jpg'],
#     'seq2': ['path/to/image3.jpg', 'path/to/image4.jpg']
# }
# N = 2
# target_size = (224, 224)

# processed_images = random_sample_images(sequences, N, target_size)
# print(processed_images.shape)  # 应输出 (2, 224, 224)


class MineDataset(Dataset):
    def __init__(self, mri_lmdb_dir, split="val", max_txt_length=None, use_augment=False, resolution=224, 
                 transform_3d=None, transform_2d=None, us_root_dir=None,
                 train_ratio=0.85, us_N=2, us_target_size=(256, 256), mri_targetsize=(64, 64, 64), seed=42, batch_size=4, use_wsi_coord=False, id_files=None):
        super(MineDataset, self).__init__()
        # self.mri_ids = self.get_mri_ids(mri_lmdb_dir)
        # self.mri_ids = os.listdir('/database/wuyonghuang/WSA/mine_task/MRI_link'); self.mri_ids.sort()
        self.mri_lmdb_dir = mri_lmdb_dir

        if id_files is not None:
            cat_ids = pd.read_excel(id_files, sheet_name='Sheet2')
        else: raise ValueError
        # note: 去掉 WN22-00234，修改 Li Liping 为 Li liping
        cat_ids = cat_ids[~(cat_ids['wsi'] == 'WN22-00234')]

        self.wsi_ids = [i for i in cat_ids['wsi'].unique() if isinstance(i, str)]
        self.cat_ids = cat_ids.drop(columns=['index'])
        self.cat_ids_flatten = list(set([i for i in self.cat_ids.values.flatten() if isinstance(i, str)]))
        self.cat_ids_len = len(self.cat_ids_flatten)

        self.train_len = int(self.cat_ids_len * train_ratio)

        self.us_N = us_N
        self.us_target_size = us_target_size
        self.mri_targetsize = mri_targetsize
        self.batch_size = batch_size
        self.use_wsi_coord = use_wsi_coord
        
        self.cat_ids = self.cat_ids.sample(frac=1, random_state=seed)
        if split == 'train':
            self.cat_ids = self.cat_ids[:self.train_len]
        elif split == 'val':
            self.cat_ids = self.cat_ids[self.train_len:]
        else: raise ValueError

        if transform_3d == 'default':
            transform_3d = Compose([
                                # LoadImaged(keys=['image']),
                                # EnsureChannelFirstd(keys=['image']),  # 将数据转换为 (C, D, H, W)，其中 C=1
                                Resized(keys=['image'], spatial_size=self.mri_targetsize),  # 调整图像大小
                                # ScaleIntensityRanged(keys=['image'], a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True),
                                RandFlipd(keys=['image'], prob=0.5, spatial_axis=[0]),  # 随机翻转
                                # RandAffined(keys=['image'], prob=0.5, rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1)),  # 随机仿射变换
                                # Rand3DElasticd(keys=['image'], prob=0.5, sigma_range=(5, 8), magnitude_range=(100, 200)),  # 随机弹性变形
                                RandGaussianNoised(keys=['image'], prob=0.5, mean=0.0, std=0.1)  # 随机噪声
                            ])
            
        if transform_2d == 'default':
            transform_2d = None

        self.transform_3d = transform_3d
        self.transform_2d = transform_2d
        # self.max_txt_length = max_txt_length
        if max_txt_length is not None:
            warnings.warn("max_txt_length is deprecated and the text will be directly returned without any processing.")

        # 超声
        self.us_root_dir = us_root_dir

        # 病理
        self.wsi_root = '/database/wuyonghuang/WSA/mine_task/WSI_patches/feat_giga_level2/pt_files'
        self.wsi_path = {i: os.path.join(self.wsi_root, i+'.pt') for i in self.wsi_ids}
        # self.wsi_root = '/database/wuyonghuang/jiajihua/kidney/DTFD_tool_patchlevel2/aug_feat/feat_weak/pt_files'
        # self_wsi_path = {i: j for idx_pt, i in enumerate(self.wsi_ids)}
        assert np.mean([os.path.exists(os.path.join(self.wsi_root, i+'.pt')) for i in self.wsi_ids]) == 1

        # 文本
        with open('/database/wuyonghuang/WSA/mine_task/bishe_dataset_prompt.json', 'r') as f:
            self.raw_text = json.loads(f.read())
        
        # id 去索引 类别描述
        df = pd.read_excel('/database/wuyonghuang/WSA/mine_task/US-MRI-WSI-TEXT-huashan20240226.xlsx', header=1)
        df['tumorcls3'] = df['CNS5分类'].apply(cns5)
        df['idh'] = df['CNS5分类'].apply(idh)

        self.df = df[['病人姓名', 'tumorcls3','idh', '1p/19q']].set_index('病人姓名')

    def open_lmdb(self):
        self.env = lmdb.open(self.mri_lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False)        
        self.mris = self.env.begin(buffers=True)
    
    def __getitem__(self, idx):
        wsi_id, us_id, mri_id = self.cat_ids.iloc[idx]
        assert np.mean([not isinstance(i, str) for i in [wsi_id, us_id, mri_id]]) != 1, '不能同时为 nan, 必须有一个模态是存在的'
        unnan_id = [i for i in [wsi_id, us_id, mri_id] if isinstance(i, str)]

        if isinstance(mri_id, str):
            # with self.env.begin(write=False) as txn:
            #     mri_data = pickle.loads(txn.get(mri_id.encode('ascii')))
            if not hasattr(self, 'txn'):
                self.open_lmdb()
            mri_data = pickle.loads(self.mris.get(mri_id.encode('ascii')))
            for modality in mri_data:
                if len(mri_data[modality].shape) == 4:
                    mri_data[modality] = torch.from_numpy(
                        np.stack([Image.fromarray(i).convert('L') for i in mri_data[modality]], axis=0)).float()    # [None, ...]
                elif len(mri_data[modality].shape) == 3:
                    mri_data[modality] = torch.from_numpy(mri_data[modality]).float()   # [None, ...]
                else:
                    raise ValueError
                if self.transform_3d is not None:
                    mri_data[modality] = self.transform_3d({'image': mri_data[modality][None, ...]})['image'][0]
                else:
                    raise ValueError
            
            t1c = True if 'T1c' in mri_data.keys() else False
            flair = True if 'Flair' in mri_data.keys() else False
        else:
            mri_data = {}
            mri_data['T1c'] = torch.zeros(self.mri_targetsize)
            mri_data['Flair'] = torch.zeros(self.mri_targetsize)
            t1c = False
            flair = False
        
        # 超声
        if isinstance(wsi_id, str):
            us_path = os.path.join(self.us_root_dir, us_id)
            items = os.listdir(us_path)
            items.sort()
            sequences = {}
            for item in items:
                sequence_id = item.split('_')[0]
                if sequence_id not in sequences.keys():
                    sequences[sequence_id] = [os.path.join(us_path, item)]
                else:
                    sequences[sequence_id].append(os.path.join(us_path, item))
            us_processed_images = random_sample_images(sequences, self.us_N, self.us_target_size)
        else:
            sequences = {}
            us_processed_images = torch.zeros(self.us_N, *self.us_target_size)

        # 病理
        wsi_coord = None
        if isinstance(wsi_id, str):
            if self.use_wsi_coord:
                h5_file = h5py.File(self.wsi_path[wsi_id].replace('pt_files', 'h5_files').replace('.pt', '.h5'), 'r')
                wsi_feat = torch.from_numpy(h5_file['features'][()])
                wsi_coord = torch.from_numpy(h5_file['coords'][()])
                h5_file.close()
            else:
                wsi_feat = torch.load(self.wsi_path[wsi_id])
        else:
            wsi_feat = torch.zeros(0)  # raise ValueError

        # 文本, 文本模态是必须存在的, 因为要知道它的标签
        lab_tumorcls3, lab_idh3, lab_1p19q = self.df.loc[unnan_id[0], ['tumorcls3', 'idh', '1p/19q']]
        tumorcls_desc = random.sample(self.raw_text[lab_tumorcls3], 1)[0]   # Todo: 增加其他文本内容
        # tumorcls_text = tokenize([_preprocess_text(tumorcls_desc)], context_length=self.max_txt_length)[0]
        # eos_index = tumorcls_text.numpy().tolist().index(_tokenizer.vocab['[SEP]'])
    
        return {"case_id": [wsi_id, us_id, mri_id], "label": [lab_tumorcls3, lab_idh3, lab_1p19q],
                "data_mri": mri_data, "mri_modality": [t1c, flair],
                'data_us_seq': sequences, 'data_us': us_processed_images,
                'data_wsi': wsi_feat, 'data_wsi_coord': wsi_coord, 'text_tumor': tumorcls_desc}  # 不返回文本 embedding: [tumorcls_text, eos_index]
    
    def __len__(self):
        return len(self.cat_ids) // self.batch_size * self.batch_size

    def __del__(self):
        if hasattr(self, 'env_pairs'):
            self.env_pairs.close()
        if hasattr(self, 'env_imgs'):
            self.env_imgs.close()


def custom_collate_fn(batch):
    case_ids = [item['case_id'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # Initialize an empty dictionary for data_mri
    # data_mri = {}
    # for key in batch[0]['data_mri'].keys():
    #     data_mri[key] = torch.stack([item['data_mri'][key] for item in batch if key in item['data_mri']], dim=0)

    data_mri = {}
    possible_keys = ['T1c', 'Flair']
    mri_modality = [item['mri_modality'] for item in batch] # t1c, flair
    
    for key in possible_keys:
        # Check if the key exists in at least one sample
        if any(key in item['data_mri'] for item in batch):
            data_mri[key] = torch.stack([item['data_mri'][key] if key in item['data_mri'] else torch.zeros(224, 224, 224) for item in batch], dim=0) # 128, 128, 128
    
    data_us = torch.from_numpy(np.stack([item['data_us'] for item in batch], axis=0))
    data_wsi = [item['data_wsi'] for item in batch]
    data_wsi_coord = [[item['data_wsi_coord'] if 'data_wsi_coord' in item.keys() else None for item in batch]]
    text_tumor = [item['text_tumor'] for item in batch]
    
    data_us_seq = [item['data_us_seq'] for item in batch]

    # Collate case_ids and labels (assuming they are lists of strings)
    case_ids_collated = [list(x) for x in zip(*case_ids)]
    labels_collated = [list(x) for x in zip(*labels)]
    mri_modality_collated = [list(x) for x in zip(*mri_modality)]
    
    # Collate text_tumor, assuming it is a list of (feature_vector, int)
    # text_tumor_features = torch.stack([item[0] for item in text_tumor], dim=0)    # note: 弃用
    # text_tumor_indices = torch.tensor([item[1] for item in text_tumor])
    text = [item['text_tumor'] for item in batch]
    
    return {
        'case_id': case_ids_collated, 'label': labels_collated,
        'data_mri': data_mri, 'mri_modality': mri_modality_collated,
        'data_us_seq': data_us_seq, 'data_us': data_us,
        'data_wsi': data_wsi, 'data_wsi_coord': data_wsi_coord, 'text_tumor': text, # (text_tumor_features, text_tumor_indices)
    }


def pad_dataset(dataset, global_batch_size):
    # edit dataset.__len__() of the dataset
    dataset.dataset_len = ceil(dataset.dataset_len / global_batch_size) * global_batch_size
    dataset.global_batch_size = global_batch_size


def fetch_resolution(vision_model):
    # fetch the resolution from the vision model config
    vision_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{vision_model.replace('/', '-')}.json"
    with open(vision_model_config_file, 'r') as fv:
        model_info = json.load(fv)
    return model_info["image_resolution"]


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler
    dataset: LMDBDataset
    epoch_id: int


def get_dataset(args, is_train, max_txt_length=64, epoch_id=0, mine=True, use_wsi_coord=False):
    if is_train:
        db_path = args.train_data
    else:
        db_path = args.val_data
    assert db_path is not None

    # pad the dataset splits using the beginning samples in the LMDB files
    # to make the number of samples enough for a full final global batch
    batch_size = args.batch_size if is_train else args.valid_batch_size
    global_batch_size = batch_size * torch.distributed.get_world_size()

    if mine:
        dataset = MineDataset(mri_lmdb_dir='/database/wuyonghuang/WSA/medical_data_lmdb', 
                              us_root_dir='/database/wuyonghuang/WSA/mine_task/US_PNG', 
                              split="train" if is_train else "val", max_txt_length=max_txt_length, 
                              use_augment=False, resolution=fetch_resolution(args.vision_model), 
                              transform_3d='default', transform_2d=None, train_ratio=0.85,
                              us_N=args.us_N, us_target_size=(224, 224), mri_targetsize=(224, 224, 224), seed=42, batch_size=batch_size, use_wsi_coord=use_wsi_coord) # 128, 128, 128
        
        num_samples, dataset.dataset_len = [len(dataset)] * 2
        pad_dataset(dataset, global_batch_size)
    else:
        dataset = LMDBDataset(
            db_path, 
            split="train" if is_train else "val",
            max_txt_length=max_txt_length,
            use_augment=args.use_augment if is_train else False,
            resolution=fetch_resolution(args.vision_model),
        ) 

        # pad the dataset splits using the beginning samples in the LMDB files
        # to make the number of samples enough for a full final global batch
        batch_size = args.batch_size if is_train else args.valid_batch_size
        global_batch_size = batch_size * torch.distributed.get_world_size()
        pad_dataset(dataset, global_batch_size)

        num_samples = dataset.dataset_len
    # Update in 22.12.11: We have changed the **validation** dataset sampler during finetuning
    # from sequential to shuffled (in a determistic order between experiments and epochs). 
    # This is to avoid there being one text matching multiple images (or vice versa) in a local batch
    # which will affect the correctness of computing the validation in-batch accuracy.
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)
    sampler.set_epoch(epoch_id if is_train else 0)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=args.num_workers if is_train else args.valid_num_workers,
        sampler=sampler,
        collate_fn=custom_collate_fn if mine else None
    )
            

    dataloader.num_samples = num_samples
    # assert num_samples % dataset.global_batch_size == 0
    dataloader.num_batches = num_samples // dataset.global_batch_size

    return DataInfo(dataloader, sampler, dataset, epoch_id)


def get_dataset_singleGPU(args, is_train, max_txt_length=64, epoch_id=0, mine=True, use_wsi_coord=False):
    if is_train:
        db_path = args.train_data
    else:
        db_path = args.val_data
    assert db_path is not None

    # 单机版本，直接使用指定的batch_size
    batch_size = args.batch_size if is_train else args.valid_batch_size

    if mine:
        dataset = MineDataset(mri_lmdb_dir='/database/wuyonghuang/WSA/medical_data_lmdb', 
                              us_root_dir='/database/wuyonghuang/WSA/mine_task/US_PNG', 
                              split="train" if is_train else "val", max_txt_length=max_txt_length, 
                              use_augment=False, resolution=fetch_resolution(args.vision_model), 
                              transform_3d='default', transform_2d=None, train_ratio=0.85,
                              us_N=args.us_N, us_target_size=(224, 224), mri_targetsize=(224, 224, 224), 
                              seed=42, batch_size=batch_size, use_wsi_coord=use_wsi_coord,
                              id_files=args.id_files)
        
        num_samples, dataset.dataset_len = [len(dataset)] * 2
        # 不需要pad了，因为不需要考虑分布式的global batch size
    else:
        dataset = LMDBDataset(
            db_path, 
            split="train" if is_train else "val",
            max_txt_length=max_txt_length,
            use_augment=args.use_augment if is_train else False,
            resolution=fetch_resolution(args.vision_model),
        )
        num_samples = dataset.dataset_len

    # 使用普通的RandomSampler替代DistributedSampler
    if is_train:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        # 验证集也使用随机采样，但是固定种子以保持一致性
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        sampler = torch.utils.data.RandomSampler(dataset, generator=generator)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=args.num_workers if is_train else args.valid_num_workers,
        sampler=sampler,
        collate_fn=custom_collate_fn if mine else None
    )
            
    dataloader.num_samples = num_samples
    dataloader.num_batches = num_samples // batch_size
    
    # 创建一个简单的DataInfo对象，保持接口一致
    return DataInfo(dataloader, sampler, dataset, epoch_id)


def get_data(args, epoch_id=0, max_txt_length=64, is_mine=True, use_wsi_coord=False, single_version=False):
    data = {}

    if single_version:
        get_dataset_func = get_dataset_singleGPU
    else:
        get_dataset_func = get_dataset

    if args.train_data:
        data["train"] = get_dataset_func(
            args, 
            is_train=True,  
            max_txt_length=max_txt_length, 
            epoch_id=epoch_id, mine=is_mine, use_wsi_coord=use_wsi_coord)

    if args.val_data:
        data["val"] = get_dataset_func(
            args, 
            is_train=False, 
            max_txt_length=max_txt_length, 
            epoch_id=epoch_id, mine=is_mine, use_wsi_coord=use_wsi_coord)

    return data # isinstanse(data['train'].dataset, MineDataset)


if __name__ == '__main__':
    dataset = MineDataset(mri_lmdb_dir='/database/wuyonghuang/WSA/medical_data_lmdb', 
                          us_root_dir='/database/wuyonghuang/WSA/mine_task/US_PNG', 
                          split="train", max_txt_length=64, use_augment=False, resolution=224, 
                          transform_3d='default', transform_2d=None, train_ratio=0.85)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        pin_memory=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )

    itera = iter(dataloader)

    t1 = time.time()
    a = next(itera)
    t2 = time.time()
    print(t2 - t1)

    for i in range(len(itera)):
        try:
            i_data = next(itera)
        except:
            itera = iter(dataloader)
            i_data = next(itera)

        print(i_data.keys())

        # 以下都是标签、对模态是否缺失的说明
        batch_tumor_label = i_data['label'][0]
        batch_idh_label   = i_data['label'][1]
        batch_1p19q_label = i_data['label'][2]

        batch_wsi         = [isinstance(moi, str) for moi in i_data['case_id'][0]]
        batch_us          = [isinstance(moi, str) for moi in i_data['case_id'][1]]
        batch_t1c         = i_data['mri_modality'][0]
        batch_flair       = i_data['mri_modality'][1]

        # 数据
        data_t1c   = i_data['data_mri']['T1c']
        data_flair = i_data['data_mri']['Flair']
        data_wsi   = i_data['data_wsi']
        data_us    = i_data['data_us']
        text, eos  = i_data['text_tumor']

    """
    'case_id'    : [['wsi_id1', 'wsi_id2'], ['us_id1', 'us_id2'], ['mri_id1', 'mri_id2']], 

    'label'      : [['glioblastoma', 'glioblastoma'], ['IDH wild type', 'IDH wild type'], ['无共缺失', '无共缺失']],

    'mri_modality': [[False, False], [True, False]] # 表示对于t1c和flair两种模态, 每个病人中是否出现了缺失, 第一个list中为 [False, False], 说明第一个病人缺失了这两种模态数据

    'data_mri'   : {'T1c': torch.randn(B, 64, 64, 64), 'Flair': torch.randn(B, 64, 64, 64)}

    'data_us_seq': [{'seq1_id1': [seq1_id1_path1, seq1_id1_path2], {'seq1_id2': [seq1_id2_path1, seq1_id2_path2]}, ...},  {'seq2_id1': [seq2_id1_path1, seq2_id1_path1], {'seq2_id2': [seq2_id2_path1, seq2_id2_path2]}, ...}]    # 暂时不考虑这个

    'data_us'    : torch.randn(B, 2, 256, 256), 

    'data_wsi'   : [torch.Size([N1, 768]), torch.Size([N2, 768])]

    'data_wsi_coord': [ [torch.Size([N1, 2]), torch.Size([N2, 2])] ]

    'text_tumor' : (torch.randn(B, 64), torch.randint(100, size=(B,)) )

    """