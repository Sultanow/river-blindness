###############################################
import contextlib
import glob
import inspect
import logging
import math
import os
import platform
import random
import re
import shutil
import signal
import threading
import time
import urllib
from datetime import datetime
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from typing import Optional
from zipfile import ZipFile
import pkg_resources as pkg
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
###############################################
###############################################



import logging
import mmdet
import mmcv
import numpy as np
import pandas as pd
import os
import torch
import urllib.request

from mmcv import Config
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
from mmcv.utils.logging import get_logger
from mmdet.apis import train_detector, set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

from Dataset import OnchoDataset
from detection_util import create_predictions
from gdsc_util import load_sections_df


###############################################
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
###############################################
###############################################


###############################################
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]
###############################################
###############################################


def check_versions():
    logger = logging.getLogger(__name__)
    logger.info("Checking torch and mmdet config")
    logger.info("Torch version")
    logger.info(torch.__version__)
    logger.info("Torch sees CUDA?")
    logger.info(torch.cuda.is_available())
    logger.info("MMDet version")
    logger.info(mmdet.__version__)
    logger.info("Compiled CUDA version")
    logger.info(get_compiling_cuda_version())
    logger.info("CUDA Compiler version")
    logger.info(get_compiler_version())


def load_config(data_folder):
    base_file = "/mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py"
    cfg = Config.fromfile(base_file)
    
    # Modify dataset type and path
    cfg.dataset_type = 'OnchoDataset'           # This is a custom data loader script we created for the GDSC you can view it in src/Dataset.py
    cfg.data_root = data_folder

    cfg.data.train.type = 'OnchoDataset'
    cfg.data.train.data_root = data_folder      # path to the folder data
    cfg.data.train.img_prefix = 'train/'         # path from data_root to the images folder
    cfg.data.train.ann_file = 'gdsc_train.csv'  # the file containing the train data labels

    cfg.data.test.type = 'OnchoDataset'
    cfg.data.test.data_root = data_folder
    cfg.data.test.img_prefix = 'train/'
    cfg.data.test.ann_file = 'gdsc_train.csv'

    cfg.data.val.type = 'OnchoDataset'          # We will not use a separate validation data set in this tutorial, but we need to specify the values to overwrite the COCO defaults.
    cfg.data.val.data_root = data_folder
    cfg.data.val.img_prefix = 'train/'
    cfg.data.val.ann_file = 'gdsc_train.csv'

    # Download weights
    weights_url = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
    weights_path = weights_url.split('/')[-1]
    weights_path, headers = urllib.request.urlretrieve(weights_url, filename=weights_path)
    cfg.load_from = weights_path

    # Set up working dir to save files and logs.
    # model_dir = os.environ.get("SM_MODEL_DIR")
    model_dir = "/opt/ml/model/"
    
    if model_dir is None:        
        cfg.work_dir = f'{data_folder}/tutorial_exps/'
    else:
        cfg.work_dir = model_dir
    
    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    cfg.data.samples_per_gpu = 1 # These numbers will change depending on the size of your model and GPU.
    cfg.data.workers_per_gpu = 1 # These values are what we have found to be best for this model and GPU

    # modify number of classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = 1  # a worm section is the only object we are detecting
    
    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU and mulitply by the number of GPU workers.
    cfg.optimizer.lr = 0.02 / 8 * cfg.data.workers_per_gpu
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 50
    
    # Change the evaluation metric since we use customized dataset.
    cfg.evaluation.metric = 'mAP'
    # We can set the evaluation interval to reduce the evaluation times
    cfg.evaluation.interval = 1
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 1
    # How long do we want to train
    cfg.runner.max_epochs = 5
    
    # set device to GPU
    cfg.device = "cuda"
    
    # Tutorial 5: Increase max number of sections, i.e., boxes
    cfg.model.test_cfg.rcnn.max_per_img = 400
    
    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),      
        dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),       
        dict(
            type='Resize',
            img_scale=(4096, 4096), # Changed
            multiscale_mode='value',
            keep_ratio=True),
        dict(
            type='RandomCrop',
            crop_size=(0.3, 0.3),
            crop_type='relative', # Switched from relative_range to relative (fixed crops only)
            allow_negative_crop=True),
        dict(
            type='RandomFlip',
            flip_ratio=[0.5, 0.5], 
            direction=['horizontal', 'vertical']), 
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ]
    cfg.data.train.pipeline = cfg.train_pipeline

    # Modify data pipeline (test-time augmentation)
    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(4096, 4096), # Changed
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])
    ]
    cfg.data.test.pipeline = cfg.test_pipeline

    # Modify data pipeline (val-time augmentation)
    cfg.data.val.pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(4096, 4096), # Changed
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])
    ]

    cfg = Config.fromstring(cfg.pretty_text, '.py') # An ugly workaround to enable calling dict key by a dot (reconstruct object)
    
    return cfg, base_file

###############################################
def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']
###############################################
###############################################



###############################################
def check_requirements(requirements=ROOT / 'code/requirements.txt', exclude=(), install=True, cmds=()):
    print(f"ROOT: {ROOT}")
    # Check installed dependencies meet requirements (pass *.txt file or list of packages)
    prefix = colorstr('red', 'bold', 'requirements:')
    # check_python()  # check python version
    if isinstance(requirements, (str, Path)):  # requirements.txt file
        file = Path(requirements)
        assert file.exists(), f"{prefix} {file.resolve()} not found, check failed."
        with file.open() as f:
            requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(f) if x.name not in exclude]
    else:  # list or tuple of packages
        requirements = [x for x in requirements if x not in exclude]

    n = 0  # number of packages updates
    for i, r in enumerate(requirements):
        try:
            pkg.require(r)
        except Exception:  # DistributionNotFound or VersionConflict if requirements not met
            s = f"{prefix} {r} not found and is required by YOLOv5"
            if install and AUTOINSTALL:  # check environment variable
                print(f"{s}, attempting auto-update...")
                try:
                    assert check_online(), f"'pip install {r}' skipped (offline)"
                    print(check_output(f'pip install "{r}" {cmds[i] if cmds else ""}', shell=True).decode())
                    n += 1
                except Exception as e:
                    print(f'{prefix} {e}')
            else:
                print(f'{s}. Please install and rerun your command.')

    if n:  # if packages updated
        source = file.resolve() if 'file' in locals() else requirements
        s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        print(emojis(s))
###############################################
###############################################

        
if __name__ == "__main__":
    logger = get_logger(__name__)
    
    check_versions()    
    # data_folder = os.environ.get("SM_CHANNEL_TRAIN")
    ###############################################
    check_requirements(exclude=['thop'])
    ###############################################
    ###############################################
    
    
    data_folder="/opt/ml/input/data/images/"
    #output_folder = os.environ.get("SM_MODEL_DIR")
    output_folder = "/opt/ml/model/"
    
    logger.info("Loading config")
    cfg, _ = load_config(data_folder)

    logging.info("Building dataset")
    datasets = [build_dataset(cfg.data.train)]

    logging.info("Building model")
    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("train_cfg"))
    model.CLASSES = datasets[0].CLASSES  # Add an attribute for visualization convenience

    logging.info("Training model")
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)
    
#     epoch = 'epoch_8'  # Select one of the model checkpoints to load in    
#     checkpoint = f'{output_folder}/{epoch}.pth' 
    
#     logging.info("Creating Train Predictions")
#     section_df = load_sections_df(f'{data_folder}/gdsc_train.csv')
#     file_names = nodule_filter_train = np.unique(section_df.file_name.values).tolist()
#     prediction_df = create_predictions(file_names, cfg, checkpoint, device='cuda')
#     prediction_df.to_csv(f'{output_folder}/results_tutorial5_train_{epoch}.csv', sep=';')
    
#     logging.info("Creating Test Predictions")
#     file_names = pd.read_csv(f'{data_folder}/test_files.csv', sep=';', header=None)[0].values
#     prediction_df = create_predictions(file_names, cfg, checkpoint, device='cuda')
#     prediction_df.to_csv(f'{output_folder}/results_tutorial5_test_{epoch}.csv', sep=';')
