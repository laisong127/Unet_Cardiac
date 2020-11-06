import os
import argparse
import logging
import importlib
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
import torchvision
from tensorboardX import SummaryWriter

import tools._init_paths
from libs.configs.config_acdc import cfg

from libs.datasets import AcdcDataset
from libs.datasets import joint_augment as joint_augment
from libs.datasets import augment as standard_augment
from libs.datasets.acdc_dataset_Isensee import AcdcDataset_Isensee
from libs.datasets.batchgenerators.transforms import SpatialTransform
from libs.datasets.collate_batch import BatchCollator

import tools.train_utils.train_utils as train_utils
from tools.Isensee_dataset import create_dataloader_Insensee
from tools.args_config import args
from tools.train_utils.train_utils import load_checkpoint
from utils.init_net import init_weights
from utils.comm import get_rank, synchronize
from libs.losses.Cal_Diceloss import soft_dice, soft_spatial_loss, AMTA_loss, DiceLoss
seed = 12345
random.seed(seed)
np.random.seed(seed)

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# seed = 12345
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
if args.mgpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.mgpus

def create_logger(log_file, dist_rank):
    if dist_rank > 0:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.WARNING)
        return logger
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)

def create_dataloader():
    train_joint_transform = joint_augment.Compose([
        joint_augment.To_PIL_Image(),
        joint_augment.RandomAffine(0, translate=(0.125, 0.125)),
        joint_augment.RandomRotate((-180, 180)),
        joint_augment.FixResize(256)
    ])
    transform = standard_augment.Compose([
        standard_augment.to_Tensor(),
        # standard_augment.normalize([cfg.DATASET.MEAN], [cfg.DATASET.STD])])
        standard_augment.normalize_meanstd()])
    target_transform = standard_augment.Compose([
        standard_augment.to_Tensor()])

    if cfg.DATASET.NAME == 'acdc':
        train_set = AcdcDataset(data_list=cfg.DATASET.TRAIN_LIST,
                                joint_augment=train_joint_transform,
                                augment=transform, target_augment=target_transform)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
    #                         num_replicas=dist.get_world_size(), rank=dist.get_rank())
    train_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True,
                              num_workers=args.workers, shuffle=False,
                              )

    if args.train_with_eval:
        eval_transform = joint_augment.Compose([
            joint_augment.To_PIL_Image(),
            joint_augment.FixResize(256), # divided by 32
            joint_augment.To_Tensor()])
        evalImg_transform = standard_augment.Compose([
            # standard_augment.normalize([cfg.DATASET.MEAN], [cfg.DATASET.STD])],)
            standard_augment.normalize_meanstd()])

        if cfg.DATASET.NAME == 'acdc':
            test_set = AcdcDataset(data_list=cfg.DATASET.TEST_LIST,
                                   joint_augment=eval_transform,
                                   augment=evalImg_transform)

        test_loader = DataLoader(test_set, batch_size=args.batch_size, pin_memory=True,
                                 num_workers=args.workers, shuffle=False,
                                 )
    else:
        test_loader = None
    
    return train_loader, test_loader




def create_optimizer(model):
    if cfg.TRAIN.OPTIMIZER == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                              momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise NotImplementedError
    return optimizer

def create_scheduler(optimizer, last_epoch):
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg.TRAIN.DECAY_STEP_LIST:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg.TRAIN.LR_DECAY
        # return max(cur_decay, cfg.TRAIN.LR_CLIP / cfg.TRAIN.LR)
        return cur_decay

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

    return lr_scheduler

def create_model(cfg):
    network = cfg.TRAIN.NET

    module = 'libs.network.' + network[:network.rfind('.')] 
    model = network[network.rfind('.')+1:]
    
    mod = importlib.import_module(module)
    mod_func = importlib.import_module('libs.network.train_functions')
    net_func = getattr(mod, model)
    print(net_func)

    # net = net_func(num_class=cfg.DATASET.NUM_CLASS,batch=args.batch_size)
    net = net_func()
    train_func = None

    if network == 'unet.U_Net':
        train_func = getattr(mod_func, 'model_fn_decorator')
    if network == 'backbone0.CleanU_Net':
        train_func = getattr(mod_func, 'model_backbone0_decorator')
    if network == 'unet.Isensee_U_Net':
        train_func = getattr(mod_func, 'model_fn_decorator')

    return net, train_func,net_func

def train():
    print(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    # create dataloader & network & optimizer
    model, model_fn_decorator,net_func = create_model(cfg)
    init_weights(model, init_type='kaiming')
    model.cuda()
    root_result_dir = args.output_dir
    os.makedirs(root_result_dir, exist_ok=True)

    log_file = os.path.join(root_result_dir, "log_train.txt")
    logger = create_logger(log_file, get_rank())
    logger.info("**********************Start logging**********************")
    logger.info('TRAINED MODEL:{}'.format(net_func))

    # log to file
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info("CUDA_VISIBLE_DEVICES=%s" % gpu_list)

    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))
    
    logger.info("***********************config infos**********************")
    for key, val in vars(cfg).items():
        logger.info("{:16} {}".format(key, val))
    
    # log tensorboard
    if get_rank() == 0:
        tb_log = SummaryWriter(log_dir=os.path.join(root_result_dir, "tensorboard"))
    else:
        tb_log = None


    train_loader, test_loader = create_dataloader()
    # train_loader, test_loader = create_dataloader_Insensee()

    optimizer = create_optimizer(model)

    # load checkpoint if it is possible
    start_epoch = it = best_res = 0
    last_epoch = -1
    if args.ckpt is not None:
        pure_model = model
        it, start_epoch, best_res = load_checkpoint(pure_model, optimizer, args.ckpt, logger)
        last_epoch = start_epoch + 1
    
    lr_scheduler = create_scheduler(optimizer, last_epoch=last_epoch)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98, last_epoch=-1)

    criterion = None


    # start training
    logger.info('**********************Start training**********************')
    ckpt_dir = os.path.join(root_result_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    trainer = train_utils.Trainer(model,
                                  model_fn=model_fn_decorator(),
                                  criterion=criterion,
                                  optimizer=optimizer,
                                  ckpt_dir=ckpt_dir,
                                  lr_scheduler=lr_scheduler,
                                  model_fn_eval=model_fn_decorator(),
                                  tb_log=tb_log,
                                  logger=logger,
                                  eval_frequency=1,
                                  cfg=cfg)
    
    trainer.train(start_it=it,
                  start_epoch=start_epoch,
                  n_epochs=args.epochs,
                  train_loader=train_loader,
                  test_loader=test_loader,
                  ckpt_save_interval=args.ckpt_save_interval,
                  best_res=best_res)

    logger.info('**********************End training**********************')


# python -m torch.distributed.launch --nproc_per_node 2 --master_port $RANDOM tools/train.py --batch_size 20 --mgpus 2,3 --output_dir logs/... --train_with_eval
if __name__ == "__main__":
    train()


