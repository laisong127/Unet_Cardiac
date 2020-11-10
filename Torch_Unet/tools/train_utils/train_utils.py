import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import json

from tools.Isensee_dataset import create_dataloader_Insensee
from utils.comm import get_rank



def save_checkpoint(state, filename='checkpoint'):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)

def save_best_checkpoint(state, filename='checkpoint', is_best=False):
    filename = '{}.pth'.format(filename)
    if is_best:
        torch.save(state, os.path.join(os.path.dirname(filename), "model_best.pth"))

def checkpoint_state(model=None, optimizer=None, epoch=None, it=None, performance=0.):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'performance': performance}

def load_checkpoint(model=None, optimizer=None, filename="checkpoint", logger=None):
    if os.path.isfile(filename):
        if logger is not None:
            logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location="cpu")
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else -1
        it = checkpoint.get('it', 0.0)
        performance = checkpoint.get('performance', 0.)
        if model is not None and checkpoint['model_state'] is not None:
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if logger is not None:
            logger.info("==> Done")
    else:
        raise FileNotFoundError
    
    return it, epoch, performance

class Trainer():
    def __init__(self, model, model_fn, criterion, optimizer, ckpt_dir, lr_scheduler, model_fn_eval,
                 tb_log, logger, eval_frequency=1, cfg=None):
        self.model, self.model_fn, self.optimizer, self.model_fn_eval = model, model_fn, optimizer, model_fn_eval

        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.ckpt_dir = ckpt_dir
        self.tb_log = tb_log
        self.logger = logger
        self.eval_frequency = eval_frequency
        self.cfg = cfg


    def _train_it(self, batch, epoch=0):
        self.model.train()

        self.optimizer.zero_grad()
        loss, tb_dict, disp_dict = self.model_fn(self.model, batch, self.criterion, perfermance=False)

        loss.backward(retain_graph=True)
        # 两者都是backward()的参数   默认 retain_graph=False
        # 也就是反向传播之后这个计算图的内存会被释放，这样就没办法进行第二次反向传播了
        self.optimizer.step()
        return loss.item(), tb_dict, disp_dict
    
    def eval_epoch(self, d_loader):
        self.model.eval()

        eval_dict = {}
        # total_loss = 0

        # eval one epoch
        if get_rank() == 0: print("evaluating...")
        for i, data in enumerate(d_loader, 0):
            self.optimizer.zero_grad()

            _, tb_dict, disp_dict = self.model_fn_eval(self.model, data, self.criterion, perfermance=True)

            # total_loss += loss.item() # removed total loss

            for k, v in tb_dict.items():
                eval_dict[k] = eval_dict.get(k, 0) + v #key -- 字典中要查找的键,default -- 如果指定键的值不存在时，返回该默认值。
            if get_rank() == 0: print("\r{}/{} {:.0%}\r".format((i+1), len(d_loader), (i+1)/len(d_loader)), end='')
        if get_rank() == 0: print()

        for k, v in tb_dict.items():
            eval_dict[k] = eval_dict.get(k, 0) / (i + 1)
        
        return _, eval_dict, disp_dict # remove total_loss / (i+1)

    def train(self, start_it, start_epoch, n_epochs, train_loader, test_loader=None,
              ckpt_save_interval=5, best_res=0):
        eval_frequency = self.eval_frequency if self.eval_frequency else 1
        #===============================================================================================================
        # train_loader_change1, _ = create_dataloader_Insensee(alpha=(0., 250.),scale_range=(0.75, 1.25))
        # train_loader_change2, _ = create_dataloader_Insensee(alpha=(0., 150.),scale_range=(0.8, 1.2))
        #===============================================================================================================

        it = start_it
        train_time = 0
        for epoch in range(start_epoch, n_epochs):
        #===============================================================================================================
            # if epoch==100:
            #     train_loader = train_loader_change1
            #     print('train_loader first changing...')
            #     self.logger.info('train_loader first changing...')
            # if epoch==125:
            #     train_loader = train_loader_change2
            #     print('train_loader second changing...')
            #     self.logger.info('train_loader second changing...')
        #===============================================================================================================
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch)
            epoch_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            print('epoch:{}  lr:{}'.format(epoch, epoch_lr))
            self.logger.info('epoch:{}  lr:{}'.format(epoch, epoch_lr))
            train_start = time.time()
            for cur_it, batch in enumerate(train_loader):
                #=======================================================
                if cur_it<0.1*len(train_loader): # use 0.1 training data
                #========================================================
                    cur_lr = self.lr_scheduler.get_lr()[0]

                    loss, tb_dict, disp_dict = self._train_it(batch, epoch)
                    it += 1

                    # print infos
                    if get_rank() == 0:
                        print("Epoch/train:{}({:.0%})/{}({:.0%})".format(epoch, epoch/n_epochs,
                                        cur_it, cur_it/len(train_loader)), end="")
                        for k, v in disp_dict.items():
                            print(", ", k+": {:.6}".format(v), end="")
                        print("")

                    # tensorboard logs
                    if self.tb_log is not None:
                        # self.tb_log.add_scalar("train_loss", loss, it)
                        self.tb_log.add_scalar("train_loss", loss, it)
                        self.tb_log.add_scalar("learning_rate", cur_lr, it)
                        for key, val in tb_dict.items():
                            self.tb_log.add_scalar('train_'+key, val, it)
            train_end = time.time()
            train_epochtime = train_end - train_start
            train_time += train_epochtime
            self.logger.info("Epoch {} train time consuming(2D): {}".format(epoch, train_epochtime))


            # save trained model
            trained_epoch = epoch
            if trained_epoch % ckpt_save_interval == 0 and trained_epoch:
                ckpt_name = os.path.join(self.ckpt_dir, "checkpoint_epoch_%d" % trained_epoch)
                save_checkpoint(checkpoint_state(self.model, self.optimizer, trained_epoch, it),
                                filename=ckpt_name)

            # eval one epoch
            start_time = time.time()
            if (epoch % eval_frequency) == 0 and (test_loader is not None):
                with torch.set_grad_enabled(False):
                    _, eval_dict, disp_dict = self.eval_epoch(test_loader) # removed val loss

                if self.tb_log is not None:
                    for key, val in eval_dict.items():
                        if "vis" not in key:
                            self.tb_log.add_scalar("val_"+key, val, it)

                # save model and best model
                if get_rank() == 0:

                    res = np.mean([eval_dict["LV_dice"], eval_dict["RV_dice"], eval_dict["MYO_dice"]])
                    end_time = time.time()
                    time_val = end_time - start_time

                    self.logger.info("Epoch {}  LV_dice:{}  RV_dice:{}  MYO_dice:{} "
                                     .format(epoch, eval_dict["LV_dice"], eval_dict["RV_dice"], eval_dict["MYO_dice"]))
                    self.logger.info("Epoch {} mean dice(2D): {}".format(epoch, res))
                    self.logger.info("Epoch {} val time consuming(2D): {}".format(epoch, time_val))

                    if best_res != 0:
                        _, _, best_res = load_checkpoint(filename=os.path.join(self.ckpt_dir, "model_best.pth"))
                    is_best = res > best_res
                    best_res = max(res, best_res)

                    ckpt_name = os.path.join(self.ckpt_dir, "checkpoint_epoch_%d" % trained_epoch)
                    save_best_checkpoint(checkpoint_state(self.model, self.optimizer, trained_epoch, it, performance=res),
                                    filename=ckpt_name, is_best=is_best)
        self.logger.info("total train time (2D): {} h".format(train_time/3600))



