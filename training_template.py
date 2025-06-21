import argparse
import math
import os, sys
import random
import datetime
import time
from typing import List
import json
import numpy as np
from copy import deepcopy
import shutil

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.optim

from torch.utils.tensorboard import SummaryWriter
from loss import KGCL,HCCL


from utils.logger import setup_logger
from models.hierarchicaltransformer import build_HierTran
from utils.metric import voc_mAP
from utils.misc import clean_state_dict
from utils.slconfig import get_raw_dict

from data_utils.get_dataset_new import get_datasets
from data_utils.metrics import validate_f1


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.split(filename)[0] +'/model_best.pth.tar')
        # shutil.copyfile(filename, os.path.split(filename)[0] + '/model_best.pth.tar')

# from dataset.get_dataset import get_datasets
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', val_only=False):
        self.name = name
        self.fmt = fmt
        self.val_only = val_only
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class AverageMeterHMS(AverageMeter):
    """Meter for timer in HH:MM:SS format"""
    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val}'
        else:
            fmtstr = '{name} {val} ({sum})'
        return fmtstr.format(name=self.name,
                             val=str(datetime.timedelta(seconds=int(self.val))),
                             sum=str(datetime.timedelta(seconds=int(self.sum))))


def parser_args():
    available_models = ['Q2L-R101-448']
    parser = argparse.ArgumentParser(description='HAKGL Training')
    parser.add_argument('--exp_model')
    parser.add_argument('--dataname', help='dataname', default='odir', choices=['odir','rfmid','kaggle'])
    parser.add_argument('--dataset_dir', help='dir of dataset')
    parser.add_argument('--img_size', default=224, type=int, help='size of input images')

    parser.add_argument('--output', metavar='DIR', default="",
                        help='path to output folder')
    parser.add_argument('--num_class', default=12, type=int,
                        help="ODIR:12 ;RFMiD:20； Kaggle:50")
    parser.add_argument('--coarse_num_class', default=8, type=int,
                        help="ODIR:12 ;RFMiD:16; Kaggle: 17")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model. default is False. ')
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'Adam_twd'],
                        help='which optim to use')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='Q2L-R101-448',
                        choices=available_models,
                        help='model architecture: ' + ' | '.join(available_models) +
                             ' (default: Q2L-R101-448)')

    # loss
    parser.add_argument('--eps', default=1e-5, type=float,
                        help='eps for focal loss (default: 1e-5)')
    parser.add_argument('--dtgfl', action='store_true', default=True,
                        help='disable_torch_grad_focal_loss in asl')
    parser.add_argument('--gamma_pos', default=0, type=float,
                        metavar='gamma_pos', help='gamma pos for simplified asl loss')
    parser.add_argument('--gamma_neg', default=2, type=float,
                        metavar='gamma_neg', help='gamma neg for simplified asl loss')
    parser.add_argument('--loss_dev', default=-1, type=float,
                        help='scale factor for loss')
    parser.add_argument('--loss_clip', default=0.0, type=float,
                        help='scale factor for clip')

    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--val_interval', default=1, type=int, metavar='N',
                        help='interval of validation')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 128)')

    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)',
                        dest='weight_decay')

    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume_omit', default=[], type=str, nargs='*')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--ema-decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')
    parser.add_argument('--ema-epoch', default=0, type=int, metavar='M',
                        help='start ema epoch')

    # 分布式训练参数（单卡下不需要）
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=31, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    # data aug
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')
    parser.add_argument('--length', type=int, default=-1,
                        help='length of the holes. suggest to use default setting -1.')
    parser.add_argument('--cut_fact', type=float, default=0.5,
                        help='mutual exclusion with length. ')

    parser.add_argument('--orid_norm', action='store_true', default=False,
                        help='using mean [0,0,0] and std [1,1,1] to normalize input images')

    # * Transformer
    parser.add_argument('--enc_layers', default=0, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=8192, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=2048, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine',),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--keep_other_self_attn_dec', action='store_true',
                        help='keep the other self attention modules in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true',
                        help='keep the first self attention module in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_input_proj', action='store_true',
                        help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")

    # * training
    parser.add_argument('--amp', action='store_true', default=True,
                        help='apply amp')
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help='apply early stop')
    parser.add_argument('--kill-stop', action='store_true', default=False,
                        help='apply early stop')
    parser.add_argument('--device', default=0, type=int,)
    args = parser.parse_args()
    return args


def get_args():
    args = parser_args()
    return args


best_mAP = 0
best_f1_samples = 0


def main():
    args = get_args()
    code_dir = os.path.join(args.output, "code")
    os.makedirs(code_dir, exist_ok=True)
    current_dir = os.getcwd()

    for item in os.listdir(current_dir):
        src_path = os.path.join(current_dir, item)
        if os.path.abspath(src_path) == os.path.abspath(code_dir):
            continue

        dst_path = os.path.join(code_dir, item)
        if os.path.isfile(src_path):
            shutil.copy(src_path, dst_path)
        elif os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)



    args.world_size = 1
    args.rank = 0
    args.local_rank = 0

    seed_everything(args.seed)


    torch.cuda.set_device(args.device)

    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, distributed_rank=0, color=False, name="Q2L")
    logger.info("Command: " + ' '.join(sys.argv))
    path = os.path.join(args.output, "_config.json")
    with open(path, 'w') as f:
        json.dump(get_raw_dict(args), f, indent=2)
    logger.info("Full config saved to {}".format(path))

    logger.info('world size: {}'.format(1))
    logger.info('rank: {}'.format(0))
    logger.info('local_rank: {}'.format(args.local_rank))

    return main_worker(args, logger)


def main_worker(args, logger):
    global best_mAP
    global best_f1_samples

    model = build_HierTran(args)
    model = model.to(torch.device("cuda"))
    ema_m = ModelEma(model, args.ema_decay)


    criterion = nn.BCEWithLogitsLoss(reduction="none")



    args.lr_mult = args.batch_size / 256
    if args.optim == 'AdamW':
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
        ]
        optimizer = getattr(torch.optim, args.optim)(
            param_dicts,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
        )
    elif args.optim == 'Adam_twd':
        parameters = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.Adam(
            parameters,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        )
    else:
        raise NotImplementedError

    # tensorboard
    summary_writer = SummaryWriter(log_dir=args.output)

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if 'state_dict' in checkpoint:
                state_dict = clean_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                state_dict = clean_state_dict(checkpoint['model'])
            else:
                raise ValueError("No model or state_dicr Found!!!")
            logger.info("Omitting {}".format(args.resume_omit))
            for omit_name in args.resume_omit:
                del state_dict[omit_name]
            model.load_state_dict(state_dict, strict=False)
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            del checkpoint
            del state_dict
            torch.cuda.empty_cache()
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    train_dataset, val_dataset, test_dataset = get_datasets(args)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        _, mAP, f1_micros, f1_macros, f1_samples = validate(val_loader, model, criterion, args, logger)
        logger.info(' * mAP {mAP:.5f}'.format(mAP=mAP))
        logger.info('Validation: f1_micros: {}, f1_macros: {}, f1_samples: {}'.format(f1_micros, f1_macros, f1_samples))
        return

    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    losses_ema = AverageMeter('Loss_ema', ':5.3f', val_only=True)
    mAPs = AverageMeter('mAP', ':5.5f', val_only=True)
    f1s = AverageMeter('mAP', ':5.5f', val_only=True)
    mAPs_ema = AverageMeter('mAP_ema', ':5.5f', val_only=True)
    f1s_ema = AverageMeter('mAP', ':5.5f', val_only=True)

    f1_micros = AverageMeter('f1_micro', ':5.5f', val_only=True)
    f1_macros = AverageMeter('f1_macro', ':5.5f', val_only=True)
    f1_samples = AverageMeter('f1_samples', ':5.5f', val_only=True)

    f1_micros_ema = AverageMeter('f1_micro_ema', ':5.5f', val_only=True)
    f1_macros_ema = AverageMeter('f1_macro_ema', ':5.5f', val_only=True)
    f1_samples_ema = AverageMeter('f1_samples_ema', ':5.5f', val_only=True)

    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, losses, f1_micros, f1_macros, f1_samples, losses_ema, f1_micros_ema, f1_macros_ema,
         f1_samples_ema],
        prefix='=> Test Epoch: ')

    # one cycle learning rate
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                        epochs=args.epochs, pct_start=0.2)

    end = time.time()
    best_epoch = -1
    best_regular_epoch = -1
    best_regular_f1 = 0
    best_ema_f1 = 0
    regular_f1_list = []
    ema_f1_list = []
    torch.cuda.empty_cache()
    for epoch in range(args.start_epoch, args.epochs):
        if args.ema_epoch == epoch:
            ema_m = ModelEma(model, args.ema_decay)
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        start_epoch = time.time()
        loss, fine_l, middle_l, coarse_l = train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch,
                                                 args, logger)
        end_epoch = time.time()
        time_epoch = end_epoch - start_epoch
        print("the time of one epoch:", time_epoch)

        if summary_writer:
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('fine_loss', np.mean(np.array(fine_l)), epoch)
            summary_writer.add_scalar('middle_loss', np.mean(np.array(middle_l)), epoch)
            summary_writer.add_scalar('coarse_loss', np.mean(np.array(coarse_l)), epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % args.val_interval == 0:
            val_info = validate(val_loader, model, criterion, args, logger)
            ema_val_info = validate(val_loader, ema_m.module, criterion, args, logger)

            fine_info = val_info['fine']
            coarse_info = val_info['coarse']

            fine_ema_info = ema_val_info['fine']
            coarse_ema_info = ema_val_info['coarse']

            fine_loss_val, fine_mAP, fine_f1_micros_val, fine_f1_macros_val, fine_f1_samples_val = fine_info['loss'], fine_info['mAP'], fine_info['f1_micros'], fine_info['f1_macros'], fine_info['f1_samples']
            fine_loss_ema, fine_mAP_ema, fine_f1_micros_ema, fine_f1_macros_ema, fine_f1_samples_ema = fine_ema_info['loss'], fine_ema_info['mAP'], fine_ema_info['f1_micros'], fine_ema_info['f1_macros'], fine_ema_info['f1_samples']

            losses.update(fine_loss_val)
            mAPs.update(fine_mAP)
            f1s.update(fine_f1_samples_val)
            losses_ema.update(fine_loss_ema)
            mAPs_ema.update(fine_mAP_ema)
            f1s_ema.update(fine_f1_samples_ema)
            epoch_time.update(time.time() - end)
            end = time.time()
            eta.update(epoch_time.avg * (args.epochs - epoch - 1))

            regular_f1_list.append(fine_f1_samples_val)
            ema_f1_list.append(fine_f1_samples_ema)

            progress.display(epoch, logger)

            if summary_writer:
                summary_writer.add_scalar('val_fine/loss', fine_loss_val, epoch)
                summary_writer.add_scalar('val_fine/mAP', fine_mAP, epoch)
                summary_writer.add_scalar('val_fine/f1_micros', fine_f1_micros_val, epoch)
                summary_writer.add_scalar('val_fine/f1_macros', fine_f1_macros_val, epoch)
                summary_writer.add_scalar('val_fine/f1_samples', fine_f1_samples_val, epoch)

                summary_writer.add_scalar('val_fine/loss_ema', fine_loss_ema, epoch)
                summary_writer.add_scalar('val_fine/mAP_ema', fine_mAP_ema, epoch)
                summary_writer.add_scalar('val_fine/f1_micros_ema', fine_f1_micros_ema, epoch)
                summary_writer.add_scalar('val_fine/f1_macros_ema', fine_f1_macros_ema, epoch)
                summary_writer.add_scalar('val_fine/f1_samples_ema', fine_f1_samples_ema, epoch)

                summary_writer.add_scalar('val_coarse/loss', coarse_info['loss'], epoch)
                summary_writer.add_scalar('val_coarse/mAP', coarse_info['mAP'], epoch)
                summary_writer.add_scalar('val_coarse/f1_micros', coarse_info['f1_micros'], epoch)
                summary_writer.add_scalar('val_coarse/f1_macros', coarse_info['f1_macros'], epoch)
                summary_writer.add_scalar('val_coarse/f1_samples', coarse_info['f1_samples'], epoch)

                summary_writer.add_scalar('val_coarse/loss_ema', coarse_ema_info['loss'], epoch)
                summary_writer.add_scalar('val_coarse/mAP_ema', coarse_ema_info['mAP'], epoch)
                summary_writer.add_scalar('val_coarse/f1_micros_ema', coarse_ema_info['f1_micros'], epoch)
                summary_writer.add_scalar('val_coarse/f1_macros_ema', coarse_ema_info['f1_macros'], epoch)
                summary_writer.add_scalar('val_coarse/f1_samples_ema', coarse_ema_info['f1_samples'], epoch)


            if fine_f1_samples_val > best_regular_f1:
                best_regular_f1 = fine_f1_samples_val
                best_regular_epoch = epoch
            if fine_f1_samples_ema > best_ema_f1:
                best_ema_f1 = fine_f1_samples_ema

            if fine_f1_samples_ema > fine_f1_samples_val:
                fine_f1_samples_val = fine_f1_samples_ema
                state_dict = ema_m.module.state_dict()
            else:
                state_dict = model.state_dict()
            is_best = fine_f1_samples_val > best_f1_samples
            if is_best:
                best_epoch = epoch
            best_f1_samples = max(fine_f1_samples_val, best_f1_samples)

            logger.info("{} | Set best f1 {} in ep {}".format(epoch, best_f1_samples, best_epoch))
            logger.info("   | best regular f1 {} in ep {}".format(best_regular_f1, best_regular_epoch))

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': state_dict,
                'best_f1_samples': best_f1_samples,
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, filename=os.path.join(args.output, 'checkpoint.pth.tar'))

            if math.isnan(fine_loss_val) or math.isnan(fine_loss_ema):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_f1_samples': best_f1_samples,
                    'optimizer': optimizer.state_dict(),
                }, is_best=is_best, filename=os.path.join(args.output, 'checkpoint_nan.pth.tar'))
                logger.info('Loss is NaN, break')
                sys.exit(1)

            if args.early_stop:
                if best_epoch >= 0 and epoch - max(best_epoch, best_regular_epoch) > 8:
                    if len(ema_f1_list) > 1 and ema_f1_list[-1] < best_ema_f1:
                        logger.info("epoch - best_epoch = {}, stop!".format(epoch - best_epoch))
                        if args.kill_stop:
                            filename = sys.argv[0].split(' ')[0].strip()
                            killedlist = kill_process(filename, os.getpid())
                            logger.info("Kill all process of {}: ".format(filename) + " ".join(killedlist))
                        break

    print("Best f1_samples:", best_f1_samples)

    if summary_writer:
        summary_writer.close()
    return 0


def train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    batch_time = AverageMeter('T', ':5.3f')
    data_time = AverageMeter('DT', ':5.3f')
    speed_gpu = AverageMeter('S1', ':.1f')
    speed_all = AverageMeter('SA', ':.1f')
    losses = AverageMeter('Loss', ':5.3f')
    lr_meter = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, speed_gpu, speed_all, lr_meter, losses, mem],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr_meter.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    model.train()
    end = time.time()
    fine_loss = []
    middle_loss = []
    coarse_loss = []

    # load expert knowledge
    if args.dataname == 'odir':
        knowledge_embeds = torch.load("data/ODIR/FLAIR_ODIR_with_EK.pt").cuda()
    elif args.dataname == 'rfmid':
        knowledge_embeds = torch.load("data/RFMiD/FLAIR_ODIR_with_EK.pt").cuda()
    elif args.dataname == 'kaggle':
        knowledge_embeds = torch.load("data/KaggleDR+/FLAIR_KaggleDR+_with_EK.pt").cuda()

    hccl_loss_function = HCCL()
    kgcl_Loss_function = KGCL(knowledge_embeds,dataname=args.dataname)

    if args.datname == 'odir' or args.datname == 'kaggle':
        for i, data in enumerate(stable(train_loader, args.seed + epoch)):
            data_time.update(time.time() - end)

            images = data['image'].cuda(non_blocking=True)
            sup_labels = data['sup_label'].cuda(non_blocking=True)
            sub_labels = data['sub_label'].cuda(non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                output = model(images)
                loss_fine = criterion(output[0], sub_labels.float())
                loss_middle = None
                loss_coarse = criterion(output[1], sup_labels.float())


                loss_coarse1 = loss_coarse.sum() / loss_coarse.size(0)
                loss_fine1 = loss_fine.sum() / loss_fine.size(0)
                embedding_img = output[-1]
                embedding_img_coarse = output[-2]
                loss_hccl = hccl_loss_function(embedding_img, embedding_img_coarse)
                loss_kgcl = kgcl_Loss_function(embedding_img, sub_labels.float())


            fine_l = loss_fine.unsqueeze(1).sum() / loss_fine.size(0)
            fine_loss.append(fine_l.item())
            if loss_middle is not None:
                middle_l = loss_middle.unsqueeze(1).sum() / loss_middle.size(0)
                middle_loss.append(middle_l.item())
            else:
                middle_l = 0
                middle_loss.append(0)
            coarse_l = loss_coarse.unsqueeze(1).sum() / loss_coarse.size(0)
            coarse_loss.append(coarse_l.item())
            loss = loss_coarse1  + loss_fine1 + loss_hccl + loss_kgcl
    else:
        for i, data in enumerate(stable(train_loader, args.seed + epoch)):

            data_time.update(time.time() - end)
            images = data['image'].cuda(non_blocking=True)
            coarse_labels = data['sup_label'].cuda(non_blocking=True)
            middle_1_labels = data['middle_1_label'].cuda(non_blocking=True)
            middle_2_labels = data['middle_2_label'].cuda(non_blocking=True)
            fine_labels = data['sub_label'].cuda(non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                output = model(images)
                loss_fine = criterion(output[0], fine_labels.float())
                loss_middle2 = criterion(output[1], middle_2_labels.float())
                loss_middle1 = criterion(output[2], middle_1_labels.float())
                loss_coarse = criterion(output[3], coarse_labels.float())

                loss_fine = loss_fine.sum() / loss_fine.size(0)
                loss_middle2 = loss_middle2.sum() / loss_middle2.size(0)
                loss_middle1 = loss_middle1.sum() / loss_middle1.size(0)
                loss_coarse = loss_coarse.sum() / loss_coarse.size(0)

                embedding_img_fine = output[-1]
                embedding_img_middle1 = output[-2]
                embedding_img_middle2 = output[-3]
                embedding_img_coarse = output[-4]
                loss_hccl = hccl_loss_function(embedding_img_fine, embedding_img_middle1) + hccl_loss_function(embedding_img_middle1, embedding_img_middle2) + hccl_loss_function(embedding_img_middle2, embedding_img_coarse)
                loss_kgcl = kgcl_Loss_function(embedding_img_fine, fine_labels.float())


            loss = loss_fine + loss_middle2 + loss_middle1 + loss_coarse + loss_hccl + loss_kgcl

    losses.update(loss.item(), images.size(0))
    mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)


    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    lr_meter.update(get_learning_rate(optimizer))
    if epoch >= args.ema_epoch:
        ema_m.update(model)
    batch_time.update(time.time() - end)
    end = time.time()
    speed_gpu.update(images.size(0) / batch_time.val, batch_time.val)
    speed_all.update(images.size(0) / batch_time.val, batch_time.val)

    if i % args.print_freq == 0:
        progress.display(i, logger)


    return losses.avg, fine_loss, middle_loss, coarse_loss

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


@torch.no_grad()
def validate(val_loader, model, criterion, args, logger):
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, mem],
        prefix='Test: ')

    model.eval()
    fine_saved_data = []
    middle_saved_data = []
    coarse_saved_data = []
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images = data['image'].cuda(non_blocking=True)
            coarse_labels = data['sup_label'].cuda(non_blocking=True)
            fine_labels = data['sub_label'].cuda(non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                output = model(images)
                loss = criterion(output[0], fine_labels.float())
                loss = loss.sum() / loss.size(0)
                if args.loss_dev > 0:
                    loss *= args.loss_dev
                output_fine = torch.sigmoid(output[0])
                output_coarse = torch.sigmoid(output[1])
            losses.update(loss.item(), images.size(0))
            mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

            fine_item = torch.cat((output_fine.detach().cpu(), fine_labels.detach().cpu()), 1)
            coarse_item = torch.cat((output_coarse.detach().cpu(), coarse_labels.detach().cpu()), 1)
            fine_saved_data.append(fine_item)
            coarse_saved_data.append(coarse_item)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i, logger)

        fine_saved_data = torch.cat(fine_saved_data, 0).numpy()
        fine_saved_name = 'fine_saved_data_tmp.txt'
        np.savetxt(os.path.join(args.output, fine_saved_name), fine_saved_data)

        coarse_saved_data = torch.cat(coarse_saved_data, 0).numpy()
        coarse_saved_name = 'coarse_saved_data_tmp.txt'
        np.savetxt(os.path.join(args.output, coarse_saved_name), coarse_saved_data)

        print("Calculating metrics:")
        fine_filenamelist = [os.path.join(args.output, fine_saved_name)]
        fine_mAP, fine_aps = voc_mAP(fine_filenamelist, args.num_class, return_each=True)
        fine_f1_dict = validate_f1(fine_filenamelist, args.num_class)
        fine_f1_micros = fine_f1_dict['val_micro']
        fine_f1_macros = fine_f1_dict['val_macro']
        fine_f1_samples = fine_f1_dict['val_samples']

        coarse_filenamelist = [os.path.join(args.output, coarse_saved_name)]
        coarse_mAP, coarse_aps = voc_mAP(coarse_filenamelist, args.coarse_num_class, return_each=True)
        coarse_f1_dict = validate_f1(coarse_filenamelist, args.coarse_num_class)
        coarse_f1_micros = coarse_f1_dict['val_micro']
        coarse_f1_macros = coarse_f1_dict['val_macro']
        coarse_f1_samples = coarse_f1_dict['val_samples']


        logger.info("  fine_mAP: {}".format(fine_mAP))
        logger.info("  fine_aps: {}".format(np.array2string(fine_aps, precision=5)))
        logger.info("  fine_f1_micros: {}".format(np.array2string(fine_f1_micros, precision=5)))
        logger.info("  fine_f1_macros: {}".format(np.array2string(fine_f1_macros, precision=5)))
        logger.info("  fine_f1_samples: {}".format(np.array2string(fine_f1_samples, precision=5)))

        logger.info("  coarse_mAP: {}".format(coarse_mAP))
        logger.info("  coarse_aps: {}".format(np.array2string(coarse_aps, precision=5)))
        logger.info("  coarse_f1_micros: {}".format(np.array2string(coarse_f1_micros, precision=5)))
        logger.info("  coarse_f1_macros: {}".format(np.array2string(coarse_f1_macros, precision=5)))
        logger.info("  coarse_f1_samples: {}".format(np.array2string(coarse_f1_samples, precision=5)))

    return {
        "fine":{
            "loss": losses.avg,
            "mAP": fine_mAP,
            "aps": fine_aps,
            "f1_micros": fine_f1_micros,
            "f1_macros": fine_f1_macros,
            "f1_samples": fine_f1_samples
        },
        "coarse":{
            "loss": losses.avg,
            "mAP": coarse_mAP,
            "aps": coarse_aps,
            "f1_micros": coarse_f1_micros,
            "f1_macros": coarse_f1_macros,
            "f1_samples": coarse_f1_samples
        }
            }

def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def kill_process(filename: str, holdpid: int) -> List[str]:
    import subprocess, signal
    res = subprocess.check_output("ps aux | grep {} | grep -v grep | awk '{{print $2}}'".format(filename), shell=True,
                                  cwd="./")
    res = res.decode('utf-8')
    idlist = [i.strip() for i in res.split('\n') if i != '']
    print("kill: {}".format(idlist))
    for idname in idlist:
        if idname != str(holdpid):
            os.kill(int(idname), signal.SIGKILL)
    return idlist


def compare_loss(loss_fine, loss_middle, loss_coarse):
    if loss_middle is None:
        loss_fine = loss_fine.unsqueeze(1)
        loss_coarse = loss_coarse.unsqueeze(1)
        w = torch.cat((loss_fine, loss_coarse), dim=1)
        max_w = torch.max(w, dim=1)
        loss = max_w[0].sum() / loss_fine.size(0)
    else:
        loss_fine = loss_fine.unsqueeze(1)
        loss_middle = loss_middle.unsqueeze(1)
        loss_coarse = loss_coarse.unsqueeze(1)
        w = torch.cat((loss_fine, loss_middle, loss_coarse), dim=1)
        max_w = torch.max(w, dim=1)
        loss = max_w[0].sum() / loss_fine.size(0)
    return loss


def seed_everything(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)


def stable(dataloader, seed):
    seed_everything(seed)
    return dataloader


if __name__ == '__main__':
    main()
