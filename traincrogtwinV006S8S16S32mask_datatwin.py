import argparse
import datetime
import os
import shutil
import sys
import time
import numpy as np
import warnings
from functools import partial
from collections import OrderedDict
import torch.nn.functional as F
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_API_KEY"] = '99ee90fdefff711f21b8b40a0fac1bdb95da2aa5'
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather, trainMetricGPU, get_seg_image)
from utils.grasp_eval import (detect_grasps, calculate_iou, calculate_max_iou, calculate_jacquard_index, visualization)
from tqdm import tqdm
import cv2
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data as data
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR

import utils.config as config
import wandb
from utils.dataset_twin import OCIDVLGTWINDataset

from model.modelV006S8s16s32att32_twinmask import CROG_TWIN

from utils.misc import (init_random_seed, set_random_seed, setup_logger,
                        worker_init_fn)

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='config/OCID-VLG/20240816swincrogV006S8S16S32_datatmask_multiple_r50.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)

    cfg.save_dir = os.path.splitext(os.path.split(args.config)[-1])[0]

    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def train_with_grasp(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    qua_loss_metter = AverageMeter('Loss_qua', ':2.4f')
    sin_loss_metter = AverageMeter('Loss_sin', ':2.4f')
    cos_loss_metter = AverageMeter('Loss_cos', ':2.4f')
    wid_loss_metter = AverageMeter('Loss_wid', ':2.4f')
    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter = AverageMeter('Prec@50', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [
            batch_time, data_time, lr, loss_meter,
            qua_loss_metter, sin_loss_metter, cos_loss_metter, wid_loss_metter,
            iou_meter, pr_meter
        ],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs))

    model.train()
    time.sleep(2)
    end = time.time()

    # size_list = [320, 352, 384, 416, 448, 480, 512]
    # idx = np.random.choice(len(size_list))
    # new_size = size_list[idx]

    for i, data in enumerate(train_loader):
        # image, target, text = data
        # ins_mask, grasp_quality_mask, grasp_sin_mask, grasp_cos_mask, grasp_width_mask = target

        image = data["img"]
        image1 = data["img_twin"]
        text = data["word_vec"]
        ins_mask = data["mask"]
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_sin_mask = data["grasp_masks"]["sin"]
        grasp_cos_mask = data["grasp_masks"]["cos"]
        grasp_wid_mask = data["grasp_masks"]["wid"]

        data_time.update(time.time() - end)
        # data
        image = image.cuda(non_blocking=True)
        image1 = image1.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        ins_mask = ins_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_qua_mask = grasp_qua_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_sin_mask = grasp_sin_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_cos_mask = grasp_cos_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_wid_mask = grasp_wid_mask.cuda(non_blocking=True).unsqueeze(1)

        # # multi-scale training
        # image = F.interpolate(image, size=(new_size, new_size), mode='bilinear')

        # forward
        with amp.autocast():
            pred, target, loss, loss_dict = model(image, image1, text, ins_mask, grasp_qua_mask, grasp_sin_mask,
                                                  grasp_cos_mask, grasp_wid_mask)

        ins_mask_pred = pred[0]
        ins_mask_target = target[0]

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()

        # metric
        iou, pr5 = trainMetricGPU(ins_mask_pred, ins_mask_target, 0.35, 0.5)
        dist.all_reduce(loss.detach())
        dist.all_reduce(iou)
        dist.all_reduce(pr5)
        loss = loss / dist.get_world_size()
        iou = iou / dist.get_world_size()
        pr5 = pr5 / dist.get_world_size()

        loss_meter.update(loss.item(), image.size(0))
        qua_loss_metter.update(loss_dict["m_qua"], image.size(0))
        sin_loss_metter.update(loss_dict["m_sin"], image.size(0))
        cos_loss_metter.update(loss_dict["m_cos"], image.size(0))
        wid_loss_metter.update(loss_dict["m_wid"], image.size(0))
        iou_meter.update(iou.item(), image.size(0))
        pr_meter.update(pr5.item(), image.size(0))
        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)
            # if dist.get_rank() in [-1, 0]:
            #     wandb.log(
            #         {
            #             "time/batch": batch_time.val,
            #             "time/data": data_time.val,
            #             "training/lr": lr.val,
            #             "training/loss": loss_meter.val,
            #             "training/loss_qua": qua_loss_metter.val,
            #             "training/loss_sin": sin_loss_metter.val,
            #             "training/loss_cos": cos_loss_metter.val,
            #             "training/loss_wid": wid_loss_metter.val,
            #             "training/iou": iou_meter.val,
            #             "training/prec@50": pr_meter.val,
            #         },
            #         step=epoch * len(train_loader) + (i + 1))


@torch.no_grad()
def validate_with_grasp(val_loader, model, epoch, args):
    def inverse(img, mat, w, h):
        inv_img = cv2.warpAffine(img, mat, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderValue=0.)
        return inv_img

    iou_list = []
    num_correct_grasps = 0
    num_total_grasps = 0
    model.eval()
    time.sleep(2)

    num_grasps = [1, 5]
    num_correct_grasps = [0, 0]
    num_total_grasps = [0, 0]

    pbar = tqdm(val_loader)
    for data in pbar:
        # data
        image = data["img"]
        image1 = data["img_twin"]
        text = data["word_vec"]
        ins_mask = data["mask"]
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_sin_mask = data["grasp_masks"]["sin"]
        grasp_cos_mask = data["grasp_masks"]["cos"]
        grasp_wid_mask = data["grasp_masks"]["wid"]
        inverse_matrix = data["inverse"]
        ori_sizes = data["ori_size"]
        grasp_targets = data["grasps"]

        image = image.cuda(non_blocking=True)
        image1 = image1.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        ins_mask = ins_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_qua_mask = grasp_qua_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_sin_mask = grasp_sin_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_cos_mask = grasp_cos_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_wid_mask = grasp_wid_mask.cuda(non_blocking=True).unsqueeze(1)

        # inference & get predictions from model
        pred, target = model(image, image1, text, ins_mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask,
                             grasp_wid_mask)

        # predictions
        ins_mask_preds = pred[0]
        grasp_qua_mask_preds = pred[1]
        grasp_sin_mask_preds = pred[2]
        grasp_cos_mask_preds = pred[3]
        grasp_wid_mask_preds = pred[4]

        # targets
        ins_mask_targets = target[0]
        grasp_qua_mask_targets = target[1]
        grasp_sin_mask_targets = target[2]
        grasp_cos_mask_targets = target[3]
        grasp_wid_mask_targets = target[4]

        # Interpolate the predicted ins mask to the same size of input image
        ins_mask_preds = torch.sigmoid(ins_mask_preds)
        grasp_qua_mask_preds = torch.sigmoid(grasp_qua_mask_preds)
        grasp_wid_mask_preds = torch.sigmoid(grasp_wid_mask_preds)

        if ins_mask_preds.shape[-2:] != image.shape[-2:]:
            ins_mask_preds = F.interpolate(ins_mask_preds,
                                           size=image.shape[-2:],
                                           mode='bicubic',
                                           align_corners=True).squeeze(1)

            grasp_qua_mask_preds = F.interpolate(grasp_qua_mask_preds,
                                                 size=image.shape[-2:],
                                                 mode='bicubic',
                                                 align_corners=True).squeeze(1)

            grasp_sin_mask_preds = F.interpolate(grasp_sin_mask_preds,
                                                 size=image.shape[-2:],
                                                 mode='bicubic',
                                                 align_corners=True).squeeze(1)

            grasp_cos_mask_preds = F.interpolate(grasp_cos_mask_preds,
                                                 size=image.shape[-2:],
                                                 mode='bicubic',
                                                 align_corners=True).squeeze(1)

            grasp_wid_mask_preds = F.interpolate(grasp_wid_mask_preds,
                                                 size=image.shape[-2:],
                                                 mode='bicubic',
                                                 align_corners=True).squeeze(1)

        # iterate over the whole batch
        for idx in range(ins_mask_preds.shape[0]):
            inv_mat = inverse_matrix[idx]
            ori_size = ori_sizes[idx]
            h, w = ori_size

            ins_mask_pred = ins_mask_preds[idx].cpu().numpy()
            grasp_qua_mask_pred = grasp_qua_mask_preds[idx].squeeze().cpu().numpy()
            grasp_sin_mask_pred = grasp_sin_mask_preds[idx].squeeze().cpu().numpy()
            grasp_cos_mask_pred = grasp_cos_mask_preds[idx].squeeze().cpu().numpy()
            grasp_wid_mask_pred = grasp_wid_mask_preds[idx].squeeze().cpu().numpy()

            ins_mask_target = ins_mask_targets[idx].squeeze().cpu().numpy()
            grasp_target = grasp_targets[idx]
            grasp_qua_mask_target = grasp_qua_mask_targets[idx].squeeze().cpu().numpy()
            grasp_sin_mask_target = grasp_sin_mask_targets[idx].squeeze().cpu().numpy()
            grasp_cos_mask_target = grasp_cos_mask_targets[idx].squeeze().cpu().numpy()
            grasp_wid_mask_target = grasp_wid_mask_targets[idx].squeeze().cpu().numpy()

            # Inverse to original size
            ins_mask_pred = inverse(ins_mask_pred, inv_mat, w, h)
            ins_mask_pred = (ins_mask_pred > 0.35)
            grasp_qua_mask_pred = inverse(grasp_qua_mask_pred, inv_mat, w, h)
            grasp_sin_mask_pred = inverse(grasp_sin_mask_pred, inv_mat, w, h)
            grasp_cos_mask_pred = inverse(grasp_cos_mask_pred, inv_mat, w, h)
            grasp_wid_mask_pred = inverse(grasp_wid_mask_pred, inv_mat, w, h)

            ins_mask_target = inverse(ins_mask_target, inv_mat, w, h)
            grasp_qua_mask_target = inverse(grasp_qua_mask_target, inv_mat, w, h)
            grasp_sin_mask_target = inverse(grasp_sin_mask_target, inv_mat, w, h)
            grasp_cos_mask_target = inverse(grasp_cos_mask_target, inv_mat, w, h)
            grasp_wid_mask_target = inverse(grasp_wid_mask_target, inv_mat, w, h)

            # Calculate IoU between predicted instance mask and gt
            inter = np.logical_and(ins_mask_pred, ins_mask_target)
            union = np.logical_or(ins_mask_pred, ins_mask_target)

            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)

            # Calculate grasp configurations
            for i in range(len(num_grasps)):
                num_g = num_grasps[i]
                grasp_preds, _ = detect_grasps(grasp_qua_mask_pred, grasp_sin_mask_pred, grasp_cos_mask_pred,
                                               grasp_wid_mask_pred, num_g)

                j_index = calculate_jacquard_index(grasp_preds, grasp_target)

                num_correct_grasps[i] += j_index
                num_total_grasps[i] += 1

    J_index = [0, 0]
    for i in range(len(num_grasps)):
        J_index[i] = num_correct_grasps[i] / num_total_grasps[i]

    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(image.device)
    iou_list = concat_all_gather(iou_list)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    head = 'Evaluation: Epoch=[{}/{}]  IoU={:.2f}  J_index@1: {:.2f}  J_index@5: {:.2f}'.format(
        epoch, args.epochs, 100. * iou.item(), 100. * J_index[0], 100. * J_index[1])
    logger.info(head + temp)
    return iou.item(), prec, J_index
@logger.catch
def main():
    torch.multiprocessing.set_start_method('spawn')

    args= get_parser()
    args.manual_seed = init_random_seed(args.manual_seed)
    set_random_seed(args.manual_seed, deterministic=False)

    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node * args.world_size
    # mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args, ), join=True)

    children = []
    for i in range(args.world_size):
        subproc = mp.Process(target=main_worker, args=(i, args))
        children.append(subproc)
        subproc.start()

    for i in range(args.world_size):
        children[i].join()


def main_worker(gpu, args):
    # args.output_dir = os.path.join(args.output_folder, args.exp_name)

    args.output_dir =  os.path.join(args.output_folder, args.save_dir)
    # local rank & global rank
    args.gpu = gpu
    args.rank = args.rank * args.ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=args.gpu,
                 filename="train.log",
                 mode="a")

    # dist init
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)

    # wandb
    # if args.rank == 0:
    #     wandb.init(job_type="training",
    #                mode="online",
    #                config=args,
    #                project="CROG",
    #                name=args.exp_name,
    #                tags=[args.dataset, args.clip_pretrain])
    dist.barrier()

    # build model
    model= CROG_TWIN(args)
    backbone = []
    head = []
    for k, v in model.named_parameters():
        if k.startswith('backbone') and 'positional_embedding' not in k:
            backbone.append(v)
        else:
            head.append(v)
    logger.info('Backbone with decay={}, Head={}'.format(len(backbone), len(head)))
    param_list = [{
        'params': backbone,
        'initial_lr': args.lr_multi * args.base_lr
    }, {
        'params': head,
        'initial_lr': args.base_lr
    }]


    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    logger.info(model)
    logger.info(args)
    
    # build optimizer & lr scheduler
    optimizer = torch.optim.Adam(param_list,
                                 lr=args.base_lr,
                                 weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer,
                            milestones=args.milestones,
                            gamma=args.lr_decay)
    scaler = amp.GradScaler()
    
    # # resume
    # best_IoU = 0.0
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         logger.info("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(
    #             args.resume, map_location=torch.device('cpu'))
    #         args.start_epoch = checkpoint['epoch']
    #         best_IoU = checkpoint["best_iou"]
    #         state_dict = checkpoint['state_dict']
    #         new_state_dict = OrderedDict()
    #         for k, v in state_dict.items():
    #             name = k[7:] # remove `module.`
    #             new_state_dict[name] = v
    #         # load params
    #         model.load_state_dict(new_state_dict)
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         scheduler.load_state_dict(checkpoint['scheduler'])
    #         logger.info("=> loaded checkpoint '{}' (epoch {})".format(
    #             args.resume, checkpoint['epoch']))
    #     else:
    #         raise ValueError(
    #             "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
    #             .format(args.resume))
    
    
    
    model = nn.parallel.DistributedDataParallel(model.cuda(),
                                                device_ids=[args.gpu],
                                                find_unused_parameters=True)

    # build dataset
    args.batch_size = int(args.batch_size / args.ngpus_per_node)
    args.batch_size_val = int(args.batch_size_val / args.ngpus_per_node)
    args.workers = int(
        (args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)

        
    train_data = OCIDVLGTWINDataset(root_dir=args.root_path,
                            input_size=args.input_size,
                            word_length=args.word_len,
                            split='train',
                            version=args.version,
                            twin=args.use_twin_clip,args=args)
    val_data = OCIDVLGTWINDataset(root_dir=args.root_path,
                            input_size=args.input_size,
                            word_length=args.word_len,
                            split='val',
                            version=args.version,
                            twin=args.use_twin_clip,args=args)
        

    # build dataloader
    init_fn = partial(worker_init_fn,
                      num_workers=args.workers,
                      rank=args.rank,
                      seed=args.manual_seed)
    train_sampler = data.distributed.DistributedSampler(train_data,
                                                        shuffle=True)
    val_sampler = data.distributed.DistributedSampler(val_data, shuffle=False)
    train_loader = data.DataLoader(train_data,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.workers,
                                   pin_memory=True,
                                   worker_init_fn=init_fn,
                                   sampler=train_sampler,
                                   drop_last=True,
                                   collate_fn=OCIDVLGTWINDataset.collate_fn)
    val_loader = data.DataLoader(val_data,
                                 batch_size=args.batch_size_val,
                                 shuffle=False,
                                 num_workers=args.workers_val,
                                 pin_memory=True,
                                 sampler=val_sampler,
                                 drop_last=False,
                                 collate_fn=OCIDVLGTWINDataset.collate_fn)

    best_IoU = 0.0
    best_j_index = 0.0
    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
            checkpoint = torch.load(
                args.resume, map_location=map_location)
            args.start_epoch = checkpoint['epoch']
            best_IoU = checkpoint["best_iou"]
            best_j_index = checkpoint["best_j_index"]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
            
            del checkpoint
            torch.cuda.empty_cache()
        else:
            raise ValueError(
                "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
                .format(args.resume))

    # start training
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        # shuffle loader
        train_sampler.set_epoch(epoch_log)

        # train
        train_with_grasp(train_loader, model, optimizer, scheduler, scaler, epoch_log,  args)
        # evaluation
        if args.use_grasp_masks:
            iou, prec_dict, j_index = validate_with_grasp(val_loader, model, epoch_log, args)
        else:
            exit()
            # iou, prec_dict, j_index = validate_without_grasp(val_loader, model, epoch_log, args)

        # save model
        if dist.get_rank() == 0:
            lastname = os.path.join(args.output_dir, "last_model.pth")
            torch.save(
                {
                    'epoch': epoch_log,
                    'cur_iou': iou,
                    'best_iou': best_IoU,
                    'best_j_index': best_j_index,
                    'prec': prec_dict,
                    'j_index': j_index,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, lastname)
            if iou >= best_IoU:
                best_IoU = iou
                bestname = os.path.join(args.output_dir, "best_iou_model.pth")
                shutil.copyfile(lastname, bestname)
            
            if j_index[0] >= best_j_index:
                best_j_index = j_index[0]
                bestname = os.path.join(args.output_dir, "best_jindex_model.pth")
                shutil.copyfile(lastname, bestname)

        # update lr
        scheduler.step(epoch_log)
        torch.cuda.empty_cache()

    time.sleep(2)
    # if dist.get_rank() == 0:
    #     wandb.finish()

    logger.info("* Best IoU={} * ".format(best_IoU))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('* Training time {} *'.format(total_time_str))


if __name__ == '__main__':
    main()
    sys.exit(0)