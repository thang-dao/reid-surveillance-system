import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn

from utils.meter import AverageMeter
from utils.metrics import R1_mAP, accuracy
from torch.utils.tensorboard import SummaryWriter

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query):
    log_period = cfg.LOG_PERIOD
    checkpoint_period = cfg.CHECKPOINT_PERIOD
    eval_period = cfg.EVAL_PERIOD

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cuda"   
    epochs = cfg.MAX_EPOCHS

    writer = SummaryWriter(log_dir=cfg.LOG_DIR)
    logger = logging.getLogger('{}.train'.format(cfg.PROJECT_NAME))
    logger.info('start training')

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  
        model.to(device)
    # model.to(cfg.DEVICE_ID)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    loss_id = AverageMeter()
    loss_global = AverageMeter()
    loss_local = AverageMeter()
    loss_center = AverageMeter()
    evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM, metrics=cfg.EVAL_METRIC)
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        loss_id.reset()
        loss_global.reset()
        loss_local.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step()
        model.train()
        for n_iter, (img, vid) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            
            if cfg.LOSS_TYPE == 'aligned+pcb':
                score, feat, local_feat, y = model(img, target)
                loss, _loss_id, _loss_global, _loss_local, preds = loss_fn(score, feat, local_feat, target, y)
                loss.backward()
                optimizer.step()
                acc = accuracy(score, target)
                loss_meter.update(loss.item(), img.shape[0])
                loss_id.update(_loss_id.item(), img.shape[0])
                loss_global.update(_loss_global.item(), img.shape[0])
                loss_local.update(_loss_local.item(), img.shape[0])
                acc_meter.update(acc, 1)
                if (n_iter + 1) % log_period == 0:
                        logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, LossID: {:.3f}, LossGlobal: {:.3f}, LossLocal: {:.3f},  Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, loss_id.avg, loss_global.avg, loss_local.avg, acc_meter.avg, scheduler.get_lr()[0]))

            elif cfg.LOSS_TYPE == 'aligned+pcb+center':
                score, feat, local_feat, y = model(img, target)
                loss, _loss_id, _loss_global, _loss_local, _loss_center, preds = loss_fn(score, feat, local_feat, target, y)
                loss.backward()
                optimizer.step()
                if 'center' in cfg.LOSS_TYPE:
                    for param in center_criterion.parameters():
                        param.grad.data *= (1. / cfg.CENTER_LOSS_WEIGHT)
                        optimizer_center.step()
                acc = (score.max(1)[1] == target).float().mean()
                loss_meter.update(loss.item(), img.shape[0])
                loss_id.update(_loss_id.item(), img.shape[0])
                loss_global.update(_loss_global.item(), img.shape[0])
                loss_local.update(_loss_local.item(), img.shape[0])
                loss_center.update(_loss_center.item(), img.shape[0])
                acc_meter.update(acc, 1)
                if (n_iter + 1) % log_period == 0:
                        logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, LossID: {:.3f}, LossGlobal: {:.3f}, LossLocal: {:.3f}, LossCenter: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, loss_id.avg, loss_global.avg, loss_local.avg, loss_center.avg, acc_meter.avg, scheduler.get_lr()[0]))

            elif cfg.LOSS_TYPE == 'aligned+arcface':
                score, feat, local_feat = model(img,target)
                loss, _loss_id, _loss_global, _loss_local = loss_fn(score, feat, local_feat, target)  
                loss.backward()
                optimizer.step()  
                acc = accuracy(score, target)
                loss_meter.update(loss.item(), img.shape[0])
                loss_id.update(_loss_id.item(), img.shape[0])
                loss_global.update(_loss_global.item(), img.shape[0])
                loss_local.update(_loss_local.item(), img.shape[0])
                acc_meter.update(acc, 1)
                if (n_iter + 1) % log_period == 0:
                        logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, LossID: {:.3f}, LossGlobal: {:.3f}, LossLocal: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, loss_id.avg, loss_global.avg, loss_local.avg, acc_meter.avg, scheduler.get_lr()[0]))

            elif cfg.LOSS_TYPE == 'aligned+arcface+center':
                score, feat, local_feat = model(img,target)
                loss, _loss_id, _loss_global, _loss_local, _loss_center = loss_fn(score, feat, local_feat, target)  
                loss.backward()
                optimizer.step()  
                acc = accuracy(score, target)
                loss_meter.update(loss.item(), img.shape[0])
                loss_id.update(_loss_id.item(), img.shape[0])
                loss_global.update(_loss_global.item(), img.shape[0])
                loss_local.update(_loss_local.item(), img.shape[0])
                loss_center.update(_loss_center.item(), img.shape[0])
                acc_meter.update(acc, 1)
                if (n_iter + 1) % log_period == 0:
                        logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, LossID: {:.3f}, LossGlobal: {:.3f}, LossLocal: {:.3f}, LossCenter: {:.3f},  Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, loss_id.avg, loss_global.avg, loss_local.avg, loss_center.avg, acc_meter.avg, scheduler.get_lr()[0]))
                                        
            elif cfg.LOSS_TYPE == 'softmax+triplet+aligned':
                score, feat, local_feat = model(img,target)
                loss, _loss_id, _loss_global, _loss_local = loss_fn(score, feat, local_feat, target)  
                loss.backward()
                optimizer.step()  
                acc = accuracy(score, target)
                loss_meter.update(loss.item(), img.shape[0])
                loss_id.update(_loss_id.item(), img.shape[0])
                loss_global.update(_loss_global.item(), img.shape[0])
                loss_local.update(_loss_local.item(), img.shape[0])
                acc_meter.update(acc, 1)
                if (n_iter + 1) % log_period == 0:
                        logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, LossID: {:.3f}, LossGlobal: {:.3f}, LossLocal: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, loss_id.avg, loss_global.avg, loss_local.avg, acc_meter.avg, scheduler.get_lr()[0]))
                                        
            if writer is not None:
                writer.add_scalar('Train/Loss_g', loss_global.avg, n_iter+1)
                writer.add_scalar('Train/Loss_l', loss_local.avg, n_iter+1)
                writer.add_scalar('Train/Loss_x', loss_id.avg, n_iter+1)
                writer.add_scalar('Train/Loss', loss_meter.avg, n_iter+1)
                writer.add_scalar('Train/Loss', loss_center.avg, n_iter+1)
                writer.add_scalar('Train/Acc', acc_meter.avg, n_iter+1)
                writer.add_scalar(
                    'Train/Lr', scheduler.get_lr()[0], n_iter+1
                )
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))
        
        if not os.path.exists(cfg.OUTPUT_DIR):
            os.mkdir(cfg.OUTPUT_DIR)

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            model.to(device)
            model.eval()
            for n_iter, (img, vid, camid, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    _, feat, lf = model(img)
                    evaluator.update((feat, lf, vid, camid))

            cmc, mAP, _, _, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query, epoch=0):
    # device = "cuda"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger('{}.test'.format(cfg.PROJECT_NAME))
    logger.info("Enter inferencing")
    evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM, \
                       method=cfg.TEST_METHOD, reranking=cfg.RERANKING, test_distance=cfg.TEST_DISTANCE, metrics=cfg.EVAL_METRIC)
    evaluator.reset()
    model.eval()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)
    
    img_path_list = []
    for n_iter, (img, pid, camid, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.FLIP_FEATS == 'on':
                feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                    f = model(img)[0]
                    feat = feat + f
            else:
                _, feat, lf = model(img)
            evaluator.update((feat, lf, pid, camid))
            img_path_list.extend(imgpath)
    cmc, mAP, distmat, pids, camids, qfeats, gfeats, qlfeats, glfeats = evaluator.compute()
    with open(os.path.join(cfg.LOG_DIR, 'LOG_TEST.txt'), "a+") as f:
        if cfg.RERANKING:
            f.write("Re-ranking:  ")
        f.write("Epoch: {}".format(epoch))
        f.write("   mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            f.write(" CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        f.write('\n')
    np.save(os.path.join(cfg.LOG_DIR, cfg.DIST_MAT) , distmat)
    np.save(os.path.join(cfg.LOG_DIR, cfg.PIDS), pids)
    np.save(os.path.join(cfg.LOG_DIR, cfg.CAMIDS), camids)
    np.save(os.path.join(cfg.LOG_DIR, cfg.IMG_PATH), img_path_list[num_query:])
    torch.save(qfeats, os.path.join(cfg.LOG_DIR, cfg.Q_FEATS))
    torch.save(gfeats, os.path.join(cfg.LOG_DIR, cfg.G_FEATS))
    torch.save(qlfeats, os.path.join(cfg.LOG_DIR, cfg.QL_FEATS))
    torch.save(glfeats, os.path.join(cfg.LOG_DIR, cfg.GL_FEATS))

    logger.info("Validation Results")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
