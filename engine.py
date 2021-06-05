import time
from typing import Iterable
import util.misc as utils
import datetime


from util.metric import nmse, psnr, ssim, AverageMeter

import torch
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from util.vis import vis_img, save_reconstructions

writer = SummaryWriter('./log/tensorboard')

def train_one_epoch(args, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, print_freq: int, device):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for data in metric_logger.log_every(data_loader, print_freq, header):

        pd, pdfs, _ = data
        target = pdfs[1]

        pd_img = pd[1].unsqueeze(1)
        pdfs_img = pdfs[0].unsqueeze(1)
        target = target.unsqueeze(1)

        pd_img = pd_img.to(device)
        pdfs_img = pdfs_img.to(device)
        target = target.to(device)

        if args.USE_MULTI_MODEL and args.USE_CL1_LOSS:
            outputs, complement = model(pdfs_img, pd_img)
            loss = criterion(outputs, target, complement, pd_img)
        elif args.USE_MULTI_MODEL:
            outputs = model(pdfs_img, pd_img)
            loss = criterion(outputs, target)
        else:
            outputs = model(pdfs_img)
            loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()

        metric_logger.update(loss=loss['loss'])
        metric_logger.update(l1_loss=loss['l1_loss'])
        if args.USE_CL1_LOSS:
            metric_logger.update(cl1_loss = loss['cl1_loss'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    global_step = int(epoch * len(data_loader) + len(data_loader))
    for key, meter in metric_logger.meters.items():
        writer.add_scalar("train/%s" % key, meter.global_avg)

    return {"loss": metric_logger.meters['loss'].global_avg, "global_step": global_step}

@torch.no_grad()
def evaluate(args, model, criterion, data_loader, device, output_dir):
    model.eval()
    criterion.eval()
    criterion.to(device)

    nmse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    output_dic = defaultdict(dict)
    target_dic = defaultdict(dict)
    input_dic = defaultdict(dict)
    start_time = time.time()

    for data in data_loader:
        pd, pdfs, _ = data
        target = pdfs[1]

        mean = pdfs[2]
        std = pdfs[3]

        fname = pdfs[4]
        slice_num = pdfs[5]

        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)

        mean = mean.to(device)
        std = std.to(device)

        pd_img = pd[1].unsqueeze(1)
        pdfs_img = pdfs[0].unsqueeze(1)

        pd_img = pd_img.to(device)
        pdfs_img = pdfs_img.to(device)
        target = target.to(device)


        if args.USE_MULTI_MODEL and args.USE_CL1_LOSS:
            outputs, _ = model(pdfs_img, pd_img)
        elif args.USE_MULTI_MODEL:
            outputs = model(pdfs_img, pd_img)
        else:
            outputs = model(pdfs_img)
        outputs = outputs.squeeze(1)

        outputs = outputs * std + mean
        target = target * std + mean
        inputs = pdfs_img.squeeze(1) * std + mean

        for i, f in enumerate(fname):
            output_dic[f][slice_num[i]] = outputs[i]
            target_dic[f][slice_num[i]] = target[i]
            input_dic[f][slice_num[i]] = inputs[i]


    for name in output_dic.keys():
        f_output = torch.stack([v for _, v in output_dic[name].items()])
        f_target = torch.stack([v for _, v in target_dic[name].items()])
        our_nmse = nmse(f_target.cpu().numpy(), f_output.cpu().numpy())
        our_psnr = psnr(f_target.cpu().numpy(), f_output.cpu().numpy())
        our_ssim = ssim(f_target.cpu().numpy(), f_output.cpu().numpy())

        nmse_meter.update(our_nmse, 1)
        psnr_meter.update(our_psnr, 1)
        ssim_meter.update(our_ssim, 1)

    save_reconstructions(output_dic, output_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print("==> Evaluate Metric")
    print("Results ----------")
    print('Evaluate time {}'.format(total_time_str))
    print("NMSE: {:.4}".format(nmse_meter.avg))
    print("PSNR: {:.4}".format(psnr_meter.avg))
    print("SSIM: {:.4}".format(ssim_meter.avg))
    print("------------------")

    return {'NMSE': nmse_meter.avg, 'PSNR': psnr_meter.avg, 'SSIM':ssim_meter.avg}


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:num_total_examples]


@torch.no_grad()
def distributed_evaluate(args, model, criterion, data_loader, device, dataset_len):
    model.eval()
    criterion.eval()
    criterion.to(device)

    nmse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    start_time = time.time()

    output_list = []
    target_list = []
    id_list = []
    slice_list = []

    for data in data_loader:
        pd, pdfs, id = data
        target = pdfs[1]

        mean = pdfs[2]
        std = pdfs[3]

        fname = pdfs[4]
        slice_num = pdfs[5]

        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)

        id = id.to(device)

        slice_num = slice_num.to(device)
        mean = mean.to(device)
        std = std.to(device)

        pd_img = pd[1].unsqueeze(1)
        pdfs_img = pdfs[0].unsqueeze(1)

        pd_img = pd_img.to(device)
        pdfs_img = pdfs_img.to(device)
        target = target.to(device)

        if args.USE_MULTI_MODEL and args.USE_CL1_LOSS:
            outputs, _ = model(pdfs_img, pd_img)
        elif args.USE_MULTI_MODEL:
            outputs = model(pdfs_img, pd_img)
        else:
            outputs = model(pdfs_img)
        outputs = outputs.squeeze(1)
        outputs = outputs * std + mean
        target = target * std + mean

        output_list.append(outputs)
        target_list.append(target)
        id_list.append(id)
        slice_list.append(slice_num)

    final_id = distributed_concat(torch.cat((id_list), dim=0), dataset_len)
    final_output = distributed_concat(torch.cat((output_list), dim=0), dataset_len)
    final_target = distributed_concat(torch.cat((target_list), dim=0), dataset_len)
    final_slice = distributed_concat(torch.cat((slice_list), dim=0), dataset_len)

    output_dic = defaultdict(dict)
    target_dic = defaultdict(dict)

    final_id = final_id.cpu().numpy()

    for i, f in enumerate(final_id):
        output_dic[f][final_slice[i]] = final_output[i]
        target_dic[f][final_slice[i]] = final_target[i]

    for name in output_dic.keys():
        f_output = torch.stack([v for _, v in output_dic[name].items()])
        f_target = torch.stack([v for _, v in target_dic[name].items()])
        our_nmse = nmse(f_target.cpu().numpy(), f_output.cpu().numpy())
        our_psnr = psnr(f_target.cpu().numpy(), f_output.cpu().numpy())
        our_ssim = ssim(f_target.cpu().numpy(), f_output.cpu().numpy())

        nmse_meter.update(our_nmse, 1)
        psnr_meter.update(our_psnr, 1)
        ssim_meter.update(our_ssim, 1)



    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print("==> Evaluate Metric")
    print("Results ----------")
    print('Evaluate time {}'.format(total_time_str))
    print("NMSE: {:.4}".format(nmse_meter.avg))
    print("PSNR: {:.4}".format(psnr_meter.avg))
    print("SSIM: {:.4}".format(ssim_meter.avg))
    print("------------------")

    return {'NMSE': nmse_meter.avg, 'PSNR': psnr_meter.avg, 'SSIM':ssim_meter.avg}

def do_vis(dataloader):

    for idx, data in enumerate(dataloader):
        pd, pdfs, _ = data

        pd_img, pd_target, pd_mean, pd_std, pd_fname, pd_slice = pd
        pdfs_img, pdfs_target, pdfs_mean, pdfs_std, pdfs_fname, pdfs_slice = pdfs

        pd_mean = pd_mean.unsqueeze(1).unsqueeze(2)
        pd_std = pd_std.unsqueeze(1).unsqueeze(2)

        pdfs_mean = pdfs_mean.unsqueeze(1).unsqueeze(2)
        pdfs_std = pdfs_std.unsqueeze(1).unsqueeze(2)

        pdfs_img = pdfs_img * pdfs_std + pdfs_mean
        pdfs_target = pdfs_target * pdfs_std + pdfs_mean
        pd_target = pd_target * pd_std + pd_mean

        vis_img(pdfs_img.squeeze(0), str(idx), 'pdfs_lr', 'show_rc')
        vis_img(pdfs_target.squeeze(0), str(idx), 'pdfs_target', 'show_rc')
        vis_img(pd_target.squeeze(0), str(idx), 'pd_target', 'show_rc')




