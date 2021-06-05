
import torch
import time
import os
import datetime
import random
import numpy as np
import argparse

from models import build_model_from_name
from models.loss import build_criterion
from data.fastmri import build_dataset
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
from engine import train_one_epoch, evaluate
from util.misc import init_distributed_mode, get_rank
from config import build_config


def main(args, work):
    init_distributed_mode(args)

    # build criterion and model first
    model = build_model_from_name(args, work)
    criterion = build_criterion(args)

    seed = args.SEED + get_rank()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(args.SOLVER.DEVICE)

    model.to(device)
    criterion.to(device)

    if args.RESUME != '':
        print('resume from %s' % args.RESUME)
        checkpoint = torch.load(args.RESUME, map_location=lambda storage, loc: storage)
        checkpoint = checkpoint['model']
        checkpoint = {key.replace("module.", ""): val for key, val in checkpoint.items()}
        model.load_state_dict(checkpoint, strict=True)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params: %.2f M' % (n_parameters / 1024 / 1024))

    # build dataset
    dataset_val = build_dataset(args, mode='val')

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    dataloader_val = DataLoader(dataset_val, batch_size=args.SOLVER.BATCH_SIZE,
                                sampler=sampler_val, num_workers=args.SOLVER.NUM_WORKERS,
                                pin_memory=True)

    start_time = time.time()
    evaluate(args, model, criterion, dataloader_val, device, args.TEST_OUTPUTDIR)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('evaluate time {}'.format(total_time_str))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="a unit Cross Multi modity transformer")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument(
        "--experiment", default="edsr", help="choose a experiment to do")
    args = parser.parse_args()

    print('doing ', args.experiment)

    cfg = build_config(args.experiment)
    main(cfg, args.experiment)





