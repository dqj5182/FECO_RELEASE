import random
import numpy as np
from contextlib import contextmanager

import torch
import torchvision.transforms as transforms

from lib.core.config import cfg


def worker_init_fn(worder_id):
    np.random.seed(np.random.get_state()[1][0] + worder_id)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optim_groups(module):
    trainable_params = {pn: p for pn, p in module.named_parameters() if p.requires_grad}
    frozen_params = {pn: p for pn, p in module.named_parameters() if not p.requires_grad}

    if not trainable_params:
        print("Warning: No trainable parameters found in Module!")

    if frozen_params:
        print(f"Info: {len(frozen_params)} parameters are frozen and will not be updated.")

    optim_groups = [
        {
            "params": list(trainable_params.values()),
        }
    ]

    return optim_groups


def get_transform(backbone_type):
    if 'vit' in backbone_type:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.MODEL.img_mean_vit, std=cfg.MODEL.img_std_vit)
        ])
    elif 'dino' in backbone_type:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.MODEL.img_mean_dino, std=cfg.MODEL.img_std_dino)
        ])
    elif 'resnet' in backbone_type:
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=cfg.MODEL.img_mean, std=cfg.MODEL.img_std)
                    ])
    else:
        raise NotImplementedError

    return transform


def move_to_device(d, device):
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                move_to_device(v, device)
            elif isinstance(v, torch.Tensor):
                d[k] = v.to(device)
    elif isinstance(d, torch.Tensor):
        return d.to(device)


@contextmanager
def tmp_freeze_module(module, eval_mode=True):
    prev_req = [p.requires_grad for p in module.parameters()]
    prev_mode = module.training
    for p in module.parameters():
        p.requires_grad_(False)
    if eval_mode:
        module.eval()
    try:
        yield
    finally:
        for p, r in zip(module.parameters(), prev_req):
            p.requires_grad_(r)
        module.train(prev_mode)


def infinite_loader(dl):
    while True:
        for b in dl:
            yield b


def load_training_setup(resume_path, model, optimizer, scheduler, device, logger):
    start_epoch = 0
    global_step = 0
    best_checkpoint_path = ''

    if resume_path:
        logger.info(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location='cpu')

        missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=True)
        if missing or unexpected:
            logger.warning(f"State dict mismatch. Missing: {missing}, Unexpected: {unexpected}")

        if optimizer is not None and 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            logger.info("Optimizer state loaded.")
        else:
            logger.info("Optimizer state not loaded.")

        if scheduler is not None and 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
            logger.info("Scheduler state loaded.")
        else:
            logger.info("Scheduler state not loaded.")

        last_epoch_saved = int(ckpt.get('epoch', -1))
        start_epoch = last_epoch_saved + 1
        global_step = int(ckpt.get('global_step', 0))
        best_checkpoint_path = resume_path

        logger.info(
            f"Resume OK. Last finished epoch: {last_epoch_saved}. "
            f"Starting at epoch: {start_epoch}. Global step: {global_step}."
        )
    else:
        logger.info("Starting fresh training run.")

    return start_epoch, global_step, best_checkpoint_path