# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import os
import math
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
import models_mae
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_cb
from pytorch_lightning.loggers import WandbLogger
def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='~/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters on tpus
    parser.add_argument("--tpus", default=8, type=int, help='The number of tpus to use')

    # logging / ckpting / resume parameters
    parser.add_argument('--experiment_name', default=None, type=str, help='Experiment name shown in wandb')
    parser.add_argument('--project', default='mae-tpus', type=str, help='Experiment project group in wandb')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume_path', default=None, type=str, help='resume from checkpoint')
    parser.add_argument('--resume_version', default=None, type=str, help='resume wandb version')    

    return parser

class MaskedAutoEncoder(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
        print("Model = %s" % str(self.model))

        world_size = args.tpus
        eff_batch_size = args.batch_size * args.accum_iter * world_size # The number of tpus
        self.loader_length = args.total_data // eff_batch_size

        ## Define optimizer / lr_scheduler hyperparameters
        if args.lr is None:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256

        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        self.lr = args.lr
        self.warmup_epochs = args.warmup_epochs
        self.total_epochs = args.epochs
        self.accum_iter = args.accum_iter

        # Define mask ratio
        self.mask_ratio = args.mask_ratio

    def configure_optimizers(self):
        param_groups = optim_factory.add_weight_decay(self.model, self.args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=self.args.lr, betas=(0.9, 0.95))
        print(optimizer)

        return [optimizer]
    
    def forward(self, image):
        return self.model(image)
    
    def training_step(self, data, batch_idx):
        image, _ = data

        loss, _, _ = self.model(image, self.mask_ratio)

        self.log_dict({'train/loss':loss})
        self.log_dict({'epoch': self.current_epoch, 'step':self.global_step})

        ## Adjust learning rate
        if batch_idx % self.accum_iter == 0:
            self.adjust_learning_rate(batch_idx)

        return loss
    
    def adjust_learning_rate(self, data_iter_step):
        optimizer = self.optimizers()
        epoch = self.current_epoch + (data_iter_step / self.loader_length)

        if data_iter_step % self.accum_iter:
            """Decay the learning rate with half-cycle cosine after warmup"""
            ### Linearly warm up learning rate until epoch < warmup_epochs
            if epoch < self.warmup_epochs: 
                lr = self.lr * epoch / self.warmup_epochs 
            else:
                lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * \
                    (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)))
            for param_group in optimizer.param_groups:
                if "lr_scale" in param_group:
                    param_group["lr"] = lr * param_group["lr_scale"]
                else:
                    param_group["lr"] = lr
            return lr

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # fix the seed for reproducibility
    pl.seed_everything(args.seed)

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    args.total_data = len(dataset_train)

    # Define loader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # Define model
    pl_mae_model = MaskedAutoEncoder(args)

    # Define wandb logger
    if not args.resume: ## Fool proof
        args.resume_version = None
        args.resume_path = None
    
    wandb_logger = WandbLogger(
        name=args.experiment_name,
        project=args.project,
        log_model=True,
        version=args.resume_version,
    )

    wandb_logger.log_hyperparams(args)
    wandb_logger.watch(pl_mae_model, log=True)

    checkpoint_last_k_callback = pl_cb.ModelCheckpoint(
        every_n_epochs=1,
        monitor='step',
        mode='max',
        save_top_k=5,
        filename="{name}_{epoch:03d}",
        save_last=True
    )
    lr_monitor = pl_cb.LearningRateMonitor(logging_interval='step')

    pl_trainer = pl.Trainer(
        callbacks=[checkpoint_last_k_callback, lr_monitor],
        max_epochs=args.epochs,
        logger=wandb_logger,
        precision=16,
        resume_from_checkpoint=args.resume_path,
        profiler="simple",
        track_grad_norm = 2,
        tpu_cores=args.tpus,
    )

    pl_trainer.fit(pl_mae_model, data_loader_train)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
