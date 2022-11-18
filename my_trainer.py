import argparse

import torch
from torch import optim, nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize
import pytorch_lightning as pl

from my_config import get_default_config, update_config, Config
from my_model import FASNet, Baseline
from src.data_io.dataset_folder import DatasetFolderFT, opencv_loader
from src.data_io import transform as trans


def exclude_from_wt_decay(named_params, weight_decay,
                          skip_list=("bn", "bias")):
    params = []
    excluded_params = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        elif any(layer_name in name for layer_name in skip_list):
            excluded_params.append(param)
        else:
            params.append(param)

    return [
        {"params": params, "weight_decay": weight_decay},
        {
            "params": excluded_params,
            "weight_decay": 0.0,
        },
    ]


def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor, top_k=(1,)):
    max_k = max(top_k)
    batch_size = targets.size(0)
    _, pred = outputs.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    ret = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1. / batch_size))
    return ret


class Module(pl.LightningModule):
    def __init__(self, conf: Config):
        super().__init__()
        self.model = Baseline(conf)
        self.conf = conf
        self.cls_criterion = nn.CrossEntropyLoss()
        self.ft_criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        samples, ft_samples, targets = batch
        outputs, feature_map = self(samples)

        loss_cls = self.cls_criterion(outputs, targets)
        loss_ft = self.ft_criterion(feature_map, ft_samples)

        if self.conf.fourier_supervision:
            loss = self.conf.cls_loss_weight * loss_cls + (1 - self.conf.cls_loss_weight) * loss_ft
        else:
            loss = loss_cls

        acc = compute_metrics(outputs, targets)[0]

        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, logger=True)
        if self.conf.fourier_supervision:
            self.log("cls_loss", loss_cls, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log("ft_loss", loss_ft, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        outputs = self(samples)

        acc = compute_metrics(outputs, targets)[0]
        self.log("val_acc", acc, on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self):
        if self.conf.exclude_decay:
            parameters = exclude_from_wt_decay(self.named_parameters(), weight_decay=self.conf.weight_decay)
            optimizer = optim.SGD(parameters, lr=self.conf.lr,
                                  momentum=self.conf.momentum)
        else:
            parameters = self.parameters()
            optimizer = optim.SGD(parameters, lr=self.conf.lr,
                                  weight_decay=self.conf.weight_decay,
                                  momentum=self.conf.momentum)

        scheduler = MultiStepLR(optimizer=optimizer,
                                milestones=self.conf.milestones,
                                gamma=self.conf.gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # step
                "frequency": 1,
            }
        }

    def train_dataloader(self):
        train_transform = trans.Compose([
            trans.ToPILImage(),
            trans.RandomResizedCrop(size=tuple(self.conf.input_size),
                                    scale=(0.9, 1.1)),
            trans.ColorJitter(brightness=0.4,
                              contrast=0.4, saturation=0.4, hue=0.1),
            trans.RandomRotation(10),
            trans.RandomHorizontalFlip(),
            trans.ToTensor()
        ])
        root_path = '{}/{}'.format(self.conf.train_root_path, self.conf.patch_info)
        train_ds = DatasetFolderFT(root_path, train_transform,
                                   None, self.conf.ft_width, self.conf.ft_height)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.conf.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.conf.num_worker)

        return train_loader

    def val_dataloader(self):
        val_transform = trans.Compose([
            trans.ToPILImage(),
            Resize(size=tuple(self.conf.input_size)),
            trans.ToTensor()
        ])
        root_path = '{}/{}'.format(self.conf.val_root_path, self.conf.patch_info)
        # No fourier for validation data
        val_ds = ImageFolder(root_path, val_transform, loader=opencv_loader)
        val_loader = DataLoader(
            val_ds,
            batch_size=self.conf.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.conf.num_worker)

        return val_loader


class MyCallback(pl.Callback):
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: Module):
        if pl_module.conf.cls_loss_weight_epoch == trainer.current_epoch:
            print("Updated weight coefficient")
            pl_module.conf.cls_loss_weight = pl_module.conf.new_cls_loss_weight


def main(args):
    conf = get_default_config()
    conf = update_config(args, conf)

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    ckpt_callback = pl.callbacks.ModelCheckpoint(save_top_k=3,
                                                 mode="max",
                                                 monitor="val_acc",  # val_acc
                                                 save_last=True,
                                                 every_n_epochs=1)
    loss_weight_cb = MyCallback()
    swa_cb = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-3,
                                                    swa_epoch_start=0.75,
                                                    annealing_epochs=5,
                                                    annealing_strategy="cos",
                                                    device=None)

    module = Module(conf)
    trainer = pl.Trainer(
        default_root_dir=conf.run_dir,
        max_epochs=conf.epochs,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=conf.log_every_n_steps,
        callbacks=[ckpt_callback, lr_monitor, loss_weight_cb, swa_cb],
        accumulate_grad_batches=conf.batch_accumulation
    )
    trainer.fit(module, ckpt_path=conf.ckpt_path)


def parse_args():
    desc = "Face Anti-Spoofing"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--patch_info", type=str, default="1_80x80",
                        help="[org_1_80x60 / 1_80x80 / 2.7_80x80 / 4_80x80]")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    _args = parse_args()
    main(_args)
