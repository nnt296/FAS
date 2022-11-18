from typing import Union, List

import torch
from src.utility import get_width_height, get_kernel


class Config:
    # ----------------------training---------------
    lr: float = 1e-3

    # [9, 13, 15]
    # milestones: List[int] = [17, 29]  # down learning rate
    milestones: List[int] = [12]  # down learning rate
    gamma: float = 0.1
    epochs: int = 32
    momentum: float = 0.9
    batch_size: int = 64

    exclude_decay: bool = True
    weight_decay: float = 5e-4

    cls_loss_weight: float = 0.5

    cls_loss_weight_epoch: int = -1  # Set to -1 to disable
    new_cls_loss_weight: float = 0.8

    imagenet_pretrained: bool = True
    ckpt_path: str = None

    batch_accumulation: int = 1  # Set to other to enable
    num_worker: int = 6

    fourier_supervision: bool = True

    # model
    num_classes: int = 2
    input_channel = 3
    embedding_size = 128

    patch_info: str = ""  # Patch to train on. Example: 1.0_384x384
    input_size: Union[List[int], int] = [384, 384]
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    kernel_size: Union[List[int], int] = [24, 24]
    ft_height: int = 2 * kernel_size[0]
    ft_width: int = 2 * kernel_size[1]

    # dataset
    train_root_path: str = './datasets/train'
    val_root_path: str = './datasets/val'

    # log path
    run_dir: str = "runs"
    log_every_n_steps: int = 5

    def set_hw(self, height, width):
        self.input_size = [height, width]
        self.kernel_size = get_kernel(height, width)
        self.ft_height = 2 * self.kernel_size[0]
        self.ft_width = 2 * self.kernel_size[1]


def get_default_config():
    return Config()


def update_config(args, conf: Config):
    # conf.devices = args.devices
    conf.patch_info = args.patch_info
    w_input, h_input = get_width_height(args.patch_info)
    conf.input_size = [h_input, w_input]
    conf.kernel_size = get_kernel(h_input, w_input)
    conf.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # resize fourier image size
    conf.ft_height = 2 * conf.kernel_size[0]
    conf.ft_width = 2 * conf.kernel_size[1]

    return conf
