from itertools import islice
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, RandomSampler
import webdataset as wds
from random import random


# 1. Dataloader

# dataset_folder = "images.tar.gz"
train_dataset_paths = "toy-data/shard{1..7}.tar.gz::toy-data/shard{1..7}.tar.gz"
valid_dataset_paths = "toy-data/shard{8..9}.tar.gz"

class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else = lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob
    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)

transform = transforms.Compose([
    transforms.Resize(32),
    RandomApply(0.1, transforms.RandomResizedCrop(32, scale=(0.5, 1.0), ratio=(0.98, 1.02)), transforms.CenterCrop(32)),
    transforms.ToTensor()
])



batch_size = 32
num_workers = 4
train_ds = wds.WebDataset(train_dataset_paths).decode("pilrgb").rename(image="jpg;png;jpeg;webp").map_dict(image=transform).to_tuple("image").shuffle(800)
train_dl = wds.WebLoader(train_ds, batch_size=batch_size, num_workers=num_workers)

valid_ds = wds.WebDataset(valid_dataset_paths).decode("pilrgb").rename(image="jpg;png;jpeg;webp").map_dict(image=transform).to_tuple("image")
valid_dl = wds.WebLoader(valid_ds, batch_size=batch_size, num_workers=num_workers)

# 2. model

from magvit2_pytorch import (
    VideoTokenizer,
    VideoTokenizerTrainer
)

tokenizer = VideoTokenizer(
    image_size = 32,
    init_dim = 16,
    max_dim = 32,
    codebook_size = 1024, # 1024
    layers = (
        # 'residual',
        # 'compress_space',
        # ('consecutive_residual', 2),
        # 'linear_attend_space',
        # 'compress_space',
        # ('consecutive_residual', 2),
        # 'attend_space',
    ),
    perceptual_loss_weight = 0.,
    adversarial_loss_weight = 0.,
    quantizer_aux_loss_weight = 0.,
    use_gan = False
)

# 3. Train

trainer = VideoTokenizerTrainer(
    tokenizer,
    use_wandb_tracking = True,
    train_dataloader = train_dl,    
    valid_dataloader = valid_dl,
    dataset_type = 'images',                        # 'videos' or 'images', prior papers have shown pretraining on images to be effective for video synthesis
    batch_size = batch_size,
    discr_start_after_step = 100,
    grad_accum_every = 1,
    learning_rate = 1e-4,
    num_train_steps = 1_000_000
)

with trainer.trackers(project_name = 'magvit2', run_name = 'baseline'):
    trainer.train()