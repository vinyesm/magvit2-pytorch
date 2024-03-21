from itertools import islice
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, RandomSampler
import webdataset as wds
from random import random
import numpy as np


# 1. Dataloader

train_dataset_paths = "tiny-video-data.tar"
valid_dataset_paths = "tiny-video-data.tar"

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



def assemble_frames(sample):
    print(sample.keys())
    print(sample["__key__"])
    images = np.array([sample[f"image{i:03d}.jpeg"] for i in range(1, 2)])
    return images

ds = wds.WebDataset(train_dataset_paths).decode("rgb").map(assemble_frames)

for images in ds:
    break

print(images.shape)

exit()

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
        'residual',
        'compress_space',
        ('consecutive_residual', 2),
        'linear_attend_space',
        'compress_space',
        ('consecutive_residual', 2),
        'attend_space',
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
    learning_rate = 2e-5,
    num_train_steps = 1_000_000,
    validate_every_step = 10,
)

# with trainer.trackers(project_name = 'magvit2', run_name = 'baseline'):
import wandb

wandb.init(project="magvit2")
trainer.train()