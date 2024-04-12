from itertools import islice
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, RandomSampler
import webdataset as wds
from random import random
import numpy as np
from PIL import Image



# 1. Dataloader

train_dataset_paths = "phys101_frames.tar"
valid_dataset_paths = "phys101_frames.tar"

# class RandomApply(nn.Module):
#     def __init__(self, prob, fn, fn_else = lambda x: x):
#         super().__init__()
#         self.fn = fn
#         self.fn_else = fn_else
#         self.prob = prob
#     def forward(self, x):
#         fn = self.fn if random() < self.prob else self.fn_else
#         return fn(x)

# transform = transforms.Compose([
#     transforms.Resize(32),
#     RandomApply(0.1, transforms.RandomResizedCrop(32, scale=(0.5, 1.0), ratio=(0.98, 1.02)), transforms.CenterCrop(32)),
#     transforms.ToTensor()
# ])

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])

# TODO: handle batch size > 1
batch_size = 1
num_workers = 64

def apply_transform(image):
    # Convert the data type to uint8 if it's float32
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)
    # Reshape the array if necessary (remove singleton dimensions)
    if image.shape[-1] == 1:
        image = np.squeeze(image, axis=-1)
    pil_image = Image.fromarray(image)
    # Apply transformation here
    transformed_image = transform(pil_image)
    return transformed_image

def assemble_frames(sample):
    #print(sample.keys())
    #print(sample["__key__"])
    frames = [x for x in sample.keys() if x.split('.')[-1] == "jpeg"]
    # TODO add padding and handle it in trainer (now all tensors need same size)
    # this is a workaround for 
    # RuntimeError: stack expects each tensor to be equal size, but got [71, 3, 32, 32] at entry 0 and [89, 3, 32, 32] at entry 1
    frames = frames[:5]
    images = np.array([apply_transform(sample[frame]) for frame in frames])
    return images

train_ds = wds.WebDataset(train_dataset_paths).decode("rgb").map(assemble_frames).shuffle(100)
train_dl = wds.WebLoader(train_ds, batch_size=batch_size, num_workers=num_workers)

valid_ds = wds.WebDataset(valid_dataset_paths).decode("rgb").map(assemble_frames).shuffle(100)
valid_dl = wds.WebLoader(valid_ds, batch_size=batch_size, num_workers=num_workers)

for images in train_ds:
    break

print(images.shape)

# exit()

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
    dataset_type = 'videos',                        # 'videos' or 'images', prior papers have shown pretraining on images to be effective for video synthesis
    batch_size = batch_size,
    discr_start_after_step = 100,
    grad_accum_every = 1,
    learning_rate = 2e-5,
    num_train_steps = 1_000_000,
    validate_every_step = 10,
)

# with trainer.trackers(project_name = 'magvit2', run_name = 'baseline'):
import wandb

wandb.init(project="magvit2-video")
trainer.train()


