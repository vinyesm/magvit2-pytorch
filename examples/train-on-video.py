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


transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor()
])

batch_size = 8
num_workers = 64


def assemble_frames(sample):
    # print(sample.keys())
    print(sample["__key__"])
    frames = [x for x in sample.keys() if x.split('.')[-1] == "jpeg"]
    # TODO add padding and handle it in trainer (now all tensors need same size)
    if frames:
        # print(f"nb frames: {len(frames)}")
        frames = [sample[frame] for frame in frames[:10]]
        # [print(frame.size) for frame in frames] 
        images = np.array([transform(frame) for frame in frames])
        images_transposed = np.transpose(images, (1, 0, 2, 3))
        return images_transposed
    else:
        return None

train_ds = wds.WebDataset(train_dataset_paths).shuffle(10000).decode("pil").map(assemble_frames)
train_dl = wds.WebLoader(train_ds, batch_size=batch_size, num_workers=num_workers)

valid_ds = wds.WebDataset(valid_dataset_paths).shuffle(10000).decode("pil").map(assemble_frames)
valid_dl = wds.WebLoader(valid_ds, batch_size=1, num_workers=num_workers)

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
    image_size = 64,
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

wandb.init(project="magvit2-video", save_code=True)
trainer.train()

# training pil decoding, 10-frame: done -> cosmic-yogurt-30
# TODO: rgb decoding vs pil
# TODO: torchrgb vs pil vs rgb
# TODO: handle videos of different lengths
# TODO: handle non-square videos
# TODO: add adversarial and perceptual loss


