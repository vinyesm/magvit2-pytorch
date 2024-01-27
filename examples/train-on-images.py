from itertools import islice
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader
import webdataset as wds

# 1. Dataloader

dataset_folder = "images.tar.gz"

transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor()
])

batch_size = 1
num_workers = 1
ds = wds.WebDataset(dataset_folder).decode("pilrgb").rename(image="jpg;png;jpeg;webp").map_dict(image=transform).to_tuple("image")
dl = wds.WebLoader(ds, batch_size=batch_size, num_workers=num_workers)

# 2. model

from magvit2_pytorch import (
    VideoTokenizer,
    VideoTokenizerTrainer
)

tokenizer = VideoTokenizer(
    image_size = 128,
    init_dim = 64,
    max_dim = 512,
    codebook_size = 1024,
    layers = (
        'residual',
        'compress_space',
        ('consecutive_residual', 2),
        'compress_space',
        ('consecutive_residual', 2),
        'linear_attend_space',
        'compress_space',
        ('consecutive_residual', 2),
        'attend_space',
        'compress_time',
        ('consecutive_residual', 2),
        'compress_time',
        ('consecutive_residual', 2),
        'attend_time',
    )
)

# 3. Train

trainer = VideoTokenizerTrainer(
    tokenizer,
    use_wandb_tracking = True,
    train_dataloader = dl,     # folder of either videos or images, depending on setting below
    dataset_type = 'images',                        # 'videos' or 'images', prior papers have shown pretraining on images to be effective for video synthesis
    batch_size = batch_size,
    grad_accum_every = 8,
    learning_rate = 2e-5,
    num_train_steps = 1_000_000
)

with trainer.trackers(project_name = 'magvit2', run_name = 'baseline'):
    trainer.train()