import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

Device = "cuda" if torch.cuda.is_available() else "cpu"
Train_dir = "data/train"
Val_dir = "data/val"
LR = 2e-4
Batch_size = 4
Num_workers = 1
Image_size = 256
Channels_img = 3
L1_Lambda = 100
Lambda_gp = 10
Num_epochs = 200
Load_model = False
Save_model = False
Checkpoint_disc = "disc.pth.tar"
Checkpoint_gen = "gen.pth.tar"

both_transform = A.Compose(
    [A.Resize(width=256, height=256)], additional_targets={"image0":"image"}
)

transform_image = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2()
    ]
)

transform_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2()
    ]
)
