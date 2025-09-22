import torch
from utils import *
import torch.nn as nn
import torch.optim as optim
from Config import config
from Customdataset.dataset import CustomDataset
from Model.discriminator import Discriminator
from Model.generator import Generator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True

def train(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop): # (x, y) --> (images, labels)
        x = x.to(config.Device)
        y = y.to(config.Device)

        # Train discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            G_l1loss = l1_loss(y_fake, y) * config.L1_Lambda
            G_loss = (G_fake_loss + G_l1loss)

        gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real = torch.sigmoid(D_real).mean().item(),
                D_fake = torch.sigmoid(D_fake).mean().item()
            )

def main():
    disc = Discriminator(in_channels=3).to(config.Device)
    gen = Generator(in_channels=3).to(config.Device)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LR, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LR, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_Loss = nn.L1Loss()

    if config.Load_model:
        load_checkpoint(
            config.Checkpoint_gen, gen, opt_gen, config.LR
        )
        load_checkpoint(
            config.Checkpoint_disc, disc, opt_disc, config.LR
        )

    train_dataset = CustomDataset(root_dir=config.Train_dir)
    train_loader = DataLoader(
        train_dataset, batch_size=config.Batch_size,
        shuffle=True, num_workers=config.Num_workers
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = CustomDataset(root_dir=config.Val_dir)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.Num_epochs):
        train(
            disc, gen, train_loader, opt_disc, opt_gen, L1_Loss, BCE, g_scaler, d_scaler
        )

        if config.Save_model and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.Checkpoint_gen)
            save_checkpoint(disc, opt_disc, filename=config.Checkpoint_disc)

        save_examples(gen, val_loader, epoch, folder='evaluation')


if __name__ == '__main__':
    main()
