import os
import glob
import random

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tf
import matplotlib.pyplot as plt

# ============================================================
# Reproducibility
# ============================================================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# Config (the values you can tune are gathered here)
# ============================================================
IMAGE_SIZE = 512          # must be divisible by 256 (8 downsampling layers -> 2^8)
BATCH_SIZE = 4
EPOCHS = 200
LR_G = 3e-4
LR_D = 2e-4
BETAS = (0.5, 0.999)  # ??
LAMBDA_L1 = 150            # ?? weight of L1 term vs adversarial term in generator loss
DECAY_START_EPOCH = EPOCHS // 2   # ?? start linear LR decay halfway through
CKPT_DIR = "checkpoints"
SAMPLE_DIR = "samples"
LOG_EVERY = 1

assert IMAGE_SIZE % 256 == 0, "IMAGE_SIZE must be a multiple of 256 for this 8-layer U-Net."

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Dataset
# ============================================================
class TextureNormalDataset(Dataset):
    """
    root_dir/
        textures/*.jpg|*.png
        normals/<same_name>.jpg|*.png
    """

    def __init__(self, root_dir, image_size=IMAGE_SIZE, augment=True):
        self.root_dir = root_dir
        self.image_size = image_size
        self.augment = augment  # turn OFF for validation/test sets

        self.texture_paths = sorted(
            glob.glob(os.path.join(root_dir, "textures", "*.jpg")) +
            glob.glob(os.path.join(root_dir, "textures", "*.png"))
        )

        self.normal_paths = []
        for tex_path in self.texture_paths:
            name = os.path.splitext(os.path.basename(tex_path))[0]
            normal_path_png = os.path.join(root_dir, "normals", name + ".png")
            normal_path_jpg = os.path.join(root_dir, "normals", name + ".jpg")

            if os.path.exists(normal_path_png):
                self.normal_paths.append(normal_path_png)
            elif os.path.exists(normal_path_jpg):
                self.normal_paths.append(normal_path_jpg)
            else:
                raise FileNotFoundError(f"Normal map not found for: {name}")

        assert len(self.texture_paths) == len(self.normal_paths), "Dataset mismatch!"
        if len(self.texture_paths) == 0:
            raise RuntimeError(f"No images found under {root_dir}")

    def __len__(self):
        return len(self.texture_paths)

    def transform_pair(self, x, y):
        # Resize first (deterministic, independent of augmentation)
        x = tf.resize(x, (self.image_size, self.image_size))
        y = tf.resize(y, (self.image_size, self.image_size))

        if self.augment and random.random() > 0.5:
            x = tf.hflip(x)
            y = tf.hflip(y)

            # IMPORTANT: a normal map is not a plain image. R/G/B encode the
            # surface normal's X/Y/Z direction (usually stored as
            # channel = (component + 1) / 2, i.e. in [0, 1]).
            # Mirroring the surface horizontally flips the sign of the
            # X component of every normal vector, so the encoded R channel
            # must be inverted too (R' = 1 - R). Skipping this teaches the
            # network a physically wrong texture -> normal mapping for
            # every flipped sample.
            y = tf.to_tensor(y)
            y[0, :, :] = 1.0 - y[0, :, :]
            y = tf.to_pil_image(y)

        x = tf.to_tensor(x)
        y = tf.to_tensor(y)

        x = tf.normalize(x, (0.5,) * 3, (0.5,) * 3)
        y = tf.normalize(y, (0.5,) * 3, (0.5,) * 3)

        return x, y

    def __getitem__(self, idx):
        x = Image.open(self.texture_paths[idx]).convert("RGB")
        y = Image.open(self.normal_paths[idx]).convert("RGB")
        return self.transform_pair(x, y)


# ============================================================
# Weight init (standard for pix2pix-style GANs; default PyTorch
# init tends to make GAN training unstable / slow to converge)
# ============================================================
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# ============================================================
# Generator: U-Net with skip connections
# ============================================================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        def down(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        def up(in_c, out_c, dropout=False):
            layers = [
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
            ]
            if dropout:
                layers.append(nn.Dropout(0.2))
            return nn.Sequential(*layers)

        self.d1 = down(3, 64, normalize=False)
        self.d2 = down(64, 128)
        self.d3 = down(128, 256)
        self.d4 = down(256, 512)
        self.d5 = down(512, 512)
        self.d6 = down(512, 512)
        self.d7 = down(512, 512, normalize=False)
        self.d8 = down(512, 512, normalize=False)

        self.u1 = up(512, 512, dropout=True)
        self.u2 = up(1024, 512, dropout=True)
        self.u3 = up(1024, 512, dropout=True)
        self.u4 = up(1024, 512)
        self.u5 = up(1024, 256)
        self.u6 = up(512, 128)
        self.u7 = up(256, 64)

        self.final = nn.ConvTranspose2d(128, 3, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8)
        u2 = self.u2(torch.cat([u1, d7], 1))
        u3 = self.u3(torch.cat([u2, d6], 1))
        u4 = self.u4(torch.cat([u3, d5], 1))
        u5 = self.u5(torch.cat([u4, d4], 1))
        u6 = self.u6(torch.cat([u5, d3], 1))
        u7 = self.u7(torch.cat([u6, d2], 1))

        return self.tanh(self.final(torch.cat([u7, d1], 1)))


# ============================================================
# Discriminator: PatchGAN
# ============================================================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(6, 64, normalize=False),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], 1))


# ============================================================
# Evaluation / visualization helpers (safe at module level —
# these only define functions, they don't execute anything)
# ============================================================
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += nn.L1Loss()(pred, y).item()
    avg = total_loss / len(loader)
    print("Average L1 Loss on test set:", avg)
    model.train()
    return avg


def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def save_result(model, dataset, index, device, out_dir=SAMPLE_DIR):
    model.eval()
    x, y = dataset[index]
    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device))[0]
    model.train()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1); plt.title("Texture"); plt.axis("off"); plt.imshow(denorm(x))
    plt.subplot(1, 3, 2); plt.title("Generated Normal"); plt.axis("off"); plt.imshow(denorm(pred))
    plt.subplot(1, 3, 3); plt.title("Real Normal"); plt.axis("off"); plt.imshow(denorm(y))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"result_{index}.png"))
    plt.close()


# Linear decay to 0 after DECAY_START_EPOCH, flat before that (standard pix2pix schedule)
def lr_lambda(epoch):
    if epoch < DECAY_START_EPOCH:
        return 1.0
    progress = (epoch - DECAY_START_EPOCH) / max(1, (EPOCHS - DECAY_START_EPOCH))
    return max(0.0, 1.0 - progress)


# ============================================================
# Everything below actually RUNS the program (builds datasets,
# spawns DataLoader worker processes, trains, saves results).
#
# On Windows, multiprocessing uses "spawn": each DataLoader worker
# re-imports this file from scratch. If this code sat at module
# level (no guard), every worker would re-run it too — including
# building its own DataLoader and spawning its own workers —
# causing exactly the recursive-launch crash you hit. Wrapping it
# in `if __name__ == "__main__":` ensures workers only import
# class/function definitions, never re-execute the training run.
# ============================================================
def main():
    # ---- Data ----
    train_dataset = TextureNormalDataset("dataset/train", augment=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        drop_last=True,  # avoids a size-1 batch breaking BatchNorm on the last step
    )

    # NOTE: augment=False here on purpose — evaluation must be deterministic.
    test_dataset = TextureNormalDataset("dataset/test", augment=False)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # ---- Models ----
    G = Generator().to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

    optimizer_G = optim.Adam(G.parameters(), lr=LR_G, betas=BETAS)
    optimizer_D = optim.Adam(D.parameters(), lr=LR_D, betas=BETAS)

    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
    scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda)

    # ---- Training loop ----
    history = {"loss_D": [], "loss_G": []}

    for epoch in range(EPOCHS):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Train Discriminator
            fake_y = G(x)

            pred_real = D(x, y)                      # single forward pass, reused below
            pred_fake = D(x, fake_y.detach())         # detach: don't train G here

            real_label = torch.ones_like(pred_real)
            fake_label = torch.zeros_like(pred_fake)

            loss_D_real = criterion_GAN(pred_real, real_label)
            loss_D_fake = criterion_GAN(pred_fake, fake_label)
            loss_D = (loss_D_real + loss_D_fake) * 0.5

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # Train Generator
            pred_fake_for_G = D(x, fake_y)  # no detach: gradients must flow into G
            loss_G_GAN = criterion_GAN(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
            loss_G_L1 = criterion_L1(fake_y, y)
            loss_G = loss_G_GAN + LAMBDA_L1 * loss_G_L1

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            if i % LOG_EVERY == 0:
                print(f"[{epoch+1}/{EPOCHS}] step {i} | D: {loss_D.item():.3f} | G: {loss_G.item():.3f}")

        history["loss_D"].append(loss_D.item())
        history["loss_G"].append(loss_G.item())

        scheduler_G.step()
        scheduler_D.step()

        # Checkpoint periodically so a crash doesn't lose everything
        if (epoch + 1) % 25 == 0 or (epoch + 1) == EPOCHS:
            torch.save(G.state_dict(), os.path.join(CKPT_DIR, f"G_epoch{epoch+1}.pt"))
            torch.save(D.state_dict(), os.path.join(CKPT_DIR, f"D_epoch{epoch+1}.pt"))

    # Final weights
    torch.save(G.state_dict(), os.path.join(CKPT_DIR, "G_final.pt"))
    torch.save(D.state_dict(), os.path.join(CKPT_DIR, "D_final.pt"))

    # Loss curves (separate figures)
    plt.figure()
    plt.plot(history["loss_G"], label="G loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Generator training loss")
    plt.savefig(os.path.join(SAMPLE_DIR, "loss_curve_G.png"))
    plt.close()

    plt.figure()
    plt.plot(history["loss_D"], label="D loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Discriminator training loss")
    plt.savefig(os.path.join(SAMPLE_DIR, "loss_curve_D.png"))
    plt.close()

    # ---- Evaluation ----
    evaluate(G, test_loader, device)

    # ---- Qualitative results -> saved to disk instead of blocking plt.show() ----
    n_show = min(20, len(train_dataset))
    for idx in range(n_show):
        save_result(G, train_dataset, idx, device)

    print(f"Saved {n_show} qualitative samples to '{SAMPLE_DIR}/', checkpoints to '{CKPT_DIR}/'.")


if __name__ == "__main__":
    main()

