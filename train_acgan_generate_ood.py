import os
from dataset import CANDatasetEnet as CANDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from tfrecord.writer import TFRecordWriter
from tfrecord.tools.tfrecord2idx import create_index

# === Dataset Loader ===
def set_loader(opt):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([normalize])

    support_1_dataset = CANDataset(opt.support_1_folder, window_size=32, is_train=False, transform=transform)
    support_2_dataset = CANDataset(opt.support_2_folder, window_size=32, is_train=False, transform=transform)
    
    support_loader1 = DataLoader(support_1_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    support_loader2 = DataLoader(support_2_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    return support_loader1, support_loader2

# === Models ===
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=2, channels=3, img_size=64):
        super().__init__()
        input_dim = latent_dim + num_classes
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256 * 8 * 8),
            nn.BatchNorm1d(256 * 8 * 8),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, channels, 4, stride=2, padding=1),  # 64x64
            nn.Sigmoid()
        )

    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, channels=3, img_size=64, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )
        ds = img_size // 8
        self.adv = nn.Linear(256 * ds * ds, 1)               # Real/Fake
        self.aux = nn.Linear(256 * ds * ds, num_classes)     # Class prediction

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return self.adv(x), self.aux(x)  # ✅ Trả ra 2 giá trị


# === Save to TFRecord ===
def save_to_tfrecord(X, path):
    writer = TFRecordWriter(path)
    for x in X:
        x_np = x.cpu().numpy().astype(np.float32)
        id_seq, data_seq, timestamp = x_np[0], x_np[1], x_np[2]
        feature = {
            "id_seq": (id_seq.flatten().tolist(), "float"),
            "data_seq": (data_seq.flatten().tolist(), "float"),
            "timestamp": (timestamp.flatten().tolist(), "float"),
            "label": [-1, "int"]
        }
        writer.write(feature)
    writer.close()
    create_index(path, path + ".index")
    print(f"✅ Saved {len(X)} samples to: {path}")


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    fake = torch.ones_like(d_interpolates).to(real_samples.device)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

def train_acgan_on_dataset(dataset_loader, label_name, opt, device):
    G = Generator().to(device)
    D = Discriminator().to(device)
    loss_adv = nn.BCEWithLogitsLoss()
    loss_aux = nn.CrossEntropyLoss()

    optim_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    print(f"🔄 Training AC-GAN on {label_name}...")
    for epoch in range(opt.epochs):
        for real_imgs, _ in dataset_loader:
            bs = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            valid = torch.ones(bs, 1).to(device)
            fake = torch.zeros(bs, 1).to(device)
            real_labels = torch.full((bs,), 1, dtype=torch.long).to(device)  # all real labels = 1

            z = torch.randn(bs, 100).to(device)
            gen_labels = torch.full((bs,), 0, dtype=torch.long).to(device)  # generate label 0
            gen_label_onehot = F.one_hot(gen_labels, num_classes=2).float()

            gen_imgs = G(z, gen_label_onehot)

            # Train Discriminator
            real_pred, real_aux = D(real_imgs)
            fake_pred, fake_aux = D(gen_imgs.detach())

            loss_real = loss_adv(real_pred, valid) + loss_aux(real_aux, real_labels)
            loss_fake = loss_adv(fake_pred, fake) + loss_aux(fake_aux, gen_labels)
            # loss_D = (loss_real + loss_fake) / 2
            lambda_gp = 10
            gp = compute_gradient_penalty(D, real_imgs.data, gen_imgs.data)
            loss_D = (loss_real + loss_fake) / 2 + lambda_gp * gp
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

            # Train Generator
            fake_pred, fake_aux = D(gen_imgs)
            loss_G = loss_adv(fake_pred, valid) + loss_aux(fake_aux, gen_labels)
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

        print(f"[Epoch {epoch+1}/{opt.epochs}] Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")

    # Generate pseudo unknown attacks
    G.eval()
    Z = torch.randn(opt.num_ood, 100).to(device)
    gen_labels = torch.full((opt.num_ood,), 0, dtype=torch.long).to(device)
    labels_onehot = F.one_hot(gen_labels, num_classes=2).float()
    with torch.no_grad():
        X_fake = G(Z, labels_onehot)

    return X_fake


# === Main ===
def main(opt):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    loader1, loader2 = set_loader(opt)
    # === Train on each separately ===
    X_fake1 = train_acgan_on_dataset(loader1, "Support 1", opt, device)
    X_fake2 = train_acgan_on_dataset(loader2, "Support 2", opt, device)

    # === Merge and save ===
    X_merged = torch.cat([X_fake1, X_fake2], dim=0)
    save_to_tfrecord(X_merged, opt.output_path)
    print("✅ Done generating and saving OOD samples.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--support_1_folder', type=str, required=True)
    parser.add_argument('--support_2_folder', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='generated_acgan_ood.tfrecord')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_ood', type=int, default=1000)
    opt = parser.parse_args()
    main(opt)
