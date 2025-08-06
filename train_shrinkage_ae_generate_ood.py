import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from tfrecord.writer import TFRecordWriter
from tfrecord.tools.tfrecord2idx import create_index

from dataset import CANDatasetEnet as CANDataset

# ===== Shrinkage Autoencoder =====
class ShrinkageAutoencoder(nn.Module):
    def __init__(self, input_shape=(3, 64, 64), latent_dim=128):
        super().__init__()
        c, h, w = input_shape
        self.input_dim = c * h * w
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.input_dim),
            nn.Sigmoid()
        )
        self.output_shape = input_shape

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z).view(-1, *self.output_shape)
        return z, x_hat


# ===== Data Loader =====
def set_loader(opt):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([normalize])

    train_dataset = CANDataset(
        root_dir=opt.data_folder, window_size=32, is_train=True, transform=transform
    )

    support_1_dataset = CANDataset(
        root_dir=opt.support_1_folder, window_size=32, is_train=False, transform=transform
    )
    support_2_dataset = CANDataset(
        root_dir=opt.support_2_folder, window_size=32, is_train=False, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    support_1_loader = DataLoader(support_1_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    support_2_loader = DataLoader(support_2_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    return train_loader, support_1_loader, support_2_loader


# ===== Shrinkage loss =====
def shrinkage_loss(x_hat, x, z, lambda_shrink=1e-4):
    recon_loss = F.mse_loss(x_hat, x)
    shrink = torch.mean(torch.norm(z, p=2, dim=1) ** 2)
    return recon_loss + lambda_shrink * shrink


# ===== Soft Brownian Offset =====
def soft_brownian_offset(Z, d_minus=2.0, d_plus=0.05, sigma=0.1, kappa=7, max_iter=100):
    Z_np = Z.cpu().numpy()
    N, D = Z_np.shape
    Z_ood = []
    for _ in range(N):
        x = Z_np[np.random.randint(0, N)]
        x_hat = x.copy()
        for _ in range(max_iter):
            gamma = np.random.normal(0, 1, size=D)
            gamma /= np.linalg.norm(gamma)
            x_hat += d_plus * gamma
            d_star = np.min(np.linalg.norm(Z_np - x_hat, axis=1))
            rho = 1.0 / (1 + np.exp((d_star + d_minus) / (sigma * d_minus) - kappa))
            if np.random.rand() < rho:
                break
        Z_ood.append(x_hat)
    return torch.tensor(np.stack(Z_ood)).float().to(Z.device)


# ===== TFRecord Writer =====
def write_ood_to_tfrecord(X_ood, tfrecord_path):
    assert X_ood.shape[1] == 3, "Expected 3 channels (id_seq, data_seq, timestamp)"
    N = X_ood.shape[0]
    writer = TFRecordWriter(tfrecord_path)
    for i in range(N):
        x = X_ood[i].cpu().numpy().astype(np.float32)
        id_seq = x[0].flatten().tolist()
        data_seq = x[1].flatten().tolist()
        timestamp = x[2].flatten().tolist()
        label = [-1]
        feature = {
            "id_seq": (id_seq, "float"),
            "data_seq": (data_seq, "float"),
            "timestamp": (timestamp, "float"),
            "label": (label, "int"),
        }
        writer.write(feature)
    writer.close()
    create_index(tfrecord_path, tfrecord_path + ".index")
    print(f"✅ Saved {N} OOD samples to: {tfrecord_path}")


# ===== Main pipeline =====
def main(opt):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    train_loader, support_1_loader, support_2_loader = set_loader(opt)

    ae = ShrinkageAutoencoder(input_shape=(3, 64, 64), latent_dim=128).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)

    print("Training Shrinkage Autoencoder...")
    ae.train()
    for epoch in range(40):
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            z, x_hat = ae(x_batch)
            loss = shrinkage_loss(x_hat, x_batch, z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss = {loss.item():.4f}")

    print("Extracting latent features from support_1 and support_2...")
    ae.eval()
    Z_support = []
    with torch.no_grad():
        for loader in [support_1_loader, support_2_loader]:
            for x_batch, _ in loader:
                x_batch = x_batch.to(device)
                z, _ = ae(x_batch)
                Z_support.append(z)
    Z_support = torch.cat(Z_support, dim=0)
    Z_support = Z_support[:1000]

    print("Generating OOD samples via SBO from support latent space...")
    Z_ood = soft_brownian_offset(Z_support)
    C = ae.output_shape[0]
    H, W = ae.output_shape[1], ae.output_shape[2]
    with torch.no_grad():
        X_ood = ae.decoder(Z_ood).view(-1, C, H, W)

    print(f"Saving OOD samples to: {opt.ood_save_path}")
    print("X_ood shape:", X_ood.shape)
    write_ood_to_tfrecord(X_ood, opt.ood_save_path)
    print("Done.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--support_1_folder', type=str, required=True)
    parser.add_argument('--support_2_folder', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--ood_save_path', type=str, default='generated_ood.tfrecord')
    opt = parser.parse_args()
    main(opt)
