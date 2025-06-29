import torch
import torch.nn as nn
import torch.autograd as autograd

class Generator(nn.Module):
    def __init__(self, latent_dim=128, out_dim=8256, hidden_dim=1024):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim=8256, hidden_dim=1024):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

def gradient_penalty(critic, real_feats, fake_feats, device, lambda_gp=10.0):
    batch_size = real_feats.size(0)
    alpha = torch.rand(batch_size, 1).to(device)
    alpha = alpha.expand_as(real_feats)

    interpolated = alpha * real_feats + (1 - alpha) * fake_feats
    interpolated = interpolated.to(device).requires_grad_(True)

    # Critic score trên sample interpolated
    d_interpolated = critic(interpolated)

    # Tính gradient của output Critic theo input
    gradients = autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)

    gp = lambda_gp * ((grad_norm - 1) ** 2).mean()
    return gp

def loss_generator_wgan_gp(d_fake):
    # d_fake: output của Critic với input là fake feature
    return -d_fake.mean()

def loss_discriminator_wgan_gp(d_real, d_fake):
    return d_fake.mean() - d_real.mean()  # Note: không cần log()
