import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNoisePredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, dim)
        )

    def forward(self, x, t):
        t = t.float().unsqueeze(1) / 30  # normalize timestep
        x = torch.cat([x, t], dim=1)
        return self.net(x)


class GaussianDiffusion:
    def __init__(self, model, timesteps=30):
        self.model = model
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-5, 1e-3, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hats[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hats[t])[:, None]
        return sqrt_alpha_hat * x_start + sqrt_one_minus_alpha_hat * noise

    def p_losses(self, x_start, t, noise=None, cond=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        if cond is not None:
            x_input = torch.cat([x_noisy, cond], dim=1)
        else:
            x_input = x_noisy

        predicted_noise = self.model(x_input, t)
        return F.mse_loss(predicted_noise, noise)

    def sample(self, shape, cond=None, real_embeddings=None, noise_scale=0.1):
        """
        If real_embeddings is provided, use them + mild noise instead of starting from random noise.
        """
        device = next(self.model.parameters()).device

        if real_embeddings is not None:
            x = real_embeddings.to(device)
            noise = torch.randn_like(x) * noise_scale
            x = x + noise
        else:
            x = torch.randn(shape).to(device)

        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((x.shape[0],), t, dtype=torch.long).to(device)
            if cond is not None:
                x_input = torch.cat([x, cond], dim=1)
            else:
                x_input = x

            noise_pred = self.model(x_input, t_tensor)
            beta_t = self.betas[t]
            x = (1 / torch.sqrt(1 - beta_t)) * (x - beta_t * noise_pred)
            if t > 0:
                x += torch.sqrt(beta_t) * torch.randn_like(x)

        return x
