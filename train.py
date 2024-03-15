from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch
from dvae import BetaVAE_H


vae = BetaVAE_H(c_dim=20, nc=3, infodistil_mode=True)
vae_ckpt = torch.load("./vae_chkp/celeba_vae")['model_states']['net']
vae.load_state_dict(vae_ckpt)
vae = vae.cuda()

model = Unet(
    dim = 128,
    c_dim=20,
    dim_mults = (1, 2, 2, 2)
).cuda()

diffusion = GaussianDiffusion(
    model,
    vae,
    c_dim=20,
    image_size = 64,
    timesteps = 1000,           # number of steps
    loss_type = 'l2'            # L1 or L2
).cuda()



trainer = Trainer(
    diffusion,
    './celeba',
    c_dim=20,
    train_batch_size = 128,
    train_lr = 1e-4,
    save_and_sample_every = 5000,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True                        # turn on mixed precision
)
trainer.load(2)
trainer.train()
