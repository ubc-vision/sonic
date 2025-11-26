import os
import random 
import torch
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

@torch.no_grad()
def image2latent(pipe, image, device, dtype):
    '''image: PIL.Image'''
    image = np.array(image)
    image = torch.from_numpy(image).to(dtype) / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0).to(device)
    # input image density range [-1, 1]
    latents = pipe.vae.encode(image)['latent_dist'].mean
    if pipe.vae.config.shift_factor is not None:
        latents = latents * pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    else:
        latents = latents * pipe.vae.config.scaling_factor
    if hasattr(pipe, '_pack_latents'):
        latents = pipe._pack_latents(
            latents,
            1, 
            pipe.transformer.config.in_channels // 4,
            (int(image.shape[-2]) // pipe.vae_scale_factor),
            (int(image.shape[-1]) // pipe.vae_scale_factor),
        )
    return latents

@torch.no_grad()
def latent2image(pipe, latents, device, dtype, custom_shape=None):
    '''return: PIL.Image'''
    if hasattr(pipe, '_unpack_latents'):
        # default square
        if custom_shape is None:
          latents = pipe._unpack_latents(
              latents, 
              int((latents.shape[1] ** 0.5) * pipe.vae_scale_factor) * 2,
              int((latents.shape[1] ** 0.5) * pipe.vae_scale_factor) * 2,
              pipe.vae_scale_factor,
          )
        else:
          latents = pipe._unpack_latents(
              latents, 
              custom_shape[0], custom_shape[1],
              pipe.vae_scale_factor,
          )
        
    latents = latents.to(device).to(dtype)
    if pipe.vae.config.shift_factor is not None:
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    else:
        latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type='pil')[0]
    return image

@torch.no_grad()
def image_to_latent_mask(mask_image, latent_shape, device, dtype):
    '''mask_image: PIL.Image, latent_shape: (B, C, H, W)'''
    channels = latent_shape[1]
    spatial_size = latent_shape[2]
    mask = np.array(mask_image.resize((spatial_size, spatial_size)).convert("L"))
    mask = torch.from_numpy(mask).to(dtype) / 255.0
    mask = 1.0 - mask
    mask = mask.unsqueeze(0).unsqueeze(0).to(device)
    mask = mask.repeat(1, channels, 1, 1)
    return mask