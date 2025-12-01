import os
import sys
import argparse
import math
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add Wan2.1 module to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WAN_DIR = os.path.join(SCRIPT_DIR, 'Wan2.1')
sys.path.insert(0, WAN_DIR)

from wan.configs import WAN_CONFIGS
from wan.text2video import WanT2V
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import cache_video

# Import local utility functions
from utils.utils import (
    seed_everything,
    get_video_fps,
    load_video,
    apply_center_square_mask,
    create_latent_center_mask,
    build_pixel_mask_from_video,
    resize_pixel_mask_to_latent
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Spectral Optimization of Noise for Video Inpainting with Consistency")
    parser.add_argument("--target_video", type=str, required=True, help="Path to the target video")
    parser.add_argument("--mask", type=str, default=None, help="Path to the mask video (optional, if not provided will auto-mask center)")
    parser.add_argument("--mask_latent_file", type=str, default=None, help="Path to a saved latent mask (.pt file); will be used directly if provided")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt for video generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt (optional)")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to model checkpoint directory")
    parser.add_argument("--task", type=str, default="t2v-1.3B", help="Model task (default: t2v-1.3B)")
    parser.add_argument("--size", type=str, default="832*480", help="Video size (default: 832*480)")
    parser.add_argument("--seed", type=int, default=200, help="Random seed (default: 200)")
    parser.add_argument("--num_iterations", type=int, default=50, help="Number of optimization iterations")
    parser.add_argument("--step_nums", type=int, default=20, help="Number of steps for ODE solver")
    parser.add_argument("--sample_shift", type=float, default=8.0, help="Sampling shift parameter (default: 8.0)")
    parser.add_argument("--CFG_scale", type=float, default=2.0, help="CFG scale for velocity prediction")
    parser.add_argument("--learning_rate", type=float, default=25.0, help="Learning rate for optimization")
    parser.add_argument("--center_mask_fraction", type=float, default=0.4, help="Fraction for auto center mask when --mask is not provided (default: 0.5)")
    parser.add_argument("--save_every", type=int, default=5, help="Save frequency in iterations (default: 5)")
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device ID (default: 0)")
    parser.add_argument("--out_dir", type=str, default="video_results", help="Output directory (default: video_results)")
    args = parser.parse_args()

    # Set seed for reproducibility
    seed_everything(args.seed)

    device = torch.device(f"cuda:{args.device_id}")
    print(f"Using device: {device}")
    print(f"Target video: {args.target_video}")
    if args.mask_latent_file:
        print(f"Mask: Latent mask file ({args.mask_latent_file})")
    elif args.mask:
        print(f"Mask: Video mask ({args.mask})")
    else:
        print(f"Mask: Auto center mask")
    print(f"Prompt: {args.prompt}")
    print(f"Seed: {args.seed}")

    # pipeline setup
    cfg = WAN_CONFIGS[args.task]
    pipeline = WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device_id
    )
    pipeline.model.eval().requires_grad_(False)
    pipeline.text_encoder.model.to(pipeline.device)

    video_name = os.path.splitext(os.path.basename(args.target_video))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join(args.out_dir, f"{timestamp}_{video_name}_steps{args.step_nums}_iter{args.num_iterations}")
    os.makedirs(save_folder, exist_ok=True)
    print(f"Results saved to: {save_folder}")
    epsilon_path = os.path.join(save_folder, "epsilon")
    x_0_hat_path = os.path.join(save_folder, "x_0_hat")
    os.makedirs(epsilon_path, exist_ok=True)
    os.makedirs(x_0_hat_path, exist_ok=True)

    # load target video and get FPS
    target_frames = load_video(args.target_video, device)
    video_fps = get_video_fps(args.target_video)

    # handle masking
    masked_video_to_save = None
    mask_details = None
    pixel_mask_override = None

    if args.mask:
        mask_frames = load_video(args.mask, device)
        if mask_frames.shape != target_frames.shape:
            raise ValueError("Mask video must have the same dimensions as target video")

        pixel_mask_override = build_pixel_mask_from_video(mask_frames).to(
            device=target_frames.device, dtype=target_frames.dtype
        )
        masked_frames = target_frames * pixel_mask_override + (-1.0) * (1.0 - pixel_mask_override)
        masked_video_to_save = masked_frames.detach().cpu()

        masked_fraction = float((1.0 - pixel_mask_override).mean().item())
        print(f"Mask loaded: {masked_fraction:.2%} of pixels masked")
    else:
        masked_frames, pixel_mask, mask_bbox = apply_center_square_mask(
            target_frames, args.center_mask_fraction
        )
        if pixel_mask is not None:
            masked_video_to_save = masked_frames.detach().cpu()
            _, _, height, width = target_frames.shape
            top, bottom, left, right = mask_bbox
            side = bottom - top
            mask_details = {
                "top": int(top),
                "bottom": int(bottom),
                "left": int(left),
                "right": int(right),
                "side": int(side),
                "height": int(height),
                "width": int(width),
            }
            print(f"Auto center mask applied: {side}px square")

    # encode target video as latent
    original_target_latent = pipeline.vae.encode([target_frames])[0]

    # create mask in latent space
    if args.mask_latent_file:
        mask = torch.load(args.mask_latent_file, map_location=device).to(torch.float32)
        if mask.shape != original_target_latent.shape:
            raise ValueError(
                f"Mask latent shape {tuple(mask.shape)} does not match target latent shape {tuple(original_target_latent.shape)}"
            )
        print(f"Loaded latent mask from {args.mask_latent_file}")
    elif pixel_mask_override is not None:
        mask = resize_pixel_mask_to_latent(pixel_mask_override, original_target_latent)
    elif mask_details is not None:
        mask = create_latent_center_mask(original_target_latent, mask_details)
    else:
        mask = torch.ones_like(original_target_latent)

    torch.save(mask.cpu(), os.path.join(save_folder, "mask_latent.pt"))

    # save masked video for reference
    if masked_video_to_save is not None:
        masked_video_path = os.path.join(save_folder, f"{video_name}_masked.mp4")
        cache_video(
            masked_video_to_save.unsqueeze(0),
            save_file=masked_video_path,
            fps=video_fps,
            nrow=1,
        )
        print(f"Masked video saved to {masked_video_path}")

    target_latent = original_target_latent * mask

    # generate initial noise
    rand_generator = torch.Generator(device=original_target_latent.device)
    rand_generator.manual_seed(args.seed)
    epsilon_original = torch.randn(
        original_target_latent.shape,
        dtype=original_target_latent.dtype,
        device=original_target_latent.device,
        generator=rand_generator,
    )
    epsilon = epsilon_original.clone()

    # optimize spectral noise with Adam
    epsilon_freq = torch.fft.rfftn(epsilon, s=epsilon.shape[-3:], dim=(-3, -2, -1))
    epsilon_optim_target = torch.nn.Parameter(epsilon_freq.detach().clone(), requires_grad=True)
    optimizer = torch.optim.Adam([epsilon_optim_target], lr=args.learning_rate)

    # prepare text embeddings
    prompt_text = args.prompt or ""
    negative_text = args.negative_prompt or pipeline.sample_neg_prompt

    pipeline.text_encoder.model.to(pipeline.device)
    prompt_embeds = pipeline.text_encoder([prompt_text], pipeline.device)
    negative_embeds = pipeline.text_encoder([negative_text], pipeline.device)

    model_dtype = next(pipeline.model.parameters()).dtype
    prompt_embeds = [t.to(device=pipeline.device, dtype=model_dtype) for t in prompt_embeds]
    negative_embeds = [t.to(device=pipeline.device, dtype=model_dtype) for t in negative_embeds]

    c, t_tokens, h_tokens, w_tokens = original_target_latent.shape
    patch_t, patch_h, patch_w = pipeline.patch_size
    seq_len = math.ceil((h_tokens * w_tokens) / (patch_h * patch_w) * t_tokens / pipeline.sp_size) * pipeline.sp_size

    prompt_ctx = {'context': prompt_embeds, 'seq_len': seq_len}
    negative_ctx = {'context': negative_embeds, 'seq_len': seq_len}

    start_time = time.time()

    # optimization loop
    for i in range(args.num_iterations):
        step_start = time.time()

        epsilon_spatial = torch.fft.irfftn(epsilon_optim_target, s=epsilon.shape[-3:], dim=(-3, -2, -1))
        # block gradients to masked region
        epsilon_combined = epsilon_spatial * mask + epsilon_original * (1 - mask)
        torch.save(epsilon_combined, os.path.join(epsilon_path, f"{i}.pt"))

        # obtain x_0_hat without tracking gradients
        with torch.no_grad():
            latents = epsilon_combined.detach().to(device=pipeline.device, dtype=model_dtype)

            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=pipeline.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            scheduler.set_timesteps(args.step_nums, device=pipeline.device, shift=args.sample_shift)

            # Progress bar for diffusion steps
            step_pbar = tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps),
                           desc=f"Iter {i}/{args.num_iterations} - Diffusion steps",
                           unit="step")

            for step_idx, t in step_pbar:
                latent_input = [latents]
                timestep = torch.tensor([t], device=pipeline.device, dtype=latents.dtype)

                noise_pred_cond = pipeline.model(latent_input, t=timestep, **prompt_ctx)[0]
                noise_pred_uncond = pipeline.model(latent_input, t=timestep, **negative_ctx)[0]
                noise_pred = noise_pred_uncond + args.CFG_scale * (noise_pred_cond - noise_pred_uncond)

                latents = scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents.unsqueeze(0),
                    return_dict=False,
                )[0].squeeze(0)

            model_final_output = latents
            added_noise = epsilon_combined - model_final_output

        # compute x_0_hat with gradients
        x_0_hat = epsilon_combined - added_noise
        x_0_hat = x_0_hat.to(dtype=epsilon_spatial.dtype)

        if i % args.save_every == 0:
            clean_latent = x_0_hat.detach().to(pipeline.vae.dtype)
            video_preview = pipeline.vae.decode([clean_latent])[0]
            cache_video(
                video_preview.unsqueeze(0).cpu(),
                save_file=os.path.join(x_0_hat_path, f"{i}.mp4"),
                fps=video_fps,
                nrow=1,
            )

        # masked loss and optimizer step
        loss = F.mse_loss(target_latent, x_0_hat, reduction="none")
        loss = (loss * mask).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_time = time.time() - step_start
        cumulative_time = time.time() - start_time

        # Print summary after progress bar
        print(f"  → Loss: {loss.item():.6f} | Step time: {step_time:.2f}s")

        torch.cuda.empty_cache()

    epsilon_spatial = torch.fft.irfftn(epsilon_optim_target, s=epsilon.shape[-3:], dim=(-3, -2, -1))
    epsilon_spatial_final = epsilon_spatial * mask + epsilon_original * (1 - mask)
    torch.save(epsilon_spatial_final, os.path.join(epsilon_path, "final_epsilon.pt"))

    print(f"\nOptimization complete! Total time: {time.time() - start_time:.2f}s")
    print(f"Results saved to: {save_folder}")
