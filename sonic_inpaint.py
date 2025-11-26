import os
import argparse

import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline, StableDiffusion3InpaintPipeline
from utils.model_utils import latent2image, seed_everything, image2latent, image_to_latent_mask
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Spectral Optimization of Noise for Inpainting with Consistency")
    parser.add_argument("--seed", default=200, type=int, help="Random seed for reproducibility")
    parser.add_argument("--CFG_scale", type=float, default=2.0, help="CFG scale for velocity prediction")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (BrushBench, DIV2K, or FFHQ)")
    parser.add_argument("--image_index", type=str, required=True, help="Image index/name (e.g., '000000069' for BrushBench, '00088' for DIV2K/FFHQ)")
    parser.add_argument("--image_path", type=str, default=None, help="Path to the input image (optional, auto-constructed if not provided)")
    parser.add_argument("--mask_path", type=str, default=None, help="Path to the inpainting mask (optional, auto-constructed if not provided)")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for inpainting (optional, read from txt file if not provided)")
    parser.add_argument("--num_iterations", type=int, default=20, help="Number of optimization iterations")
    parser.add_argument("--step_nums", type=int, default=20, help="Number of steps for ODE solver")
    parser.add_argument("--learning_rate", type=float, default=3.0, help="Learning rate for optimization")
    args = parser.parse_args()

    image_ext = "jpg" if args.dataset_name == "BrushBench" else "png"

    if args.image_path is None:
        args.image_path = f"samples/{args.dataset_name}/{args.image_index}.{image_ext}"

    if args.mask_path is None:
        if args.dataset_name == "BrushBench":
            args.mask_path = f"samples/{args.dataset_name}/mask_images/{args.image_index}.png"
        else:
            args.mask_path = f"samples/{args.dataset_name}/{args.image_index}_mask.png"

    if args.prompt is None:
        prompt_file = f"samples/{args.dataset_name}/{args.image_index}_prompt.txt"
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        with open(prompt_file, "r") as f:
            args.prompt = f.read().strip().strip('"')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Image: {args.image_path}")
    print(f"Mask: {args.mask_path}")
    print(f"Prompt: {args.prompt}")

    # pipeline setup
    image_resolution = 1024
    model_path = "stabilityai/stable-diffusion-3.5-medium"
    seed_everything(args.seed)
    torch_dtype = torch.float16
    pipe = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=torch_dtype).to(device)
    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    save_folder = os.path.join("inpaint_results", f"{args.dataset_name}_{image_name}_steps{args.step_nums}_iter{args.num_iterations}")
    os.makedirs(save_folder, exist_ok=True)
    print(f"Results saved to: {save_folder}")
    epsilon_path = os.path.join(save_folder, "epsilon")
    x_0_hat_path = os.path.join(save_folder, "x_0_hat")
    os.makedirs(epsilon_path, exist_ok=True)
    os.makedirs(x_0_hat_path, exist_ok=True)

    # load GT image and encode as latent
    image = Image.open(args.image_path).convert("RGB")
    assert image.size[0] == image.size[1], "Input images must be square."
    image = image.resize((image_resolution, image_resolution))
    original_target_image = image2latent(pipe, image, device, torch_dtype).to(torch.float32)

    # Load mask based on dataset
    if args.dataset_name == "BrushBench":
        mask_image = Image.open(args.mask_path)
        mask = image_to_latent_mask(mask_image, original_target_image.shape, device, torch.float32)
    elif args.dataset_name == "FFHQ":
        mask = torch.load("setup/FFHQ_mask.pt").to(device)
    elif args.dataset_name == "DIV2K":
        mask = torch.load("setup/DIV2K_mask.pt").to(device)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    # save pixel-space images for reference
    target_image_PIL = latent2image(pipe, original_target_image, device, torch_dtype)
    target_image_PIL.save(os.path.join(save_folder, "target_image_before_mask.png"))
    target_image = original_target_image * mask
    target_image_PIL = latent2image(pipe, target_image, device, torch_dtype)
    target_image_PIL.save(os.path.join(save_folder, "target_image.png"))

    # load initial noise
    epsilon_original = torch.load("setup/initial_noise_resolution_1024.pt").to(torch.float32)
    epsilon = epsilon_original.clone().float()

    # optimize spectral noise with Adam
    epsilon_freq = torch.fft.rfftn(epsilon, s=epsilon.shape[-2:], dim=(-2, -1))
    epsilon_optim_target = torch.nn.Parameter(epsilon_freq.detach().clone())
    optimizer = torch.optim.Adam([epsilon_optim_target], lr=args.learning_rate)

    # optimization loop
    for i in range(args.num_iterations):
        epsilon_spatial = torch.fft.irfftn(epsilon_optim_target, s=epsilon.shape[-2:], dim=(-2, -1))
        # block gradients to masked region
        epsilon_combined = epsilon_spatial * mask + epsilon_original * (1 - mask)
        torch.save(epsilon_combined, os.path.join(epsilon_path, f"{i}.pt"))

        # obtain ε* without tracking gradients
        with torch.no_grad():
            model_final_output = pipe(
                args.prompt,
                num_inference_steps=args.step_nums,
                guidance_scale=args.CFG_scale,
                latents=epsilon_combined.to(torch_dtype),
                height=image_resolution,
                width=image_resolution,
                output_type='latent',
            ).images.to(torch.float32)
            added_noise = epsilon_combined - model_final_output

        # compute x_0_hat with gradients
        x_0_hat = epsilon_combined - added_noise
        x_0_hat_img = latent2image(pipe, x_0_hat, device, torch_dtype)
        x_0_hat_img.save(os.path.join(x_0_hat_path, f"{i}.png"))

        # masked loss and optimizer step
        loss = F.mse_loss(target_image, x_0_hat, reduction="none")
        loss = (loss * mask).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epsilon_spatial = torch.fft.irfftn(epsilon_optim_target, s=epsilon.shape[-2:], dim=(-2, -1))
    epsilon_spatial_final = epsilon_spatial * mask + epsilon_original * (1 - mask)
    torch.save(epsilon_spatial_final, os.path.join(epsilon_path, "final_epsilon.pt"))

    # Appending BLD-SD3.5 Inpainting Pipeline for optimized epsilon
    del pipe
    torch.cuda.empty_cache()
    seed_everything(args.seed)

    inpaint_pipe = StableDiffusion3InpaintPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    mask_invert = (1 - mask).to(torch.float16)
    image = inpaint_pipe(
        latents=epsilon_spatial_final.to(torch.float16),
        prompt=args.prompt,
        image=target_image,
        mask_image=mask_invert,
        guidance_scale=args.CFG_scale,
        strength=1.0,
        num_inference_steps=20,
        width=image_resolution,
        height=image_resolution
    ).images[0]
    inpainted_output_folder = os.path.join(save_folder, "inpainted_output")
    os.makedirs(inpainted_output_folder, exist_ok=True)
    image.save(os.path.join(inpainted_output_folder, "inpainted.png"))
