import os
import random
import torch
import numpy as np

def seed_everything(seed):
    """Set seed for reproducibility across all random number generators"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_video_fps(path):
    """Get the FPS of a video file"""
    import subprocess
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate', '-of',
             'default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, text=True, check=True
        )
        fps_str = result.stdout.strip()
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            return num / den
        return float(fps_str)
    except:
        # Default to 30 fps if detection fails
        return 30.0

def load_video(path, device):
    import torchvision.io as io
    import imageio

    try:
        frames = io.read_video(path, pts_unit='sec')[0]
    except (ImportError, RuntimeError, OSError):
        frames_list = []
        with imageio.get_reader(path) as reader:
            for frame in reader:
                frame_tensor = torch.from_numpy(frame)
                frames_list.append(frame_tensor)
        if not frames_list:
            raise ValueError(f"No frames could be read from {path}")
        frames = torch.stack(frames_list, dim=0)

    frames = frames.permute(0, 3, 1, 2).to(device).float() / 255.0
    frames = frames.mul(2).sub(1)
    return frames.permute(1, 0, 2, 3)


def apply_center_square_mask(frames, fraction):
    if frames.ndim != 4:
        raise ValueError("Expected frames tensor with shape (C, F, H, W)")

    clamped_fraction = max(0.0, min(float(fraction), 1.0))
    if clamped_fraction == 0.0:
        return frames, None, None

    _, _, height, width = frames.shape
    side_length = int(round(min(height, width) * clamped_fraction))
    side_length = max(1, min(side_length, height, width))

    top = (height - side_length) // 2
    left = (width - side_length) // 2
    bottom = top + side_length
    right = left + side_length

    masked_frames = frames.clone()
    masked_frames[:, :, top:bottom, left:right] = -1.0

    pixel_mask = torch.ones_like(frames)
    pixel_mask[:, :, top:bottom, left:right] = 0.0

    return masked_frames, pixel_mask, (top, bottom, left, right)


def create_latent_center_mask(latent_tensor, mask_details):
    if mask_details is None:
        return torch.ones_like(latent_tensor)

    mask_latent = torch.ones_like(latent_tensor)
    _, _, h_tokens, w_tokens = latent_tensor.shape

    height = mask_details["height"]
    width = mask_details["width"]
    top = mask_details["top"]
    bottom = mask_details["bottom"]
    left = mask_details["left"]
    right = mask_details["right"]

    if height <= 0 or width <= 0:
        return mask_latent

    scale_h = h_tokens / height
    scale_w = w_tokens / width

    lat_top = max(0, min(int(top * scale_h), h_tokens - 1))
    lat_bottom = max(lat_top + 1, min(int(bottom * scale_h), h_tokens))
    lat_left = max(0, min(int(left * scale_w), w_tokens - 1))
    lat_right = max(lat_left + 1, min(int(right * scale_w), w_tokens))

    mask_latent[:, :, lat_top:lat_bottom, lat_left:lat_right] = 0.0
    return mask_latent


def build_pixel_mask_from_video(mask_frames, threshold=0.5, invert=False):
    if mask_frames.ndim != 4:
        raise ValueError("Expected mask tensor with shape (C, F, H, W)")

    mask_unit = (mask_frames + 1.0) * 0.5
    if mask_unit.shape[0] > 1:
        mask_unit = mask_unit.mean(dim=0, keepdim=True)

    mask_binary = (mask_unit >= threshold).float()
    if invert:
        mask_binary = 1.0 - mask_binary

    return (1.0 - mask_binary).clamp(0.0, 1.0)


def resize_pixel_mask_to_latent(pixel_mask, latent_tensor):
    if pixel_mask.ndim != 4:
        raise ValueError("Expected pixel mask with shape (1, F, H, W)")

    mask_resized = torch.nn.functional.interpolate(
        pixel_mask.unsqueeze(0),
        size=latent_tensor.shape[1:],
        mode="nearest",
    ).squeeze(0)

    mask_resized = mask_resized.expand(latent_tensor.shape[0], -1, -1, -1).contiguous()
    return mask_resized.to(dtype=latent_tensor.dtype, device=latent_tensor.device)
