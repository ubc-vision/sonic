import torchmetrics
import torch
from PIL import Image
import argparse
from torchvision import transforms
import os
from tqdm import tqdm
import time
import csv

# Extract numeric ID from each filename (e.g. '0004.png' → 4)
def get_id(fname):
    base = os.path.splitext(fname)[0]
    # Remove non-digits, then convert to int (default 0 if empty)
    num = ''.join(filter(str.isdigit, base))
    return int(num) if num else 0

def main(args):
    predicted_folder_path = args.folder_path
    gt_folder_path = args.gt_folder_path

    # check folders exist
    if not os.path.exists(predicted_folder_path):
        raise ValueError(f"Predicted folder path does not exist: {predicted_folder_path}")
    if not os.path.exists(gt_folder_path):
        raise ValueError(f"Ground truth folder path does not exist: {gt_folder_path}")

    metric_results_root_folder = "metrics_results"
    os.makedirs(metric_results_root_folder, exist_ok=True)

    print("start metric calculation")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor()
    ])

    # load gt image from path
    gt_images = []
    gt_files = [f for f in os.listdir(gt_folder_path) if f.lower().endswith(".png") or f.lower().endswith(".jpg")]
    gt_files = sorted(gt_files, key=get_id)
    for filename in gt_files:
        img = Image.open(os.path.join(gt_folder_path, filename)).convert("RGB")
        gt_images.append(transform(img))
    
    # load predicted image from path
    predicted_images = []

    print("Predicted folder path:", predicted_folder_path)

    # Gather and sort all image files
    predicted_files = [f for f in os.listdir(predicted_folder_path) if f.lower().endswith(".png") or f.lower().endswith(".jpg")]
    predicted_files = sorted(predicted_files, key=get_id)

    # Read and transform images
    for filename in predicted_files:
        img_path = os.path.join(predicted_folder_path, filename)
        img = Image.open(img_path).convert("RGB")
        predicted_images.append(transform(img))

    fid_metric = torchmetrics.image.fid.FrechetInceptionDistance(normalize=True).to(device)
    lpips_metric = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=False, reduction="mean"
    ).to(device)
    psnr_list = []
    lpips_list = []
    ssim_list = []

    # Safety check
    assert len(predicted_images) == len(gt_images), \
        f"Different number of images: predicted={len(predicted_images)}, gt={len(gt_images)}"

    # iterate over images
    for pred, gt in tqdm(zip(predicted_images, gt_images), total=len(gt_images), desc="metric computation"):
        # Move tensors to the selected device
        pred = pred.unsqueeze(0)  # add batch dimension -> [1, 3, H, W]
        gt = gt.unsqueeze(0)
        gt = gt.to(device)
        pred = pred.to(device)
        # convert to [-1, 1] range
        pred = pred * 2 - 1
        gt = gt * 2 - 1
        # to range [0,1]
        gt_norm = gt * 0.5 + 0.5
        pred_norm = pred * 0.5 + 0.5
        # compute PSNR
        psnr = torchmetrics.functional.image.peak_signal_noise_ratio(
            pred_norm, gt_norm, data_range=1.0
        )
        psnr_list.append(psnr.cpu()) # Move result to CPU
        # compute LPIPS
        lpips_score = lpips_metric(pred.clip(-1,1), gt.clip(-1,1))
        lpips_list.append(lpips_score.cpu()) # Move result to CPU
        # compute SSIM
        ssim = torchmetrics.functional.image.structural_similarity_index_measure(
            pred_norm, gt_norm, data_range=1.0
        )
        ssim_list.append(ssim.cpu()) # Move result to CPU
        print(f"PSNR: {psnr}, LPIPS: {lpips_score}, SSIM: {ssim}")
        # compute FID
        # Ensure inputs are on the correct device (already handled by moving gt/pred earlier)
        fid_metric.update(gt_norm, real=True)
        fid_metric.update(pred_norm, real=False)
    # compute FID
    fid = fid_metric.compute()
    # compute average metrics (on CPU)
    avg_psnr = torch.mean(torch.stack(psnr_list))
    avg_lpips = torch.mean(torch.stack(lpips_list))
    avg_ssim = torch.mean(torch.stack(ssim_list))
    # compute standard deviation (on CPU)
    std_psnr = torch.std(torch.stack(psnr_list))
    std_lpips = torch.std(torch.stack(lpips_list))
    std_ssim = torch.std(torch.stack(ssim_list))
    print(f"PSNR: {avg_psnr} +/- {std_psnr}")
    print(f"LPIPS: {avg_lpips} +/- {std_lpips}")
    print(f"SSIM: {avg_ssim} +/- {std_ssim}")
    print(f"FID: {fid}") # FID is computed on the selected device, print directly
    # save to prediction folder
    current_time = time.strftime("%Y%m%d-%H%M%S")
    out_file = os.path.join(metric_results_root_folder, f"reconstruction_metrics_{current_time}.txt")
    with open(out_file, "w") as f:
        f.write(f"Processed directory: {predicted_folder_path}\n")
        f.write(f"Resolution: {args.resolution}\n")
        f.write(f"PSNR: {avg_psnr.item()} +/- {std_psnr.item()}\n") # Use .item() for scalar tensors
        f.write(f"LPIPS: {avg_lpips.item()} +/- {std_lpips.item()}\n")
        f.write(f"SSIM: {avg_ssim.item()} +/- {std_ssim.item()}\n")
        f.write(f"FID: {fid.item()}\n") # Use .item() for scalar tensors

    # --- Append metrics to a CSV log ---
    csv_path = os.path.join(metric_results_root_folder, "reconstruction_metrics_log.csv")
    row = [
        predicted_folder_path,
        avg_psnr.item(),
        avg_ssim.item(),
        avg_lpips.item(),
        fid.item(),
    ]

    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if new_file:
            writer.writerow(["folder_path", "PSNR", "SSIM", "LPIPS", "FID"])
        writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Running metrics")
    parser.add_argument("--resolution", type=int, default=1024, help="Resolution for image generation")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the inpainted images folder")
    parser.add_argument("--gt_folder_path", type=str, required=True, help="Path to the ground truth images folder")

    args = parser.parse_args()
    main(args)