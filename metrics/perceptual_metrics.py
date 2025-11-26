import torch
import json
import os
import numpy as np
from PIL import Image
import argparse
import pandas as pd
from torchmetrics.multimodal import CLIPScore
from urllib.request import urlretrieve
import open_clip
import hpsv2
import ImageReward as RM
import time
from tqdm import tqdm
import csv


class MetricsCalculator:
    def __init__(self, device, ckpt_path="data/ckpt") -> None:
        self.device = device
        # CLIP score
        self.clip_metric_calculator = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
        # Aesthetic model
        self.aesthetic_model = torch.nn.Linear(768, 1)
        aesthetic_model_url = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true"
        )
        os.makedirs(ckpt_path, exist_ok=True)
        aesthetic_model_ckpt_path = os.path.join(ckpt_path, "sa_0_4_vit_l_14_linear.pth")
        if not os.path.exists(aesthetic_model_ckpt_path):
            urlretrieve(aesthetic_model_url, aesthetic_model_ckpt_path)
        self.aesthetic_model.load_state_dict(torch.load(aesthetic_model_ckpt_path))
        self.aesthetic_model.eval()
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        # ImageReward model
        self.imagereward_model = RM.load("ImageReward-v1.0")
 

    def calculate_image_reward(self, image, prompt):
        try:
            with torch.no_grad():
                reward = self.imagereward_model.score(prompt, [image])
            return reward
        except Exception as e:
            print(f"Error in calculate_image_reward: {e}")
            return 0.0

    def calculate_hpsv21_score(self, image, prompt):
        try:
            with torch.no_grad():
                result = hpsv2.score(image, prompt, hps_version="v2.1")[0]
            return result.item()
        except Exception as e:
            print(f"Error in calculate_hpsv21_score: {e}")
            return 0.0

    def calculate_aesthetic_score(self, img):
        try:
            image = self.clip_preprocess(img).unsqueeze(0)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                prediction = self.aesthetic_model(image_features)
            return prediction.cpu().item()
        except Exception as e:
            print(f"Error in calculate_aesthetic_score: {e}")
            return 0.0

    def calculate_clip_similarity(self, img, txt):
        try:
            img = np.array(img)
            if len(img.shape) != 3:
                print(f"Warning: Invalid image shape {img.shape}")
                return 0.0

            img_tensor = torch.tensor(img).permute(2, 0, 1).to(self.device)

            with torch.no_grad():
                score = self.clip_metric_calculator(img_tensor, txt)
                score = score.cpu().item()

            return score
        except Exception as e:
            print(f"Error in calculate_clip_similarity: {e}")
            return 0.0
    

def main(args):
    predicted_folder_path = args.folder_path

    # check folder exists
    if not os.path.exists(predicted_folder_path):
        raise ValueError(f"Predicted folder path does not exist: {predicted_folder_path}")

    metric_results_root_folder = "metrics_results"
    os.makedirs(metric_results_root_folder, exist_ok=True)

    # Load prompts from JSON file
    with open(args.prompts_json, "r") as f:
        captions_dict = json.load(f)



    device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluation_df = pd.DataFrame(columns=['Image ID','Image Reward', 'HPS V2.1', 'Aesthetic Score', 'CLIP Similarity'])

    metrics_calculator=MetricsCalculator(device)

    resolution = args.resolution

    inpainted_image_files = [os.path.join(predicted_folder_path, f) for f in os.listdir(predicted_folder_path) if f.lower().endswith('.png') or f.lower().endswith('.jpg')]
    inpainted_image_files.sort()
    inpainted_images = [Image.open(f) for f in inpainted_image_files]

    print(f"Number of inpainted images: {len(inpainted_image_files)}")
    print("Starting evaluation...")

    for index, inpainted_img in tqdm(
            enumerate(inpainted_images),
            total=len(inpainted_images),
            desc="Evaluating metrics"):
        # Get filename without extension to use as key
        filename = os.path.basename(inpainted_image_files[index])
        image_key = os.path.splitext(filename)[0]

        # Get prompt from dict
        if image_key not in captions_dict:
            print(f"Warning: No prompt found for image {image_key}, skipping...")
            continue
        prompt = captions_dict[image_key]

        inpainted_img = inpainted_img.resize((resolution, resolution))

        evaluation_result=[f"{index:05d}"]

        for metric in evaluation_df.columns.values.tolist()[1:]:

            if metric == 'Image Reward':
                metric_result = metrics_calculator.calculate_image_reward(inpainted_img,prompt)
                
            if metric == 'HPS V2.1':
                metric_result = metrics_calculator.calculate_hpsv21_score(inpainted_img,prompt)
            
            if metric == 'Aesthetic Score':
                metric_result = metrics_calculator.calculate_aesthetic_score(inpainted_img)
            
            if metric == 'CLIP Similarity':
                metric_result = metrics_calculator.calculate_clip_similarity(inpainted_img, prompt)

            evaluation_result.append(metric_result)
        
        evaluation_df.loc[len(evaluation_df.index)] = evaluation_result

    print("The averaged evaluation result:")
    averaged_results=evaluation_df.mean(numeric_only=True)
    print(averaged_results)

    # Save results with timestamp
    time_id = time.strftime("%Y%m%d-%H%M%S")
    averaged_results.to_csv(os.path.join(metric_results_root_folder, f"perceptual_metrics_summary_{time_id}.csv"))
    with open(os.path.join(metric_results_root_folder, f"perceptual_metrics_summary_{time_id}.csv"), "a") as f:
        f.write(f"\n{predicted_folder_path}")
    evaluation_df.to_csv(os.path.join(metric_results_root_folder, f"perceptual_metrics_detailed_{time_id}.csv"))

    print(f"Evaluation results saved in {metric_results_root_folder}")

    # --- Append final averaged metrics to a global CSV log ---
    global_csv = os.path.join(metric_results_root_folder, "perceptual_metrics_log.csv")

    # Prepare row
    row = [
        predicted_folder_path,
        averaged_results["Image Reward"],
        averaged_results["HPS V2.1"],
        averaged_results["Aesthetic Score"],
        averaged_results["CLIP Similarity"],
    ]

    # Append to CSV (with header only if file doesn't exist yet)
    write_header = not os.path.exists(global_csv)
    with open(global_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "folder_path",
                "Image Reward",
                "HPS V2.1",
                "Aesthetic Score",
                "CLIP Similarity",
            ])
        writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Running BrushNet metrics evaluation")
    parser.add_argument("--resolution", type=int, default=1024, help="Resolution for image resizing")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing inpainted images")
    parser.add_argument("--prompts_json", type=str, required=True, help="Path to the JSON file containing prompts")
    args = parser.parse_args()
    main(args)