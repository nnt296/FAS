import csv
import os
from typing import Dict

import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F
import pytorch_lightning as pl

from my_config import get_default_config
from my_trainer import Module
from src.anti_spoof_predict import Detection
from src.generate_patches import CropImage
from src.data_io import transform as trans
from submission import sample_video

scale2params: Dict = {}
scale2model: Dict = {}
detector = Detection()
image_cropper = CropImage()


def load_models(model_dir: str, device):
    global scale2model, scale2params

    for filename in os.listdir(model_dir):
        filepath = os.path.join(model_dir, filename)
        if not os.path.isfile(filepath):
            continue

        scale = filename.split('_')[0]
        if scale != "original":
            scale = float(scale)
        input_size = filename.split('_')[-1].split('.')[0].split('x')
        height, width = list(map(int, input_size))

        scale2params[scale] = (height, width)
        conf = get_default_config()
        conf.set_hw(height=height, width=width)

        model = Module.load_from_checkpoint(filepath, map_location=device, conf=conf)
        scale2model[scale] = model

    return scale2model, scale2params


def predict_single_model(cropped_image, model: pl.LightningModule):
    # Preprocess
    test_transform = trans.Compose([
        trans.ToTensor(),
    ])
    image = test_transform(cropped_image)
    image = image.unsqueeze(0).to(model.device)

    # Infer
    model.eval()
    with torch.no_grad():
        result = model.forward(image)
        result = F.softmax(result).cpu().numpy()
    return result


def predict_image(image):
    global scale2model, scale2params, detector, image_cropper

    image_bbox = detector.get_bbox(image)
    if image_bbox[2] * image_bbox[3] < 100:
        # Case no face detected -> Mark as spoofing
        return 0, np.array([1, 0])

    prediction = np.zeros((1, 2))
    # sum the prediction from single model's result
    for scale, model in scale2model.items():
        h_input, w_input = scale2params[scale]

        param = {"org_img": image, "bbox": image_bbox, "scale": scale,
                 "out_w": w_input, "out_h": h_input, "crop": True}

        if scale is None or isinstance(scale, str):
            param["crop"] = False

        img = image_cropper.crop(**param)

        prediction += predict_single_model(img, model)

    pred = np.argmax(prediction)
    score = prediction[0] / np.sum(prediction[0])

    return int(pred), score


def infer_video(video_path: str):
    num_sample = 5

    chosen_frames = sample_video(video_path, num_sample=num_sample)

    predictions = []
    scores = []
    for frame in chosen_frames:
        pred, score = predict_image(frame)
        predictions.append(pred)
        scores.append(score)

    # print(predictions)
    scores = np.array(scores)
    final_score = np.zeros((2,))
    for score in scores:
        final_score += score

    final_score = final_score / np.sum(final_score)
    label = int(np.argmax(final_score))
    # print(scores)
    # print(final_score)
    # print(label)
    return label, final_score


def create_submission(video_dir: str):
    with open("predict.csv", "w") as fo:
        writer = csv.writer(fo)
        writer.writerow(["fname", "liveness_score"])

        names = os.listdir(video_dir)
        names = sorted(names, key=lambda x: int(x.split('.')[0]))
        for video_name in tqdm(names):
            video_path = os.path.join(video_dir, video_name)
            is_real, final_score = infer_video(video_path)
            # writer.writerow([video_name, is_real])
            writer.writerow([video_name, f"{final_score[1]:.5f}"])


if __name__ == "__main__":
    # vid_path = "/mnt/ZAChallenge/liveliness-detection/data/public_test/public/videos/0.mp4"
    # vid_path = "/mnt/ZAChallenge/liveliness-detection/data/public_test/public/videos/21.mp4"

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_models(model_dir="runs/lightning_logs", device=dev)

    # res, _ = infer_video(vid_path)
    # print(f"{os.path.basename(vid_path)} is {'real' if res == 1 else 'fake'}")

    create_submission("/mnt/ssd_1T/zalo_ai_22/liveness_detection/public_test_2/videos")
