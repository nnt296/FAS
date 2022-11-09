import csv
import os
import warnings
from typing import List, Union, Any

import cv2
import numpy as np
from tqdm import tqdm

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')


def sample_video(video_path: str, num_sample: int = 5) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    num_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        num_frames += 1
    cap.release()

    cap = cv2.VideoCapture(video_path)
    chosen_frames = []

    step = num_frames // (num_sample + 1)
    current = 0
    index = step
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current == index and len(chosen_frames) < num_sample:
            chosen_frames.append(frame)
            index += step
        current += 1
    cap.release()

    assert len(chosen_frames) == num_sample, f"Expect: {num_sample} got {len(chosen_frames)}"
    return chosen_frames


def predict_image(image, model_dir: str = "resources/my_spoof_models"):
    model = AntiSpoofPredict(device_id=0)
    image_cropper = CropImage()

    image_bbox = model.get_bbox(image)
    if image_bbox[2] * image_bbox[3] < 100:
        # Case no face detected -> Mark as spoofing
        return 0, np.array([1, 0])

    prediction = np.zeros((1, 2))
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {"org_img": image, "bbox": image_bbox, "scale": scale,
                 "out_w": w_input, "out_h": h_input, "crop": True}
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        prediction += model.predict(img, os.path.join(model_dir, model_name))

    pred = np.argmax(prediction)
    # score = prediction[0][pred] / 2
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
    predictions = np.array(predictions)
    scores = np.array(scores)
    # Old rule
    # if sum(predictions) / len(predictions) < 4 / num_sample:
    #     return False
    # else:
    #     return True
    final_score = np.zeros((2, ))
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
            writer.writerow([video_name, f"{final_score[is_real]:.5f}"])


if __name__ == "__main__":
    # vid_path = "/home/local/ZaChallenge/2022/liveliness/public/videos/0.mp4"
    # # vid_path = "/home/local/ZaChallenge/2022/liveliness/train/videos/1446.mp4"
    # res, _ = infer_video(vid_path)
    # print(f"{os.path.basename(vid_path)} is {'real' if res == 1 else 'fake'}")

    create_submission("/mnt/ssd_1T/zalo_ai_22/liveness_detection/public/videos")
