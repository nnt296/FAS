import os
import cv2
import random
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utility import make_if_not_exist
from src.anti_spoof_predict import Detection
from src.generate_patches import CropImage

detector = Detection()
crop = CropImage()


def sample_video(video_path: str, num_sample: int = 5):
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

    num_sample = min(max(num_frames - 3, 0), num_sample)

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

    if len(chosen_frames) != num_sample:
        print(f"Error {video_path}")
        print(f"Expect: {num_sample} got {len(chosen_frames)}")

    return chosen_frames


def crop_and_expand(frames, scale, dst_width, dst_height):
    new_frames = []
    for frame in frames:
        bbox = detector.get_bbox(frame)
        if bbox[2] * bbox[3] < 100:
            continue
        else:
            frame = crop.crop(frame, bbox, scale, dst_width, dst_height, crop=True)
            new_frames.append(frame)
    return new_frames


def prepare_data(video_dir, scale, dst_width, dst_height, mode="train", num_sample=30):
    if scale != "origin":
        fake_dir = f"./datasets/{mode}/{scale:.1f}_{dst_height}x{dst_width}/0"
        real_dir = f"./datasets/{mode}/{scale:.1f}_{dst_height}x{dst_width}/1"
    else:
        fake_dir = f"./datasets/{mode}/original_{dst_height}x{dst_width}/0"
        real_dir = f"./datasets/{mode}/original_{dst_height}x{dst_width}/1"

    make_if_not_exist(fake_dir)
    make_if_not_exist(real_dir)

    for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
        label = row["liveness_score"]
        fname = row["fname"]

        video_path = os.path.join(video_dir, fname)

        frames = sample_video(video_path, num_sample=num_sample)
        frames = crop_and_expand(frames, scale, dst_width, dst_height)

        dst_dir = fake_dir if label == 0 else real_dir
        vid_name = fname.split('.')[0]
        for j, frame in enumerate(frames):
            name = vid_name + '_' + f"{j}".zfill(3) + ".png"
            dst_path = os.path.join(dst_dir, name)
            frame = cv2.resize(frame, (dst_width, dst_height))
            cv2.imwrite(dst_path, frame)


if __name__ == '__main__':
    root = "/mnt/ssd_1T/zalo_ai_22/liveness_detection"

    label_df = pd.read_csv(os.path.join(root, "train", "label.csv"))
    fake_df = label_df[label_df["liveness_score"] == 0]
    real_df = label_df[label_df["liveness_score"] == 1]

    train_fake, val_fake = train_test_split(
        fake_df, test_size=0.1, random_state=25, shuffle=True)
    train_real, val_real = train_test_split(
        real_df, test_size=0.1, random_state=25, shuffle=True)

    train_df = pd.concat([train_fake, train_real])
    val_df = pd.concat([val_fake, val_real])

    print(f"Len train: {len(train_df)}")
    print(f"Len test: {len(val_df)}")

    # ratio 512x384
    ls_heights = [384, 384, 384]
    ls_widths = [288, 384, 384]
    ls_scales = ["original", "1.4", "2.2"]

    for _scale, _rz_width, _rz_height in zip(ls_scales, ls_widths, ls_heights):
        video_root = os.path.join(root, "train", "videos")
        prepare_data(video_root, _scale, _rz_width, _rz_height, mode="train", num_sample=30)

        # Set up validation
        video_root = os.path.join(root, "train", "videos")
        prepare_data(video_root, _scale, _rz_width, _rz_height, mode="val", num_sample=20)
