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


def generate_dateset(video_path: str, dst_dir: str, scale: float,
                     dst_width: int, dst_height: int, k: int = 5,
                     is_val=False):
    global detector, crop

    cap = cv2.VideoCapture(video_path)
    num_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        num_frames += 1
    cap.release()

    cap = cv2.VideoCapture(video_path)
    chosen_idxs = random.choices(list(range(num_frames)), k=k)
    current_idx = 0
    vid_name = os.path.basename(video_path).split('.')[0]
    is_real = int(os.path.basename(dst_dir))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in chosen_idxs:
            # Detect and scale image
            bbox = detector.get_bbox(frame)
            if bbox[2] * bbox[3] < 100:
                new_idxs = list(set(range(current_idx, num_frames)).difference(set(chosen_idxs)))
                if not len(new_idxs):
                    break
                rand_idx = random.choice(new_idxs)
                chosen_idxs.append(rand_idx)
                print(f"Empty/tiny face detected, select new idx: {vid_name} with GT label: {is_real}")
            else:
                if not is_val:
                    frame = crop.crop(frame, bbox, scale, dst_width, dst_height, crop=True)

                name = vid_name + '_' + f"{len(os.listdir(dst_dir)) + 1}".zfill(3) + ".png"
                dst_path = os.path.join(dst_dir, name)
                cv2.imwrite(dst_path, frame)
        current_idx += 1
    cap.release()


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
    # rz_height = 512
    # rz_width = 512 // 4 * 3
    rz_height = 384
    rz_width = 384

    scale = 2.
    fake_dir = f"./datasets/train/{scale:.1f}_{rz_height}x{rz_width}/0"
    real_dir = f"./datasets/train/{scale:.1f}_{rz_height}x{rz_width}/1"

    make_if_not_exist(fake_dir)
    make_if_not_exist(real_dir)

    video_root = os.path.join(root, "train", "videos")
    for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
        label = row["liveness_score"]
        fname = row["fname"]

        video_path = os.path.join(video_root, fname)
        generate_dateset(video_path,
                         dst_dir=fake_dir if label == 0 else real_dir,
                         scale=scale,
                         dst_width=rz_width,
                         dst_height=rz_height,
                         k=5)

    # # Set up validation
    # fake_dir = f"./datasets/val/original/0"
    # real_dir = f"./datasets/val/original/1"
    #
    # make_if_not_exist(fake_dir)
    # make_if_not_exist(real_dir)
    #
    # video_root = os.path.join(root, "train", "videos")
    # for i, row in tqdm(val_df.iterrows(), total=len(val_df)):
    #     label = row["liveness_score"]
    #     fname = row["fname"]
    #
    #     video_path = os.path.join(video_root, fname)
    #     generate_dateset(video_path,
    #                      dst_dir=fake_dir if label == 0 else real_dir,
    #                      scale=scale,
    #                      dst_width=rz_width,
    #                      dst_height=rz_height,
    #                      k=3, is_val=True)
