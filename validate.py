import os
import time
import argparse
import warnings

import cv2
import numpy as np
from sklearn import metrics

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')


def test_folder(image_dir: str, model_dir: str,
                device_id,
                save_dir: str = "runs",
                save_results: bool = True):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    if save_results:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    ground_truths = []
    predictions = []
    for label in sorted(os.listdir(image_dir)):
        gt = 1 if int(label) == 1 else 0
        ground_truths.append(gt)

        label_dir = os.path.join(image_dir, label)
        for image_name in sorted(os.listdir(label_dir)):
            image_path = os.path.join(label_dir, image_name)
            image = cv2.imread(image_path)
            image_bbox = model_test.get_bbox(image)
            prediction = np.zeros((1, 2))
            test_speed = 0

            # sum the prediction from single model's result
            for model_name in os.listdir(model_dir):
                h_input, w_input, model_type, scale = parse_model_name(model_name)
                param = {"org_img": image, "bbox": image_bbox, "scale": scale,
                         "out_w": w_input, "out_h": h_input, "crop": True}
                if scale is None:
                    param["crop"] = False
                img = image_cropper.crop(**param)
                start = time.time()
                prediction += model_test.predict(img, os.path.join(model_dir, model_name))
                test_speed += time.time() - start

            pred = np.argmax(prediction)
            score = prediction[0][pred] / 2

            # draw result of prediction
            if pred == 1:
                print("Image '{}' \tis Real Face. \tScore: {:.2f}.".format(image_name, score))
                result_text = "RealFace Score: {:.2f}".format(score)
                color = (255, 0, 0)
            else:
                print("Image '{}' \tis Fake Face. \tScore: {:.2f}.".format(image_name, score))
                result_text = "FakeFace Score: {:.2f}".format(score)
                color = (0, 0, 255)
            print("Prediction cost {:.2f} s".format(test_speed))

            predictions.append(1) if pred == 1 else 0
            if save_results:
                cv2.rectangle(image, (image_bbox[0], image_bbox[1]),
                              (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]), color, 2)
                cv2.putText(image, result_text, (image_bbox[0], image_bbox[1] - 5),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5 * image.shape[0] / 1024, color)

                result_image_name = f"gt_{gt}_pred_{pred}-" + image_name
                cv2.imwrite(os.path.join(save_dir, result_image_name), image)

    confusion_matrix = metrics.confusion_matrix(y_true=ground_truths, y_pred=predictions)
    print(confusion_matrix)


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/my_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--image_folder",  # There are sub label 0 and 1 within image_folder
        type=str,
        help="image used to test")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs/0/",
        help="output directory to store inference results")

    args = parser.parse_args()
    test_folder(args.image_folder, args.model_dir, args.device_id, save_dir=args.out_dir)
