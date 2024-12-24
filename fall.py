import os
import time
import shutil
from sklearn.model_selection import train_test_split
from src.pipeline.fall_detect import FallDetector
import pandas as pd

# Config function for the fall detector
def _fall_detect_config():
    _dir = os.path.dirname(os.path.abspath(__file__))
    _good_tflite_model = os.path.join(
        _dir,
        'ai_models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
    )
    _good_edgetpu_model = os.path.join(
        _dir,
        'ai_models/posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite'
    )
    _good_labels = 'ai_models/pose_labels.txt'
    config = {
        'model': {
            'tflite': _good_tflite_model,
            'edgetpu': _good_edgetpu_model,
        },
        'labels': _good_labels,
        'top_k': 3,
        'confidence_threshold': 0.6,
        'model_name': 'mobilenet'
    }
    return config

# Dataset preprocessing
def preprocess_dataset(dataset_dir, output_dir):
    categories = ['fall', 'non_fall']
    train_dir = os.path.join(output_dir, 'train')
    validation_dir = os.path.join(output_dir, 'val')

    # Create directories for train and validation sets
    for category in categories:
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(validation_dir, category), exist_ok=True)

    # Process each subject in the dataset
    for subject in os.listdir(dataset_dir):
        subject_dir = os.path.join(dataset_dir, subject)
        if not os.path.isdir(subject_dir):
            continue

        for category in categories:
            category_dir = os.path.join(subject_dir, category)
            if not os.path.exists(category_dir):
                continue

            for sub_category in os.listdir(category_dir):
                sub_category_dir = os.path.join(category_dir, sub_category)
                if not os.path.isdir(sub_category_dir):
                    continue

                images = [
                    os.path.join(sub_category_dir, f) for f in os.listdir(sub_category_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]

                if len(images) == 0:
                    print(f"No images found in {sub_category_dir}")
                    continue

                train_images, val_images = train_test_split(images, test_size=0.4, random_state=42)

                for img_path in train_images:
                    dst = os.path.join(train_dir, category, f"{subject}_{sub_category}_{os.path.basename(img_path)}")
                    shutil.copy(img_path, dst)

                for img_path in val_images:
                    dst = os.path.join(validation_dir, category, f"{subject}_{sub_category}_{os.path.basename(img_path)}")
                    shutil.copy(img_path, dst)

    print("Dataset preprocessing complete.")

# Fall prediction function
def Fall_prediction(img_1, img_2, img_3=None):
    config = _fall_detect_config()
    result = None
    fall_detector = FallDetector(**config)

    def process_response(response):
        nonlocal result
        for res in response:
            result = res['inference_result']

    process_response(fall_detector.process_sample(image=img_1))
    time.sleep(fall_detector.min_time_between_frames)
    process_response(fall_detector.process_sample(image=img_2))

    if len(result) == 1:
        category = result[0]['label']
        confidence = result[0]['confidence']
        angle = result[0]['leaning_angle']
        keypoint_corr = result[0]['keypoint_corr']

        dict_res = {
            "category": category,
            "confidence": confidence,
            "angle": angle,
            "keypoint_corr": keypoint_corr
        }
        return dict_res

    elif img_3:
        time.sleep(fall_detector.min_time_between_frames)
        process_response(fall_detector.process_sample(image=img_3))

        if len(result) == 1:
            category = result[0]['label']
            confidence = result[0]['confidence']
            angle = result[0]['leaning_angle']
            keypoint_corr = result[0]['keypoint_corr']

            dict_res = {
                "category": category,
                "confidence": confidence,
                "angle": angle,
                "keypoint_corr": keypoint_corr
            }
            return dict_res

    return None

# Create submission.csv
def create_submission(test_dir, output_csv):
    test_images = [
        os.path.join(test_dir, f) for f in os.listdir(test_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    predictions = []
    for img_path in test_images:
        # Placeholder: Replace with actual prediction logic
        prediction = Fall_prediction(img_path, img_path)  # Use the same image twice for demonstration
        if prediction:
            label = 1 if prediction["category"] == "fall" else 0
        else:
            label = 0

        predictions.append({"id": os.path.basename(img_path), "label": label})

    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)
    print(f"Submission file created at {output_csv}")

# Example usage
if __name__ == "__main__":
    dataset_dir = "./dataset/train"
    processed_dir = "./processed_split"
    test_dir = "./dataset/test"
    output_csv = "./submission.csv"

    preprocess_dataset(dataset_dir, processed_dir)
    create_submission(test_dir, output_csv)
