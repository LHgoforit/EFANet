# src/scripts/preprocess_faces.py
# ------------------------------------------------------------------
# Preprocessing script for facial image datasets (CelebA, Helen, FFHQ).
# Aligns and resizes images to 256×256 using facial landmarks.
# Designed for EFANet training with consistent input resolution.
# ------------------------------------------------------------------

import os
import glob
import argparse
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
import dlib

from torchvision.transforms import functional as TF


def align_and_crop(image, detector, predictor, output_size=256):
    """
    Align and crop a face image using dlib's facial landmark predictor.

    Parameters
    ----------
    image : np.ndarray
        Input image in RGB format.
    detector : dlib.face_detector
        Face detector.
    predictor : dlib.shape_predictor
        Landmark predictor.
    output_size : int
        Size of the output cropped image.

    Returns
    -------
    PIL.Image
        Aligned and resized face image.
    """
    dets = detector(image, 1)
    if len(dets) == 0:
        return None

    shape = predictor(image, dets[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])

    # Use eyes and mouth corners for similarity transform
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)
    mouth_center = np.mean(landmarks[48:68], axis=0)
    center = (left_eye + right_eye + mouth_center) / 3

    angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    M = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
    aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

    face_cx, face_cy = int(center[0]), int(center[1])
    half_size = output_size // 2
    cropped = aligned[
        max(0, face_cy - half_size):face_cy + half_size,
        max(0, face_cx - half_size):face_cx + half_size,
    ]

    if cropped.shape[0] != output_size or cropped.shape[1] != output_size:
        return None

    return Image.fromarray(cropped)


def preprocess_directory(input_dir, output_dir, detector, predictor, output_size=256):
    """
    Process all images in a directory and save aligned outputs.

    Parameters
    ----------
    input_dir : str
        Directory containing raw images.
    output_dir : str
        Directory to save processed images.
    detector : dlib.face_detector
        Dlib face detector.
    predictor : dlib.shape_predictor
        Dlib shape predictor.
    output_size : int
        Target size for output images.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(input_dir, '*.jpg')) +
                         glob.glob(os.path.join(input_dir, '*.png')))

    for path in tqdm(image_paths, desc=f"Processing {input_dir}"):
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        result = align_and_crop(image, detector, predictor, output_size)
        if result is not None:
            out_path = os.path.join(output_dir, os.path.basename(path))
            result.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Path to raw image folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save aligned images')
    parser.add_argument('--landmark_model', type=str, default='shape_predictor_68_face_landmarks.dat',
                        help='Path to dlib landmark model')
    parser.add_argument('--output_size', type=int, default=256, help='Output image size')
    args = parser.parse_args()

    if not os.path.exists(args.landmark_model):
        raise FileNotFoundError(f"Dlib landmark model not found: {args.landmark_model}")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.landmark_model)

    preprocess_directory(args.input_dir, args.output_dir, detector, predictor, args.output_size)


if __name__ == '__main__':
    main()
