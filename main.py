import time
import os
import cv2
import dlib
import numpy as np
import face_recognition
import onnxruntime as ort
from scipy.spatial import distance
import json
from typing import List, Tuple, Dict, Any
import argparse
import time
from datetime import datetime

os.makedirs("output_directory", exist_ok=True)


# Initialize dlib's face detector and the landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

def load_reference_image(reference_image_path: str, face_rec_model, shape_predictor) -> np.ndarray:
    try:
        reference_image = dlib.load_rgb_image(reference_image_path)
    except Exception as e:
        print(f"Error loading reference image: {e}")
        exit(1)

    face_locations = dlib.get_frontal_face_detector()(reference_image)
    if len(face_locations) == 0:
        print("No faces found in the reference image.")
        exit(1)

    reference_encoding = None
    for face_location in face_locations:
        shape = shape_predictor(reference_image, face_location)
        reference_encoding = np.array(face_rec_model.compute_face_descriptor(reference_image, shape))

    return reference_encoding

def process_frame(frame: np.ndarray, frame_number: int, reference_encoding: np.ndarray, output_dir: str, detector, predictor, face_rec_model, padding: float) -> List[Dict[str, Any]]:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_frame)

    metadata = []

    for face in faces:
        shape = predictor(rgb_frame, face)
        face_encoding = np.array(face_rec_model.compute_face_descriptor(rgb_frame, shape))

        if distance.euclidean(reference_encoding, face_encoding) < 0.6:
            landmarks_points = [(p.x, p.y) for p in shape.parts()]
            aligned_face, transform_matrix = align_face(frame, face, shape, padding)
            if aligned_face is not None:
                save_aligned_face(aligned_face, frame_number, output_dir)
                clarity_score = compute_clarity_score(aligned_face)
                rotation_angle = compute_rotation_angle(transform_matrix)

                metadata.append({
                    "frame_number": frame_number,
                    "bbox": [face.left(), face.top(), face.right(), face.bottom()],
                    "landmarks": landmarks_points,
                    "transform_matrix": transform_matrix.tolist(),
                    "clarity_score": clarity_score,
                    "rotation_angle": rotation_angle
                })

    return metadata

def align_face(image: np.ndarray, face: dlib.rectangle, shape: dlib.full_object_detection, padding: float) -> Tuple[np.ndarray, np.ndarray]:
    eye_left = np.mean([[p.x, p.y] for p in shape.parts()[36:42]], axis=0)
    eye_right = np.mean([[p.x, p.y] for p in shape.parts()[42:48]], axis=0)
    eyes_center = (eye_left + eye_right) / 2.0

    dy = eye_right[1] - eye_left[1]
    dx = eye_right[0] - eye_left[0]
    angle = np.degrees(np.arctan2(dy, dx))
    scale = 1

    M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale)
    aligned_face = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

    left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()

    height, width = image.shape[:2]
    pad_x = int((right - left) * padding)
    pad_y = int((bottom - top) * padding)

    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(width, right + pad_x)
    bottom = min(height, bottom + pad_y)

    bbox_face = aligned_face[top:bottom, left:right]
    if bbox_face.shape[0] == 0 or bbox_face.shape[1] == 0:
        return None, None

    return bbox_face, M

def save_aligned_face(face: np.ndarray, frame_number: int, output_dir: str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"aligned_face_{frame_number}.jpg")
    cv2.imwrite(filename, face)

def compute_clarity_score(face: np.ndarray) -> float:
    return np.mean(cv2.Laplacian(face, cv2.CV_64F))

def compute_rotation_angle(transform_matrix: np.ndarray) -> float:
    angle_rad = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
    return np.degrees(angle_rad)

def main(video_path: str, reference_image_path: str, output_dir: str, frame_skip: int = 1, gpu: int = -1, padding: float = 0.25) -> None:
    if not os.path.isfile(video_path):
        print(f"Error: video file '{video_path}' not found.")
        exit(1)

    if not os.path.isfile(reference_image_path):
        print(f"Error: reference image file '{reference_image_path}' not found.")
        exit(1)

    device = 'cuda' if gpu >= 0 and dlib.cuda.get_num_devices() > 0 else 'cpu'
    print(f'Device: {device}')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

    reference_encoding = load_reference_image(reference_image_path, face_rec_model, predictor)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_dir, video_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    processed_frames = 0
    total_time = 0
    metadata = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_skip == 0:
            start_time = time.time()
            frame_metadata = process_frame(frame, frame_number, reference_encoding, output_dir, detector, predictor, face_rec_model, padding)
            processing_time = time.time() - start_time

            total_time += processing_time
            processed_frames += 1

            print(f"Processed frame {frame_number} in {processing_time:.4f} seconds")

            metadata.extend(frame_metadata)

        frame_number += 1

    cap.release()

    metadata_sorted = sorted(metadata, key=lambda x: (-x['clarity_score'], x['rotation_angle']))

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata_sorted, f, indent=4)

    if processed_frames > 0:
        avg_time_per_frame = total_time / processed_frames
        print(f"Total processed frames: {processed_frames}")
        print(f"Average time per frame: {avg_time_per_frame:.4f} seconds")
        print(f"Total processing time: {total_time:.4f} seconds")   
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face alignment and metadata extraction from video")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("reference_image_path", type=str, help="Path to the reference image")
    parser.add_argument("output_dir", type=str, help="Directory to save the output results")
    parser.add_argument("--frame_skip", type=int, default=1, help="Number of frames to skip between processing")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index to use, -1 for CPU")
    parser.add_argument("--padding", type=float, default=0.25, help="Padding around the detected face for cropping")

    args = parser.parse_args()

    main(args.video_path, args.reference_image_path, args.output_dir, args.frame_skip, args.gpu, args.padding)