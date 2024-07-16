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

# Initialize dlib's face detector and the landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def load_reference_image(reference_image_path: str) -> np.ndarray:
    reference_image = face_recognition.load_image_file(reference_image_path)
    reference_encoding = face_recognition.face_encodings(reference_image)[0]
    return reference_encoding

def init_onnx_session(gpu: int) -> ort.InferenceSession:
    providers = ['CUDAExecutionProvider'] if gpu >= 0 else ['CPUExecutionProvider']
    sess = ort.InferenceSession("model.onnx", providers=providers)
    return sess

def process_frame(frame: np.ndarray, frame_number: int, reference_encoding: np.ndarray, output_dir: str) -> List[Dict[str, Any]]:
    rgb_frame = frame[:, :, ::-1]
    faces = detector(rgb_frame)

    metadata = []

    for face in faces:
        landmarks = predictor(rgb_frame, face)
        landmarks_points = [(p.x, p.y) for p in landmarks.parts()]

        face_descriptor = face_recognition.face_encodings(rgb_frame, [face])[0]
        if distance.euclidean(reference_encoding, face_descriptor) < 0.6:
            aligned_face, transform_matrix = align_face(frame, landmarks_points)
            save_aligned_face(aligned_face, frame_number, output_dir)
            metadata.append({
                "frame_number": frame_number,
                "bbox": [face.left(), face.top(), face.right(), face.bottom()],
                "landmarks": landmarks_points,
                "transform_matrix": transform_matrix.tolist()
            })

    return metadata

def align_face(image: np.ndarray, landmarks_points: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    eye_left = np.mean(landmarks_points[36:42], axis=0)
    eye_right = np.mean(landmarks_points[42:48], axis=0)
    nose = landmarks_points[30]

    eyes_center = (eye_left + eye_right) / 2.0

    dy = eye_right[1] - eye_left[1]
    dx = eye_right[0] - eye_left[0]
    angle = np.degrees(np.arctan2(dy, dx))
    scale = 1

    M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale)
    aligned_face = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
    return aligned_face, M

def save_aligned_face(face: np.ndarray, frame_number: int, output_dir: str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"aligned_face_{frame_number}.jpg")
    cv2.imwrite(filename, face)

def main(video_path: str, reference_image_path: str, output_dir: str, frame_skip: int = 1, gpu: int = 0) -> None:
    reference_encoding = load_reference_image(reference_image_path)
    #sess = init_onnx_session(gpu)

    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    metadata = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_skip == 0:
            frame_metadata = process_frame(frame, frame_number, reference_encoding, output_dir)
            metadata.extend(frame_metadata)

        frame_number += 1

    cap.release()

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract and align target face from video.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument("reference_image_path", type=str, help="Path to the reference image of the target actor.")
    parser.add_argument("output_dir", type=str, help="Directory to save the aligned faces and metadata.")
    parser.add_argument("--frame_skip", type=int, default=1, help="Number of frames to skip during processing.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use for processing. Set to -1 for CPU only.")

    args = parser.parse_args()
    main(args.video_path, args.reference_image_path, args.output_dir, args.frame_skip, args.gpu)
