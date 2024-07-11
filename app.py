import os
import cv2
import dlib
import numpy as np
import face_recognition
import onnxruntime as ort
from scipy.spatial import distance
import json
from typing import List, Tuple, Dict, Any


# Initialize dlib's face detector and the landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the reference image and encode the target face
reference_image_path = "path/to/reference/image.jpg"
reference_image = face_recognition.load_image_file(reference_image_path)
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# Initialize ONNX runtime session for GPU processing
gpu_available = ort.get_device().upper() != "CPU"
sess = ort.InferenceSession("model.onnx", providers=['CUDAExecutionProvider'] if gpu_available else ['CPUExecutionProvider'])

def process_frame(frame: np.ndarray, frame_number: int, output_dir: str) -> List[Dict[str, Any]]:
    """
    Processes a single frame to detect and align faces, and save the aligned faces and metadata.

    Args:
        frame (np.ndarray): The video frame to process.
        frame_number (int): The current frame number.
        output_dir (str): Directory to save the aligned faces and metadata.

    Returns:
        List[Dict[str, Any]]: Metadata for detected and aligned faces.
    """
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
    """
    Aligns the face in the image based on the provided landmarks points.

    Args:
        image (np.ndarray): The original image containing the face.
        landmarks_points (List[Tuple[int, int]]): The landmark points of the face.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The aligned face image and the transformation matrix.
    """
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
    """
    Saves the aligned face image to the specified directory.

    Args:
        face (np.ndarray): The aligned face image.
        frame_number (int): The frame number from which the face was extracted.
        output_dir (str): The directory to save the aligned face image.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"aligned_face_{frame_number}.jpg")
    cv2.imwrite(filename, face)

def main(video_path: str, reference_image_path: str, output_dir: str, frame_skip: int = 1, gpu: int = 0) -> None:
    """
    Main function to process the video, detect and align target faces, and save the results.

    Args:
        video_path (str): Path to the video file.
        reference_image_path (str): Path to the reference image of the target actor.
        output_dir (str): Directory to save the aligned faces and metadata.
        frame_skip (int): Number of frames to skip during processing (default is 1).
        gpu (int): GPU ID to use for processing. Set to -1 for CPU only (default is 0).
    """
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    metadata = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_skip == 0:
            frame_metadata = process_frame(frame, frame_number, output_dir)
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
