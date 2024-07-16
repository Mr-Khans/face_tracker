# face_tracker
face tracker and crop cut

## Python 3.10.12

create venv:

python -m venv venv

activate:

source venv/bin/activate


pip install -r requirements.txt

## requirements.txt


opencv-python-headless==4.10.0.84

dlib==19.24.4

face-recognition==1.3.0

face-recognition-models==0.3.0

onnxruntime-gpu==1.18.1

numpy==1.25.2

scipy==1.11.4


## Test result_1:

CPU

Total processed frames: 385

Average time per frame: 0.0287 seconds

Total processing time: 11.0584 seconds

GPU

Total processed frames: 385

Average time per frame: 0.0278 seconds

Total processing time: 10.6859 seconds


## RUN

python face_processing.py video.mp4 reference.jpg output_dir --frame_skip 2 --gpu 0 --padding 0.3

# Model

wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
bzip2 -dk dlib_face_recognition_resnet_model_v1.dat.bz2


## Brief Report on the Work Done
Used Python Version
Python 3.10.12
Libraries Used and Motivation for Selection
opencv-python-headless: OpenCV library is used for image and video processing. The "headless" version is chosen for resource efficiency, as it does not require a graphical interface.
dlib: The dlib library is used for face detection and recognition on images. It provides efficient algorithms and models for these tasks.
face_recognition: The face_recognition library is built on top of dlib and provides a high-level interface for face recognition. It simplifies the use of dlib's functionality.
onnxruntime-gpu: The onnxruntime-gpu library is used for accelerating computations using a GPU. It supports models in the ONNX format, enabling the use of pre-trained machine learning models.
numpy: The numpy library is used for efficient work with multi-dimensional arrays and matrices. It is necessary for image data processing.
scipy: The scipy library is used for scientific computations and data processing. It may be used for additional image processing or computations in this project.
Problems Encountered During Task Resolution
Padding Selection: Problems arose with selecting the correct padding when detecting faces. To resolve this issue, automatic padding selection based on face sizes was implemented.
ONNX and DLIB Compatibility: Using ONNX for acceleration computations led to compatibility issues with the dlib library. The current version of ONNX does not have a realization for the used version of dlib, which may lead to errors or suboptimal performance.
Personal Comments
The project was interesting and allowed me to apply knowledge in the fields of image processing and machine learning. The use of dlib and face_recognition libraries simplified the implementation of main face recognition functions. However, the compatibility issue with ONNX and DLIB may require further research and the search for alternative solutions for accelerating computations.

python main.py path/to/video.mp4 path/to/Screenshot_1.png path/to/output_directory --frame_skip 1 --gpu 0

