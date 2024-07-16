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

#Model

wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
bzip2 -dk dlib_face_recognition_resnet_model_v1.dat.bz2


