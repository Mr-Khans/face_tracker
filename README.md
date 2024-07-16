# face_tracker
face tracker and crop cut

Python 3.10.12

create venv:
python -m venv venv
activate:
source venv/bin/activate


pip install -r requirements.txt

requirements.txt


opencv-python-headless==4.10.0.84
dlib==19.24.4
face-recognition==1.3.0
face-recognition-models==0.3.0
onnxruntime-gpu==1.18.1
numpy==1.25.2
scipy==1.11.4

Test result_1:
CPU
Total processed frames: 385
Average time per frame: 0.0287 seconds
Total processing time: 11.0584 seconds
GPU
Total processed frames: 385
Average time per frame: 0.0278 seconds
Total processing time: 10.6859 seconds

