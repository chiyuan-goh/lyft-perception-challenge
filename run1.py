import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
from keras.models import load_model
import cv2

file = sys.argv[-1]

if file == 'run1.py':
    print("Error loading video")
    quit


# Define encoder function
def encode(array):
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


video = skvideo.io.vread(file)

answer_key = {}

# Frame numbering starts at 1
frame = 1
model = load_model("m1")


for rgb_frame in video:
    cv2_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    cv2_frame = cv2_frame.reshape((1,600, 800, 3))
    prediction = model.predict(cv2_frame)

    prediction = prediction.argmax(axis=2).reshape((600, 800))
    car_img = np.where(prediction == 0, 1, 0).astype('uint8')
    road_img = np.where(prediction == 1, 1, 0).astype('uint8')

    answer_key[frame] = [encode(car_img), encode(road_img)]
    frame += 1

# Print output in proper json format
print(json.dumps(answer_key))