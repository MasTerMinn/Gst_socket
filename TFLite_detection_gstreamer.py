import os
import argparse
import cv2
import numpy as np
import time
import importlib.util
import socket
import json
from threading import Thread

def create_socket():
    host = '192.168.1.4'  # Replace with the IP address of the receiver machine
    port = 6000  # You can choose any port, but it must be the same on both sides
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Use UDP for simplicity
    return client_socket, host, port

# Send data to another computer
def send_bbox(client_socket, host, port, bbox_data):
    try:
        # Convert numpy data types (float32) to native Python types (float)
        # Recursively convert all elements in bbox_data to native Python types (float, int, str, etc.)
        def convert_np_data(obj):
            if isinstance(obj, np.generic):  # Check if the object is a numpy type
                return obj.item()  # Convert numpy type to native Python type (e.g., float32 -> float)
            elif isinstance(obj, dict):  # If it's a dictionary, recursively convert values
                return {key: convert_np_data(value) for key, value in obj.items()}
            elif isinstance(obj, list):  # If it's a list, recursively convert items
                return [convert_np_data(item) for item in obj]
            else:
                return obj  # If it's a native Python type, return it as is

        # Convert the bounding box data to native Python types
        bbox_data_native = convert_np_data(bbox_data)

        # Convert to JSON format
        message = json.dumps(bbox_data_native)  # Now it's serializable
        client_socket.sendto(message.encode(), (host, port))
        print("Sent bounding box data to receiver.")
    except Exception as e:
        print(f"Error sending data: {e}")




class VideoStream:
    def __init__(self, resolution=(1280, 720), framerate=30):
        self.stream = cv2.VideoCapture(0)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', required=True)
parser.add_argument('--graph', default='detect.tflite')
parser.add_argument('--labels', default='labelmap.txt')
parser.add_argument('--threshold', default=0.5)
parser.add_argument('--resolution', default='1280x720')
parser.add_argument('--edgetpu', action='store_true')
args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = map(int, args.resolution.split('x'))
use_TPU = args.edgetpu

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

if use_TPU and GRAPH_NAME == 'detect.tflite':
    GRAPH_NAME = 'edgetpu.tflite'

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == '???':
    del(labels[0])

if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

outname = output_details[0]['name']
if 'StatefulPartitionedCall' in outname:
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

frame_rate_calc = 1
freq = cv2.getTickFrequency()
videostream = VideoStream(resolution=(resW, resH), framerate=30).start()
time.sleep(1)

# GStreamer pipeline (replace host=0.0.0.0 with Coral IP for broadcasting)
gst_str = (
    'appsrc ! videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency '
    '! rtph264pay config-interval=1 pt=96 ! udpsink host=PC_IP port=5000'
)
# Replace PC_IP with actual IP of PC receiver
gst_str = gst_str.replace("PC_IP", "192.168.1.4")  # <- your PC's IP address

out = cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, 30.0, (resW, resH), True)
if not out.isOpened():
    print("Failed to open GStreamer pipeline")
    exit()

# Create socket for sending data
client_socket, host, port = create_socket()

while True:
    t1 = cv2.getTickCount()
    frame1 = videostream.read()
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            ymin = int(max(1, (boxes[i][0] * resH)))
            xmin = int(max(1, (boxes[i][1] * resW)))
            ymax = int(min(resH, (boxes[i][2] * resH)))
            xmax = int(min(resW, (boxes[i][3] * resW)))
            
            # Prepare data to send
            bbox_data = {
                'label': labels[int(classes[i])],
                'confidence': scores[i],
                'bounding_box': {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
            }
            
            # Send the data to the receiver machine
            send_bbox(client_socket, host, port, bbox_data)
            
            

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10),
                          (xmin+labelSize[0], label_ymin+baseLine-10), (255,255,255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1
    cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc),
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videostream.stop()
out.release()
cv2.destroyAllWindows()

